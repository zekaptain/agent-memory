"""A model call: one question put to a model server, answered with a JSON object.

A call is one or more attempts. The first sends the request as asked. A model reply that was
cut off gets one more attempt with a wider budget, when the caller names one: the requests
are seeded or run at temperature 0, so the same budget would stop at the same place. A reply
that still cannot be used gets a last attempt with thinking off and the JSON grammar
dropped, when the caller allows it. So does a server that refuses the thinking switch, with
the switch left out. What happened chooses the next attempt, never the words of an error.

A fault another attempt cannot change ends the call at once: a server that cannot be
reached, a timeout, an HTTP rejection. The time and charge of a call are summed over every
attempt, a refused request included.

The servers differ only in what they are sent and how their reply reads: Ollama's native
chat, llama-server's OpenAI-compatible chat, and OpenRouter's hosted one. A caller supplies
the prompt, the budget, the shape of the reply, and what the object means.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from . import endpoints
from .db import AgentMemoryError

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
# OpenRouter passes a provider's rejection of the schema through as a 400; some providers
# say 422. Auth, credit, timeout, and rate-limit codes are not about the schema.
SCHEMA_REJECTED = frozenset({400, 422})


class ModelCallError(AgentMemoryError):
    """A model call ended without a JSON object."""


class ModelUnavailable(ModelCallError):
    """The server could not be reached, timed out, or rejected the request."""


class CutOff(ModelCallError):
    """The last model reply was cut off."""


class NoJson(ModelCallError):
    """The last model reply held no JSON object."""


class ThinkingRefused(endpoints.HttpStatusError):
    """The server rejected a request because it carried the thinking switch."""


@dataclass(frozen=True)
class Schema:
    """A JSON schema the reply must match, under the name a server files it by."""

    name: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class Ask:
    """What one attempt asks. `shape` is the grammar the reply is held to: any JSON object,
    a schema, or none. `think` switches thinking on or off; None says nothing about it and
    leaves the choice to the server."""

    prompt: str
    budget: int
    system: str | None = None
    temperature: float = 0
    seed: int | None = None
    shape: Schema | Literal["json"] | None = "json"
    think: bool | None = None


@dataclass(frozen=True)
class Reply:
    """A model reply: its text, the thinking before it, whether it was cut off, the model
    the server says it used, and what the attempt cost. `schema_dropped` says the server
    turned the schema down and answered as plain JSON."""

    text: str
    thinking: str
    cut_off: bool
    snapshot: str
    latency_ms: float
    cost_usd: float = 0.0
    schema_dropped: bool = False


@dataclass(frozen=True)
class Result:
    """What a model call came to: the object, and the cost of every attempt it took.
    `thinking_dropped` says the object came from the last attempt; `shape_enforced` that
    the reply it came from was held to the shape first asked for."""

    data: dict[str, Any]
    snapshot: str
    latency_ms: float
    cost_usd: float
    thinking_dropped: bool
    shape_enforced: bool


class Server(Protocol):
    """A model server: it sends one attempt and reads the model reply."""

    @property
    def kind(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def local(self) -> bool: ...
    def send(self, ask: Ask) -> Reply: ...


def call(
    server: Server,
    ask: Ask,
    *,
    who: str = "model",
    wider: int | None = None,
    drop_thinking: bool = False,
) -> Result:
    """Ask until a model reply holds a JSON object, or raise why none did.

    `wider` is the budget of the one further attempt a cut-off reply gets. `drop_thinking`
    allows the last attempt. `who` names the model's role in an error.
    """
    latency = cost = 0.0
    schema_dropped = False
    current, stage = ask, "first"
    while True:
        try:
            reply = server.send(current)
        except ThinkingRefused as refusal:
            # Only the switch was refused, so the question is worth asking without it. Off
            # cannot be said to this server, which is why the switch is left out.
            if not drop_thinking or current.think is None:
                raise ModelUnavailable(str(refusal)) from None
            latency += refusal.latency_ms
            current, stage = replace(ask, think=None, shape=None), "last"
            continue
        except endpoints.EgressError as fault:
            raise ModelUnavailable(str(fault)) from None
        latency += reply.latency_ms
        cost += reply.cost_usd
        schema_dropped = schema_dropped or reply.schema_dropped
        try:
            if reply.cut_off:
                raise CutOff(_out_of_tokens(reply, who))
            data = json_object(reply.text, who)
        except (CutOff, NoJson):
            if reply.cut_off and stage == "first" and wider is not None:
                # More room, and nothing else changed: thinking may be what the reply needs.
                shape = current.shape
                if schema_dropped and isinstance(shape, Schema):
                    shape = "json"
                current, stage = replace(current, budget=wider, shape=shape), "wider"
            elif drop_thinking and stage != "last":
                # Off is said outright, since a server that thinks by default would think
                # again. A grammar with no thinking channel is where Gemma-family models
                # emit whitespace until the budget is gone, so the reply is free text and
                # the object is searched for in it.
                current, stage = replace(ask, think=False, shape=None), "last"
            else:
                raise
            continue
        return Result(
            data=data,
            snapshot=reply.snapshot,
            latency_ms=latency,
            cost_usd=cost,
            thinking_dropped=stage == "last",
            shape_enforced=current.shape == ask.shape and not schema_dropped,
        )


def _out_of_tokens(reply: Reply, who: str) -> str:
    """Says where the budget went, so an empty reply after long thinking is not mistaken
    for a model that produced nothing."""
    where = f" after {len(reply.thinking)} chars of thinking" if reply.thinking else ""
    return f"{who} ran out of tokens{where}; JSON incomplete: {reply.text[-80:]!r}"


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def json_object(text: str, who: str = "model") -> dict[str, Any]:
    """The JSON object in a reply's text, which may sit inside prose or a code fence."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            raise NoJson(f"{who} returned no JSON: {text[:200]!r}") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise NoJson(f"{who} returned unparsable JSON ({exc.msg}): {text[:200]!r}") from None
    if not isinstance(data, dict):
        raise NoJson(f"{who} JSON is not an object")
    return data


def _messages(ask: Ask) -> list[dict[str, str]]:
    system = [{"role": "system", "content": ask.system}] if ask.system is not None else []
    return [*system, {"role": "user", "content": ask.prompt}]


@dataclass(frozen=True)
class Ollama:
    """Ollama's native chat. A model that can think does so unless told not to, so leaving
    the switch out is not off; a model that cannot think refuses a request that says on."""

    model: str
    host: str = endpoints.OLLAMA_HOST
    timeout: float = endpoints.DEFAULT_TIMEOUT
    kind = "ollama"
    local = True

    @property
    def url(self) -> str:
        return f"{self.host}/api/chat"

    def send(self, ask: Ask) -> Reply:
        body: dict[str, Any] = {"model": self.model, "stream": False}
        if ask.think is not None:
            body["think"] = ask.think
        if ask.shape is not None:
            body["format"] = ask.shape.schema if isinstance(ask.shape, Schema) else ask.shape
        options: dict[str, Any] = {"temperature": ask.temperature}
        if ask.seed is not None:
            options["seed"] = ask.seed
        # Bounds thinking plus the reply; without it a model can run past the timeout.
        options["num_predict"] = ask.budget
        body["options"] = options
        body["messages"] = _messages(ask)
        try:
            response, latency = endpoints.post_json(self.url, body, timeout=self.timeout)
        except endpoints.HttpStatusError as rejection:
            # Only the server's own words about a 400 count. The rest of the message holds
            # the host and the model's name, either of which can say "think".
            refused = (
                ask.think is not None
                and rejection.code == 400
                and "does not support thinking" in rejection.detail
            )
            if not refused:
                raise
            raise ThinkingRefused(
                str(rejection), 400, latency_ms=rejection.latency_ms, detail=rejection.detail
            ) from None
        message = response.get("message")
        message = message if isinstance(message, dict) else {}
        return Reply(
            text=str(message.get("content") or ""),
            thinking=str(message.get("thinking") or ""),
            cut_off=response.get("done_reason") == "length",
            snapshot=str(response.get("model") or self.model),
            latency_ms=latency,
        )


def _openai_body(model: str, ask: Ask, *, strict: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "stream": False, "temperature": ask.temperature}
    if ask.seed is not None:
        body["seed"] = ask.seed
    # Thinking tokens share max_tokens with the visible reply.
    body["max_tokens"] = ask.budget
    if isinstance(ask.shape, Schema):
        schema: dict[str, Any] = {"name": ask.shape.name, "schema": ask.shape.schema}
        if strict:
            schema["strict"] = True
        body["response_format"] = {"type": "json_schema", "json_schema": schema}
    elif ask.shape is not None:
        body["response_format"] = {"type": "json_object"}
    body["messages"] = _messages(ask)
    return body


def _openai_reply(response: dict[str, Any], model: str, latency: float) -> Reply:
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    first = first if isinstance(first, dict) else {}
    message = first.get("message")
    message = message if isinstance(message, dict) else {}
    return Reply(
        text=str(message.get("content") or ""),
        # llama-server calls it reasoning_content, OpenRouter reasoning.
        thinking=str(message.get("reasoning_content") or message.get("reasoning") or ""),
        cut_off=first.get("finish_reason") == "length",
        snapshot=str(response.get("model") or model),
        latency_ms=latency,
    )


@dataclass(frozen=True)
class LlamaCpp:
    """llama-server's OpenAI-compatible chat. In single-model mode the server ignores the
    request's model and answers with the one it loaded; in router mode the name must be an
    id from /v1/models, loaded on first use. Either way the reply's own `model` is the
    snapshot. `split_thinking` asks for thinking apart from the text."""

    model: str
    host: str = endpoints.LLAMACPP_HOST
    timeout: float = endpoints.DEFAULT_TIMEOUT
    split_thinking: bool = False
    kind = "llamacpp"
    local = True

    @property
    def url(self) -> str:
        return f"{self.host}/v1/chat/completions"

    def send(self, ask: Ask) -> Reply:
        body = _openai_body(self.model, ask, strict=False)
        if self.split_thinking:
            # Parsed thoughts land in message.reasoning_content, leaving content as the text.
            body["reasoning_format"] = "deepseek"
        if ask.think is not None:
            # The template kwarg is how Qwen- and Gemma-style templates are switched;
            # templates without the switch ignore it and follow the server's --reasoning.
            body["chat_template_kwargs"] = {"enable_thinking": ask.think}
        if ask.think is False:
            # Off is said both ways, because which switch a template honours depends on
            # the model.
            body["reasoning_effort"] = "none"
        try:
            response, latency = endpoints.post_json(
                self.url, body, headers={}, timeout=self.timeout
            )
        except endpoints.HttpStatusError as rejection:
            raise self._naming_known_models(rejection) from None
        return _openai_reply(response, self.model, latency)

    def _naming_known_models(self, rejection: endpoints.HttpStatusError) -> endpoints.EgressError:
        """A router answers an unknown model id with a bare 'not found'; name the ids it
        serves so the spec can be corrected without a second trip to /v1/models."""
        message = str(rejection)
        if "not found" not in message.lower():
            return rejection
        try:
            listing = endpoints.get_json(f"{self.host}/v1/models")
        except endpoints.EgressError:
            return rejection
        listed = listing.get("data")
        ids = sorted(
            str(entry["id"])
            for entry in (listed if isinstance(listed, list) else [])
            if isinstance(entry, dict) and "id" in entry
        )
        if not ids:
            return rejection
        return endpoints.HttpStatusError(
            f"{message}. Models this server knows: {', '.join(ids)}",
            rejection.code,
            latency_ms=rejection.latency_ms,
            detail=rejection.detail,
        )


@dataclass(frozen=True)
class OpenRouter:
    """OpenRouter's hosted chat: session text leaves the machine, so it needs the egress
    opt-in and a key. It has no thinking switch. It reports its charge when asked, and a
    schema one of its providers rejects is sent once more as plain JSON."""

    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout: float = endpoints.DEFAULT_TIMEOUT
    kind = "openrouter"
    local = False
    url = OPENROUTER_URL

    def send(self, ask: Ask) -> Reply:
        endpoints.require_egress(self.url, "OpenRouter")
        key = self.api_key or os.environ.get(OPENROUTER_KEY_ENV)
        if not key:
            raise endpoints.EgressError(
                f"{OPENROUTER_KEY_ENV} is not set; OpenRouter needs an API key."
            )
        headers = {"Authorization": f"Bearer {key}"}
        body = _openai_body(self.model, ask, strict=True)
        body["usage"] = {"include": True}
        rejected_ms = 0.0
        schema_dropped = False
        try:
            response, latency = endpoints.post_json(
                self.url, body, headers=headers, timeout=self.timeout
            )
        except endpoints.HttpStatusError as rejection:
            if not isinstance(ask.shape, Schema) or rejection.code not in SCHEMA_REJECTED:
                raise
            body["response_format"] = {"type": "json_object"}
            schema_dropped = True
            rejected_ms = rejection.latency_ms
            response, latency = endpoints.post_json(
                self.url, body, headers=headers, timeout=self.timeout
            )
        return replace(
            _openai_reply(response, self.model, latency + rejected_ms),
            cost_usd=_charge(response),
            schema_dropped=schema_dropped,
        )


def _charge(response: dict[str, Any]) -> float:
    usage = response.get("usage")
    if usage is None:
        return 0.0
    try:
        return float(usage.get("cost") or 0.0)
    except (AttributeError, TypeError, ValueError):
        raise endpoints.EgressError("OpenRouter reply has a malformed usage object") from None
