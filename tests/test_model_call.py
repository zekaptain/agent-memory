"""A model call: which attempt follows which, what each server is sent, and what is billed."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from agent_memory import endpoints, model_call
from agent_memory.model_call import (
    Ask,
    CutOff,
    LlamaCpp,
    ModelUnavailable,
    ModelUnreachable,
    NoJson,
    Ollama,
    OpenRouter,
    Reply,
    Schema,
    ThinkingRefused,
    json_object,
)

GOOD = '{"answer": "no"}'
ASK = Ask(prompt="P", budget=100, think=True)


def reply(text: str = GOOD, *, cut_off: bool = False, thinking: str = "", **more: Any) -> Reply:
    return Reply(
        text=text, thinking=thinking, cut_off=cut_off, snapshot="m:1", latency_ms=10.0, **more
    )


class FakeServer:
    """A server that answers from a script and remembers what it was asked."""

    kind = "fake"
    model = "m"
    local = True

    def __init__(self, *script: Reply | Exception | Callable[[Ask], Reply]) -> None:
        self.script = list(script)
        self.asked: list[Ask] = []

    def send(self, ask: Ask) -> Reply:
        self.asked.append(ask)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return step(ask) if callable(step) else step

    def seen(self, field: str) -> list[Any]:
        return [getattr(ask, field) for ask in self.asked]


def judge_like(server: model_call.Server, ask: Ask = ASK) -> model_call.Result:
    return model_call.call(server, ask, who="judge", wider=300, drop_thinking=True)


# The attempts


def test_a_usable_reply_is_one_attempt():
    server = FakeServer(reply())
    result = judge_like(server)
    assert result.data == {"answer": "no"}
    assert server.asked == [ASK]
    assert (result.thinking_dropped, result.latency_ms, result.snapshot) == (False, 10.0, "m:1")
    assert result.shape_enforced


def test_a_cut_off_reply_gets_room_and_keeps_thinking():
    server = FakeServer(reply("{", cut_off=True), reply())
    result = judge_like(server)
    assert server.seen("budget") == [100, 300]
    assert server.seen("think") == [True, True]
    assert server.seen("shape") == ["json", "json"]
    assert not result.thinking_dropped
    assert result.latency_ms == 20.0  # both attempts billed


def test_still_cut_off_with_room_the_last_attempt_drops_thinking_and_the_grammar():
    server = FakeServer(reply("{", cut_off=True), reply("", cut_off=True), reply())
    result = judge_like(server)
    # Off is said outright: a server that thinks by default would think again otherwise.
    assert server.seen("think") == [True, True, False]
    assert server.seen("budget") == [100, 300, 100]
    assert server.seen("shape") == ["json", "json", None]
    assert result.thinking_dropped
    assert not result.shape_enforced
    assert result.latency_ms == 30.0


def test_a_reply_with_no_json_object_skips_the_wider_budget():
    server = FakeServer(reply(""), reply())
    result = judge_like(server)
    assert server.seen("budget") == [100, 100]
    assert server.seen("think") == [True, False]
    assert result.thinking_dropped


def test_a_refused_thinking_switch_is_left_out_and_the_refusal_is_billed():
    refusal = ThinkingRefused("m does not support thinking", 400, latency_ms=50.0)
    server = FakeServer(refusal, reply())
    result = judge_like(server)
    assert server.seen("think") == [True, None]
    assert server.seen("shape") == ["json", None]
    assert result.thinking_dropped
    assert result.latency_ms == 60.0

    # Refused when told off, the last attempt goes out again with nothing said.
    server = FakeServer(reply(""), refusal, reply())
    assert judge_like(server).latency_ms == 70.0
    assert server.seen("think") == [True, False, None]
    # With nothing said there is no switch left to refuse.
    server = FakeServer(refusal)
    with pytest.raises(ModelUnavailable, match="does not support thinking"):
        judge_like(server)
    assert server.seen("think") == [True, None]


def test_what_happened_chooses_the_next_attempt_not_the_words_in_a_reply():
    """A failure whose text mentioned "http" used to make the last attempt leave the
    thinking switch out, so a model that thinks by default thought again."""
    server = FakeServer(reply("See https://example.test/docs first."), reply())
    assert judge_like(server).thinking_dropped
    assert server.seen("think") == [True, False]

    tail = '{"answer": "see http://x.test'
    server = FakeServer(reply(tail, cut_off=True), reply(tail, cut_off=True), reply())
    judge_like(server)
    assert server.seen("think") == [True, True, False]


@pytest.mark.parametrize(
    "fault",
    [
        endpoints.HttpStatusError("u: HTTP 404: model 'qwen3-thinking:8b' not found", 404),
        endpoints.HttpStatusError("u: HTTP 500: unexpected end of JSON input", 500),
        endpoints.HttpStatusError("u: HTTP 400: context too long", 400),
        endpoints.EgressError("http://u/api/chat: response is not JSON"),
        endpoints.EgressError("http://thinkpad.local: timed out after 300s"),
    ],
)
def test_a_fault_that_another_attempt_cannot_change_ends_the_call(fault):
    server = FakeServer(fault)
    with pytest.raises(ModelUnavailable, match=str(fault).split(":")[-1].strip()) as raised:
        judge_like(server)
    assert len(server.asked) == 1
    # The server was reached: the fault may be the doing of what was asked.
    assert not isinstance(raised.value, ModelUnreachable)


def test_a_server_that_was_not_reached_is_told_apart():
    server = FakeServer(endpoints.Unreachable("http://u: <urlopen error Connection refused>"))
    with pytest.raises(ModelUnreachable, match="Connection refused"):
        judge_like(server)
    assert len(server.asked) == 1


def test_a_fault_on_a_later_attempt_ends_the_call_too():
    server = FakeServer(reply("{", cut_off=True), endpoints.EgressError("u: timed out after 300s"))
    with pytest.raises(ModelUnavailable, match="timed out"):
        judge_like(server)
    assert len(server.asked) == 2


def test_the_error_is_about_the_last_reply():
    looping = reply("", cut_off=True, thinking="again? " * 10)
    server = FakeServer(looping)
    with pytest.raises(CutOff, match=r"judge ran out of tokens after 70 chars of thinking"):
        judge_like(server)
    assert len(server.asked) == 3

    server = FakeServer(reply("", cut_off=True), reply("", cut_off=True), reply("not json"))
    with pytest.raises(NoJson, match="judge returned no JSON: 'not json'"):
        judge_like(server)


def test_a_caller_chooses_which_later_attempts_apply():
    # One attempt: nothing to fall back to.
    server = FakeServer(reply("", cut_off=True))
    with pytest.raises(CutOff, match="model ran out of tokens; JSON incomplete: ''"):
        model_call.call(server, ASK)
    assert len(server.asked) == 1

    # Room only, as a drafter asks: thinking is never touched.
    draft = Ask(prompt="P", budget=100, think=False, seed=7)
    server = FakeServer(reply('{"lesson"', cut_off=True, thinking="x" * 300))
    with pytest.raises(CutOff, match=r"drafter ran out of tokens after 300 chars of thinking"):
        model_call.call(server, draft, who="drafter", wider=300)
    assert server.seen("budget") == [100, 300]
    assert server.seen("think") == [False, False]
    assert server.seen("seed") == [7, 7]

    server = FakeServer(reply(""))
    with pytest.raises(NoJson):
        model_call.call(server, draft, who="drafter", wider=300)
    assert len(server.asked) == 1

    # A refusal with no last attempt allowed is a fault like any other.
    server = FakeServer(ThinkingRefused("no", 400))
    with pytest.raises(ModelUnavailable):
        model_call.call(server, draft, who="drafter", wider=300)


def test_a_schema_the_server_turned_down_is_not_sent_again():
    schema = Schema("lesson", {"type": "object"})
    ask = Ask(prompt="P", budget=100, shape=schema)
    server = FakeServer(reply("{", cut_off=True, schema_dropped=True, cost_usd=0.001), reply())
    result = model_call.call(server, ask, wider=300)
    assert server.seen("shape") == [schema, "json"]
    assert not result.shape_enforced
    assert result.cost_usd == pytest.approx(0.001)


def test_json_object_is_found_in_prose_and_fences():
    assert json_object('Sure.\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert json_object('{"a": 1}') == {"a": 1}
    with pytest.raises(NoJson, match="judge returned no JSON: 'nothing here'"):
        json_object("nothing here", "judge")
    with pytest.raises(NoJson, match="unparsable JSON"):
        json_object('{"failed": {"answer": "yes"}, "category": {"answer": "wro')
    with pytest.raises(NoJson, match="not an object"):
        json_object("[1, 2]")
    # Only the start of a long reply is quoted.
    with pytest.raises(NoJson) as error:
        json_object("x" * 500)
    assert len(str(error.value)) < 260


# The servers


def transport(monkeypatch, *script: Any) -> list[dict[str, Any]]:
    """Replace the transport with a script of responses and errors; return what was sent."""
    steps = list(script)
    sent: list[dict[str, Any]] = []

    def post(url, body, headers=None, timeout=None):
        sent.append({"url": url, "headers": headers, "timeout": timeout, **copy.deepcopy(body)})
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, Exception):
            raise step
        return step, 10.0

    monkeypatch.setattr(endpoints, "post_json", post)
    return sent


def rejection(code: int, detail: str, latency_ms: float = 0.0) -> endpoints.HttpStatusError:
    return endpoints.HttpStatusError(
        f"http://server: HTTP {code}: {detail}", code, latency_ms=latency_ms, detail=detail
    )


def test_ollama_speaks_its_own_chat_format(monkeypatch):
    done = {"model": "m:1", "message": {"content": GOOD, "thinking": "hm"}, "done_reason": "stop"}
    sent = transport(monkeypatch, done)
    server = Ollama("m", host="http://ollama.test")
    assert (server.kind, server.model, server.local) == ("ollama", "m", True)
    schema = {"type": "object"}
    got = server.send(
        Ask(
            prompt="P",
            budget=700,
            system="S",
            temperature=0.2,
            seed=7,
            shape=Schema("lesson", schema),
            think=False,
        )
    )
    assert got == Reply(text=GOOD, thinking="hm", cut_off=False, snapshot="m:1", latency_ms=10.0)
    assert sent == [
        {
            "url": "http://ollama.test/api/chat",
            "headers": None,
            "timeout": endpoints.DEFAULT_TIMEOUT,
            "model": "m",
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0.2, "seed": 7, "num_predict": 700},
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}],
        }
    ]

    # Nothing said about thinking, the grammar, or a system message: the keys are left out.
    sent.clear()
    cut = server.send(Ask(prompt="P", budget=5, shape=None))
    assert set(sent[0]) == {"url", "headers", "timeout", "model", "stream", "options", "messages"}
    assert sent[0]["options"] == {"temperature": 0, "num_predict": 5}
    assert sent[0]["messages"] == [{"role": "user", "content": "P"}]
    assert not cut.cut_off
    assert Ollama("m").url == f"{endpoints.OLLAMA_HOST}/api/chat"

    transport(monkeypatch, {"message": {"content": "{"}, "done_reason": "length"})
    cut = server.send(Ask(prompt="P", budget=5))
    assert (cut.cut_off, cut.snapshot, cut.thinking) == (True, "m", "")
    transport(monkeypatch, {"message": "not an object"})
    assert server.send(Ask(prompt="P", budget=5)).text == ""


def test_ollama_tells_a_refused_thinking_switch_from_other_rejections(monkeypatch):
    server = Ollama("gemma3", host="http://thinkpad.local:11434")
    thinking = Ask(prompt="P", budget=5, think=True)

    transport(monkeypatch, rejection(400, '{"error":"\\"gemma3\\" does not support thinking"}', 50))
    with pytest.raises(ThinkingRefused) as error:
        server.send(thinking)
    assert (error.value.code, error.value.latency_ms) == (400, 50.0)

    # Only the server's own words about a 400 count: not the host, not a model's name in
    # another status, and not a request that said nothing about thinking.
    for fault in (
        rejection(400, "invalid options"),
        rejection(404, "model 'qwen3-thinking:8b' not found"),
        rejection(500, "thinking failed"),
    ):
        transport(monkeypatch, fault)
        with pytest.raises(endpoints.HttpStatusError) as error:
            server.send(thinking)
        assert not isinstance(error.value, ThinkingRefused)
    transport(monkeypatch, rejection(400, "does not support thinking"))
    with pytest.raises(endpoints.HttpStatusError) as error:
        server.send(Ask(prompt="P", budget=5))
    assert not isinstance(error.value, ThinkingRefused)


def test_a_refusal_by_ollama_costs_one_more_attempt_and_is_billed(monkeypatch):
    refused = rejection(400, '"gemma3" does not support thinking', latency_ms=50.0)
    sent = transport(monkeypatch, refused, {"message": {"content": GOOD}})
    result = judge_like(Ollama("gemma3"))
    assert [body.get("think", "absent") for body in sent] == [True, "absent"]
    assert [body.get("format", "absent") for body in sent] == ["json", "absent"]
    assert result.thinking_dropped
    assert result.latency_ms == 60.0


def test_llamacpp_speaks_openai_chat(monkeypatch):
    done = {
        "model": "served.gguf",
        "choices": [
            {"finish_reason": "stop", "message": {"content": GOOD, "reasoning_content": "hm"}}
        ],
    }
    sent = transport(monkeypatch, done)
    server = LlamaCpp("m", host="http://llama.test:8080", timeout=180, split_thinking=True)
    assert (server.kind, server.model, server.local) == ("llamacpp", "m", True)
    got = server.send(Ask(prompt="P", budget=4096, system="S", think=True))
    assert got == Reply(
        text=GOOD, thinking="hm", cut_off=False, snapshot="served.gguf", latency_ms=10.0
    )
    assert sent == [
        {
            "url": "http://llama.test:8080/v1/chat/completions",
            "headers": {},
            "timeout": 180,
            "model": "m",
            "stream": False,
            "temperature": 0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
            "reasoning_format": "deepseek",
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]

    # Off is said both ways, because which switch a template honours depends on the model.
    sent.clear()
    schema = Schema("lesson", {"type": "object"})
    LlamaCpp("m").send(Ask(prompt="P", budget=700, seed=7, shape=schema, think=False))
    body = sent[0]
    assert body["url"] == f"{endpoints.LLAMACPP_HOST}/v1/chat/completions"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "none"
    assert body["seed"] == 7
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "lesson", "schema": {"type": "object"}},
    }
    assert "reasoning_format" not in body

    # Nothing said about thinking or the grammar.
    sent.clear()
    LlamaCpp("m").send(Ask(prompt="P", budget=5, shape=None))
    assert not {"chat_template_kwargs", "reasoning_effort", "response_format"} & set(sent[0])

    # OpenRouter's name for thinking is read too; a cut-off reply says so.
    cut = {"choices": [{"finish_reason": "length", "message": {"content": "{", "reasoning": "r"}}]}
    transport(monkeypatch, cut)
    got = LlamaCpp("alias").send(Ask(prompt="P", budget=5))
    assert (got.cut_off, got.thinking, got.snapshot, got.cost_usd) == (True, "r", "alias", 0.0)
    for malformed in ({}, {"choices": []}, {"choices": ["x"]}, {"choices": [{"message": 3}]}):
        transport(monkeypatch, malformed)
        assert LlamaCpp("m").send(Ask(prompt="P", budget=5)).text == ""


def test_llamacpp_names_the_models_a_router_knows(monkeypatch):
    listed: list[str] = []
    listing: list[Any] = [{"data": [{"id": "org/b:Q8_0"}, {"id": "org/a:Q4"}, "junk"]}]

    def models(url, **kwargs):
        listed.append(url)
        if isinstance(listing[0], Exception):
            raise listing[0]
        return listing[0]

    monkeypatch.setattr(endpoints, "get_json", models)
    transport(monkeypatch, rejection(400, '{"error":{"message":"model \'typo\' not found"}}', 3))
    server = LlamaCpp("typo", host="http://r:9931")
    with pytest.raises(endpoints.HttpStatusError) as error:
        server.send(Ask(prompt="P", budget=5))
    assert str(error.value).endswith('not found"}}. Models this server knows: org/a:Q4, org/b:Q8_0')
    assert (error.value.code, error.value.latency_ms) == (400, 3.0)
    assert listed == ["http://r:9931/v1/models"]

    # If the listing itself fails, or names nothing, the first error stands.
    for unhelpful in (endpoints.EgressError("down"), {"data": []}, {"data": "junk"}):
        listing[0] = unhelpful
        with pytest.raises(endpoints.HttpStatusError) as error:
            server.send(Ask(prompt="P", budget=5))
        assert str(error.value).endswith('not found"}}')
    # Any other rejection is not about the model id.
    listed.clear()
    transport(monkeypatch, rejection(503, "Loading model"))
    with pytest.raises(endpoints.HttpStatusError, match="Loading model"):
        server.send(Ask(prompt="P", budget=5))
    assert listed == []


def openrouter_reply(content: str = GOOD, **usage: Any) -> dict[str, Any]:
    return {
        "model": "served",
        "choices": [{"message": {"content": content, "reasoning": "hm"}, "finish_reason": "stop"}],
        "usage": usage,
    }


def test_openrouter_needs_the_opt_in_and_a_key(monkeypatch):
    monkeypatch.delenv(endpoints.REMOTE_ENV, raising=False)
    monkeypatch.delenv(model_call.OPENROUTER_KEY_ENV, raising=False)
    sent = transport(monkeypatch, openrouter_reply(cost=0.0012))
    server = OpenRouter("openai/gpt-5")
    assert (server.kind, server.model, server.local) == ("openrouter", "openai/gpt-5", False)
    schema = Schema("lesson", {"type": "object"})
    ask = Ask(
        prompt="P", budget=700, system="S", temperature=0.2, seed=7, shape=schema, think=False
    )
    with pytest.raises(endpoints.Unreachable, match=endpoints.REMOTE_ENV):
        server.send(ask)
    monkeypatch.setenv(endpoints.REMOTE_ENV, "1")
    with pytest.raises(endpoints.Unreachable, match=model_call.OPENROUTER_KEY_ENV):
        server.send(ask)
    assert sent == []

    monkeypatch.setenv(model_call.OPENROUTER_KEY_ENV, "sk-test")
    got = server.send(ask)
    assert got == Reply(
        text=GOOD, thinking="hm", cut_off=False, snapshot="served", latency_ms=10.0, cost_usd=0.0012
    )
    assert sent == [
        {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {"Authorization": "Bearer sk-test"},
            "timeout": endpoints.DEFAULT_TIMEOUT,
            "model": "openai/gpt-5",
            "stream": False,
            "temperature": 0.2,
            "seed": 7,
            "max_tokens": 700,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "lesson", "schema": {"type": "object"}, "strict": True},
            },
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}],
            # The gateway reports its charge when asked. It has no thinking switch.
            "usage": {"include": True},
        }
    ]
    assert OpenRouter("m", api_key="sk-own").send(ask).cost_usd == pytest.approx(0.0012)
    assert sent[-1]["headers"] == {"Authorization": "Bearer sk-own"}

    transport(monkeypatch, openrouter_reply(cost=["free"]))
    with pytest.raises(endpoints.EgressError, match="malformed usage"):
        server.send(ask)
    # A charge that cannot be read is not recorded as none.
    transport(monkeypatch, {**openrouter_reply(), "usage": "none"})
    with pytest.raises(endpoints.EgressError, match="malformed usage"):
        server.send(ask)
    transport(monkeypatch, {**openrouter_reply(), "usage": None})
    assert server.send(ask).cost_usd == 0.0


@pytest.mark.parametrize("code", [400, 422])
def test_openrouter_sends_a_rejected_schema_once_more_as_plain_json(monkeypatch, code):
    monkeypatch.setenv(endpoints.REMOTE_ENV, "1")
    monkeypatch.setenv(model_call.OPENROUTER_KEY_ENV, "sk-test")
    schema = Schema("lesson", {"type": "object"})
    ask = Ask(prompt="P", budget=700, shape=schema)
    sent = transport(monkeypatch, rejection(code, "schema unsupported", 7.0), openrouter_reply())
    got = OpenRouter("m").send(ask)
    assert [body["response_format"]["type"] for body in sent] == ["json_schema", "json_object"]
    assert got.schema_dropped
    assert got.latency_ms == 17.0  # the rejected request is billed

    # Rejected again, the error stands: two requests, no more.
    sent = transport(monkeypatch, rejection(code, "still no"))
    with pytest.raises(endpoints.HttpStatusError, match="still no"):
        OpenRouter("m").send(ask)
    assert len(sent) == 2
    # With no schema to drop there is nothing to send differently.
    sent = transport(monkeypatch, rejection(code, "bad request"))
    with pytest.raises(endpoints.HttpStatusError):
        OpenRouter("m").send(Ask(prompt="P", budget=700))
    assert len(sent) == 1


@pytest.mark.parametrize("code", [401, 402, 429, 500, 503])
def test_openrouter_keeps_the_schema_for_faults_that_are_not_about_it(monkeypatch, code):
    monkeypatch.setenv(endpoints.REMOTE_ENV, "1")
    monkeypatch.setenv(model_call.OPENROUTER_KEY_ENV, "sk-test")
    sent = transport(monkeypatch, rejection(code, "rejected"))
    with pytest.raises(endpoints.HttpStatusError) as error:
        OpenRouter("m").send(Ask(prompt="P", budget=700, shape=Schema("lesson", {})))
    assert error.value.code == code
    assert len(sent) == 1


def test_http_rejection_keeps_what_the_server_said(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://localhost/test", 422, "bad schema", Message(), io.BytesIO(b"unsupported")
        )

    monkeypatch.setattr(endpoints.urllib.request, "urlopen", reject)
    with pytest.raises(endpoints.HttpStatusError, match="HTTP 422: unsupported") as error:
        endpoints.post_json("http://localhost/test", {})
    assert (error.value.code, error.value.detail) == (422, "unsupported")
    assert error.value.latency_ms > 0
