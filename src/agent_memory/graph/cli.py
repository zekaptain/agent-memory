"""`mem graph`: inspect the relation graph and run entity resolution. Registered on the main
app from cli.py."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

import typer

from .. import endpoints
from ..capture.session import Observed
from ..config import Config, load_config
from ..mcp_server.tools import MemoryTools
from .recall import relations_of
from .resolve import (
    BLOCK_SIZE,
    DEFAULT_MODEL,
    UNTYPED,
    Merge,
    OllamaResolver,
    Proposal,
    collect_entities,
    propose,
)
from .sql import MAX_HOPS

graph_app = typer.Typer(
    help="The relation graph: entities, neighbourhoods, and entity resolution.",
    no_args_is_help=True,
)

COMMAND = Observed("mem graph", "record", "mem_cli_v1")

SCOPES = Annotated[
    list[str] | None,
    typer.Option("--scope", help="Scope to read; repeatable. Default: global and --project."),
]
AGENT = Annotated[str, typer.Option(help="Agent recorded as the actor of these reads and writes.")]
PROJECT = Annotated[
    str | None, typer.Option(help="Project scope. Defaults to the working directory's name.")
]


def _config(ctx: typer.Context) -> Config:
    opts = ctx.obj
    return load_config(home=getattr(opts, "home", None), port=getattr(opts, "port", None))


def _tools(ctx: typer.Context, agent: str, project: str | None) -> MemoryTools:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return MemoryTools(
        _config(ctx),
        agent,
        harness_session_id=f"graph-{stamp}",
        project=project,
        observed=COMMAND,
    )


def _listing(tools: MemoryTools, scope: list[str] | None) -> dict[str, Any]:
    """Every relation in the scopes, read as a recorded recall."""
    result = tools.relations(scope)
    if result["degraded"]:
        raise typer.Exit(_fail(result["reason"]))
    return result


@graph_app.command()
def entities(
    ctx: typer.Context,
    scope: SCOPES = None,
    agent: AGENT = "human",
    project: PROJECT = None,
) -> None:
    """Every entity in the active relations, with its type and mention count. Logged as a
    graph_relations_v1 retrieval."""
    tools = _tools(ctx, agent, project)
    try:
        listing = _listing(tools, scope)
    finally:
        tools.close()
    found = collect_entities(relations_of(listing["items"]))
    if not found:
        print("no relations recorded in these scopes")
        return
    width = max(len(e.name) for e in found)
    for e in found:
        forms = f"  (also written: {', '.join(f for f in e.forms if f != e.name)})"
        print(
            f"{e.name:<{width}}  {e.entity_type or '-':<14} {e.mentions:>4}"
            + (forms if len(e.forms) > 1 else "")
        )
    print(f"\n{len(found)} entities; retrieval {listing['retrieval_id']}")


@graph_app.command()
def show(
    ctx: typer.Context,
    entity: Annotated[str, typer.Argument(help="Entity name; matching ignores case.")],
    hops: Annotated[int, typer.Option(min=1, max=MAX_HOPS)] = 2,
    scope: SCOPES = None,
    agent: AGENT = "human",
    project: PROJECT = None,
) -> None:
    """The relations within N hops of an entity. Logged as a graph_hop_v1 retrieval."""
    tools = _tools(ctx, agent, project)
    try:
        result = tools.graph(entity, hops=hops, scopes=scope, limit=100)
    finally:
        tools.close()
    if result["degraded"]:
        raise typer.Exit(_fail(result["reason"]))
    for item in result["items"]:
        print(f"{item['depth']}  {item['triple']}  [{item['memory_id']}]")
    print(
        f"\n{len(result['items'])} of {result['candidates']} edges; "
        f"retrieval {result['retrieval_id']}"
    )


@graph_app.command()
def resolve(
    ctx: typer.Context,
    scope: SCOPES = None,
    entity_type: Annotated[
        str | None, typer.Option(help="Only cluster entities of this type.")
    ] = None,
    model: Annotated[
        str, typer.Option(help="Ollama model that proposes clusters.")
    ] = DEFAULT_MODEL,
    host: Annotated[str, typer.Option(help="Ollama host.")] = endpoints.OLLAMA_HOST,
    block_size: Annotated[int, typer.Option(min=2)] = BLOCK_SIZE,
    apply: Annotated[
        bool, typer.Option("--apply", help="Record the merges. Without it, only propose.")
    ] = False,
    agent: AGENT = "human",
    project: PROJECT = None,
) -> None:
    """Propose entity merges with a local model; --apply records them as supersedes history.
    Every relation the command reads is logged as a graph_relations_v1 retrieval, and each
    merge names it. The model is given the entities named in them, only those of
    --entity-type when it is set. A block of entities the model returns no clusters for is
    named and the command exits 1, after the merges from the other blocks are shown or
    recorded."""
    tools = _tools(ctx, agent, project)
    resolver = OllamaResolver(model, host=host)
    try:
        listing = _listing(tools, scope)
        relations = relations_of(listing["items"])
        if not relations:
            print("no relations recorded in these scopes")
            return
        proposal = propose(relations, resolver, entity_type=entity_type, block_size=block_size)
        if proposal.merges:
            _print_merges(proposal.merges)
            print(
                f"\nproposed from {len(relations)} relations; retrieval {listing['retrieval_id']}"
            )
            if not apply:
                print("proposal only; pass --apply to record these merges")
        elif not proposal.failed:
            print(f"{resolver.name} proposes no merges across {len(relations)} relations")
        for merge in proposal.merges if apply else []:
            result = tools.merge_entity(
                merge.alias,
                merge.canonical,
                f"entity resolution by {resolver.name}",
                scopes=scope,
                canonical_type=merge.entity_type,
                actor=f"{agent} via {resolver.name}",
                retrieval_id=listing["retrieval_id"],
            )
            if result["degraded"]:
                raise typer.Exit(_fail(result["reason"]))
            print(
                f"merged {merge.alias!r} into {merge.canonical!r}: "
                f"{len(result['superseded'])} superseded, {len(result['retired'])} retired"
            )
        if proposal.failed:
            raise typer.Exit(_fail_blocks(proposal))
    finally:
        tools.close()


def _print_merges(merges: list[Merge]) -> None:
    width = max(len(m.alias) for m in merges)
    for m in merges:
        suffix = f"  ({m.entity_type})" if m.entity_type else ""
        print(f"{m.alias:<{width}}  ->  {m.canonical}{suffix}")


def _fail_blocks(proposal: Proposal) -> int:
    for block in proposal.failed:
        kind = "" if block.entity_type == UNTYPED else f"{block.entity_type} "
        print(
            f"mem graph: could not cluster {len(block.names)} {kind}entities "
            f"({block.names[0]!r} to {block.names[-1]!r}): {block.reason}",
            flush=True,
        )
    if proposal.untried:
        left = "block was" if proposal.untried == 1 else "blocks were"
        print(f"mem graph: {proposal.untried} more {left} not tried", flush=True)
    return 1


def _fail(reason: str) -> int:
    print(f"mem graph: {reason}", flush=True)
    return 1
