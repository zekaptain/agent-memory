"""`mem graph`: inspect the relation graph and run entity resolution. Registered on the main
app from cli.py."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer

from ..config import Config, load_config
from ..mcp_server.tools import MemoryTools
from .resolve import (
    BLOCK_SIZE,
    DEFAULT_MODEL,
    OLLAMA_HOST,
    Merge,
    OllamaResolver,
    collect_entities,
    propose,
    relations_from_rows,
)
from .sql import MAX_HOPS, relations_sql

graph_app = typer.Typer(
    help="The relation graph: entities, neighbourhoods, and entity resolution.",
    no_args_is_help=True,
)

SCOPES = Annotated[
    list[str] | None,
    typer.Option("--scope", help="Scope to read; repeatable. Default: global and --project."),
]
AGENT = Annotated[
    str, typer.Option(help="Agent recorded as the actor of these reads and writes.")
]
PROJECT = Annotated[
    str | None, typer.Option(help="Project scope. Defaults to the working directory's name.")
]


def _config(ctx: typer.Context) -> Config:
    opts = ctx.obj
    return load_config(home=getattr(opts, "home", None), port=getattr(opts, "port", None))


def _tools(ctx: typer.Context, agent: str, project: str | None) -> MemoryTools:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return MemoryTools(
        _config(ctx), agent, harness_session_id=f"graph-{stamp}", project=project
    )


@graph_app.command()
def entities(
    ctx: typer.Context,
    scope: SCOPES = None,
    agent: AGENT = "human",
    project: PROJECT = None,
) -> None:
    """Every entity in the active relations, with its type and mention count."""
    tools = _tools(ctx, agent, project)
    try:
        client = tools._client_or_raise()
        _, rows = client.query(relations_sql(tools._scopes(scope)))
    finally:
        tools.close()
    found = collect_entities(relations_from_rows(rows))
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
    print(f"\n{len(found)} entities")


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
    print(f"\n{len(result['items'])} of {result['candidates']} edges; retrieval {result['retrieval_id']}")


@graph_app.command()
def resolve(
    ctx: typer.Context,
    scope: SCOPES = None,
    entity_type: Annotated[
        str | None, typer.Option(help="Only cluster entities of this type.")
    ] = None,
    model: Annotated[str, typer.Option(help="Ollama model that proposes clusters.")] = DEFAULT_MODEL,
    host: Annotated[str, typer.Option(help="Ollama host.")] = OLLAMA_HOST,
    block_size: Annotated[int, typer.Option(min=2)] = BLOCK_SIZE,
    apply: Annotated[
        bool, typer.Option("--apply", help="Record the merges. Without it, only propose.")
    ] = False,
    agent: AGENT = "human",
    project: PROJECT = None,
) -> None:
    """Propose entity merges with a local model; --apply records them as supersedes history."""
    tools = _tools(ctx, agent, project)
    resolver = OllamaResolver(model, host=host)
    try:
        client = tools._client_or_raise()
        scopes = tools._scopes(scope)
        _, rows = client.query(relations_sql(scopes))
        relations = relations_from_rows(rows)
        if not relations:
            print("no relations recorded in these scopes")
            return
        merges = propose(relations, resolver, entity_type=entity_type, block_size=block_size)
        if not merges:
            print(f"{resolver.name} proposes no merges across {len(relations)} relations")
            return
        _print_merges(merges)
        if not apply:
            print("\nproposal only; pass --apply to record these merges")
            return
        for merge in merges:
            result = tools.merge_entity(
                merge.alias,
                merge.canonical,
                f"entity resolution by {resolver.name}",
                scopes=scopes,
                canonical_type=merge.entity_type,
                actor=f"{agent} via {resolver.name}",
            )
            if result["degraded"]:
                raise typer.Exit(_fail(result["reason"]))
            print(
                f"merged {merge.alias!r} into {merge.canonical!r}: "
                f"{len(result['superseded'])} superseded, {len(result['retired'])} retired"
            )
    finally:
        tools.close()


def _print_merges(merges: list[Merge]) -> None:
    width = max(len(m.alias) for m in merges)
    for m in merges:
        print(f"{m.alias:<{width}}  ->  {m.canonical}" + (f"  ({m.entity_type})" if m.entity_type else ""))


def _fail(reason: str) -> int:
    print(f"mem graph: {reason}", flush=True)
    return 1
