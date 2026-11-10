"""The relation graph: relations as memories, hop traversal, and entity resolution."""

from __future__ import annotations

import json

import pytest

from agent_memory import endpoints
from agent_memory.cli import main
from agent_memory.db import sql_literal
from agent_memory.graph import Relation, hop_sql, normalize_entity, relations_sql
from agent_memory.graph.resolve import (
    Cluster,
    Entity,
    OllamaResolver,
    ResolveError,
    blocks,
    cluster_prompt,
    collect_entities,
    parse_clusters,
    plan_merges,
    propose,
)
from agent_memory.mcp_server import MemoryTools, ToolError

# model


def test_normalize_entity_is_case_punctuation_and_width_insensitive():
    assert normalize_entity("  Neil  Armstrong! ") == "neil armstrong"
    assert normalize_entity("\uff2e\uff21\uff33\uff21") == "nasa"  # fullwidth NASA
    assert normalize_entity("src/agent_memory/db.py") == "src/agent memory/db.py"
    assert "|" not in normalize_entity("a|b")
    assert normalize_entity("...") == ""


def test_relation_key_sentence_and_metadata_round_trip():
    rel = Relation(" Neil Armstrong ", "Commanded", "Apollo 11", "person", None)
    assert rel.key == "neil armstrong|commanded|apollo 11"
    assert rel.sentence == "Neil Armstrong Commanded Apollo 11"
    assert rel.triple == "(Neil Armstrong) --[Commanded]--> (Apollo 11)"
    assert rel.subject_type == "PERSON"
    again = Relation.from_metadata(json.loads(json.dumps(rel.metadata())))
    assert again == rel
    assert Relation.from_metadata({"other": 1}) is None
    assert Relation.from_metadata({"graph": {"subject": "a"}}) is None
    assert Relation.from_metadata(None) is None


@pytest.mark.parametrize(
    ("subject", "predicate", "obj"),
    [("", "is", "b"), ("a", "", "b"), ("a", "is", "A"), ("a", "is", "!!!")],
)
def test_relation_rejects_empty_or_self_referencing_triples(subject, predicate, obj):
    with pytest.raises(ValueError, match="must"):
        Relation(subject, predicate, obj)


def test_renamed_rewrites_matching_endpoints_and_keeps_aliases():
    rel = Relation("Armstrong", "walked on", "Moon", "PERSON", "LOCATION")
    merged = rel.renamed("armstrong", "Neil Armstrong")
    assert merged.subject == "Neil Armstrong"
    assert merged.object == "Moon"
    assert merged.subject_type == "PERSON"
    assert merged.aliases == ("Armstrong",)
    assert merged.key == "neil armstrong|walked on|moon"
    assert rel.renamed("nobody", "X") is rel
    twice = merged.renamed("moon", "the Moon", "PLACE")
    assert twice.aliases == ("Armstrong", "Moon")
    assert twice.object_type == "PLACE"


# resolution helpers


def _entity(name: str, etype: str | None = None) -> Entity:
    return Entity(normalize_entity(name), name, etype, 1, (f"{name} did a thing",), (name,))


def test_parse_clusters_repairs_model_output():
    names = ["Edwin Aldrin", "Buzz Aldrin", "Jim Lovell", "NASA"]
    text = json.dumps(
        {
            "clusters": [
                {"canonical": "Buzz Aldrin", "aliases": ["edwin aldrin", "Buzz Aldrin"]},
                {"canonical": "Made Up", "aliases": ["Jim Lovell", "Edwin Aldrin", "Ghost"]},
                "garbage",
            ]
        }
    )
    clusters = parse_clusters(text, names)
    assert clusters[0] == Cluster("Buzz Aldrin", ("Edwin Aldrin", "Buzz Aldrin"))
    # Edwin was already placed; Ghost is not an input; Made Up is not in its own cluster.
    assert clusters[1] == Cluster("Jim Lovell", ("Jim Lovell",))
    # NASA was dropped by the model and keeps its own cluster instead of vanishing.
    assert clusters[2] == Cluster("NASA", ("NASA",))
    merges = plan_merges(clusters, "PERSON")
    assert [(m.alias, m.canonical, m.entity_type) for m in merges] == [
        ("Edwin Aldrin", "Buzz Aldrin", "PERSON")
    ]


def test_parse_clusters_extracts_json_from_prose_and_rejects_garbage():
    assert parse_clusters('Sure: {"clusters": []} done', ["A"]) == [Cluster("A", ("A",))]
    with pytest.raises(ResolveError):
        parse_clusters("no json here", ["A"])
    with pytest.raises(ResolveError):
        parse_clusters('{"nope": 1}', ["A"])


def test_collect_entities_blocks_and_prompt():
    rels = [
        Relation("Armstrong", "commanded", "Apollo 11", "PERSON", "EVENT"),
        Relation("armstrong", "walked on", "Moon", None, "LOCATION"),
        Relation("Aldrin", "flew on", "Apollo 11", "PERSON"),
    ]
    found = collect_entities(rels)
    by_norm = {e.norm: e for e in found}
    assert by_norm["armstrong"].name == "Armstrong"
    assert by_norm["armstrong"].mentions == 2
    assert by_norm["armstrong"].forms == ("Armstrong", "armstrong")
    assert by_norm["apollo 11"].entity_type == "EVENT"
    assert by_norm["moon"].entity_type == "LOCATION"
    grouped = blocks(found, block_size=1)
    assert [(t, [e.name for e in members]) for t, members in grouped] == [
        ("EVENT", ["Apollo 11"]),
        ("LOCATION", ["Moon"]),
        ("PERSON", ["Aldrin"]),
        ("PERSON", ["Armstrong"]),
    ]
    prompt = cluster_prompt("PERSON", [by_norm["armstrong"], by_norm["aldrin"]])
    assert "- Armstrong: Armstrong commanded Apollo 11; armstrong walked on Moon" in prompt
    assert "PERSON entities" in prompt


def test_propose_skips_single_entity_blocks_and_calls_model_per_block():
    calls: list[tuple[str, list[str]]] = []

    class Stub:
        name = "stub"

        def clusters(self, entity_type, entities):
            calls.append((entity_type, [e.name for e in entities]))
            names = [e.name for e in entities]
            return parse_clusters(
                json.dumps({"clusters": [{"canonical": names[0], "aliases": names}]}), names
            )

    rels = [
        Relation("Neil Armstrong", "commanded", "Apollo 11", "PERSON", "EVENT"),
        Relation("Armstrong", "walked on", "Moon", "PERSON", "LOCATION"),
        Relation("Apollo 11", "landed on", "Moon", "EVENT", "LOCATION"),
    ]
    merges = propose(rels, Stub())
    assert calls == [("PERSON", ["Armstrong", "Neil Armstrong"])]
    assert [(m.alias, m.canonical) for m in merges] == [("Neil Armstrong", "Armstrong")]
    assert propose(rels, Stub(), entity_type="event") == []


def test_ollama_resolver_posts_json_and_reports_truncation(monkeypatch):
    seen: list[dict] = []

    def post(url, body, **_):
        seen.append({"url": url, **body})
        return {"message": {"content": '{"clusters": []}'}}, 1.0

    monkeypatch.setattr(endpoints, "post_json", post)
    resolver = OllamaResolver("tiny", host="http://ollama.test")
    assert resolver.clusters("PERSON", [_entity("A"), _entity("B")]) == [
        Cluster("A", ("A",)),
        Cluster("B", ("B",)),
    ]
    assert seen[0]["url"] == "http://ollama.test/api/chat"
    assert seen[0]["model"] == "tiny"
    assert seen[0]["format"] == "json"
    assert "- A: A did a thing" in seen[0]["messages"][0]["content"]

    monkeypatch.setattr(
        endpoints,
        "post_json",
        lambda *a, **k: ({"message": {"content": "{"}, "done_reason": "length"}, 1.0),
    )
    with pytest.raises(ResolveError):
        resolver.clusters("PERSON", [_entity("A"), _entity("B")])


# live: relate, graph, merge


@pytest.fixture
def tools(live_cfg):
    t = MemoryTools(live_cfg, "factory-droid", harness_session_id="graph-1", project="apollo")
    yield t
    t.close()


def q(client, sql):
    return client.query(sql)[1]


def _seed(tools) -> dict[str, str]:
    ids = {}
    for s, p, o, st, ot in [
        ("Neil Armstrong", "commanded", "Apollo 11", "PERSON", "EVENT"),
        ("Apollo 11", "landed on", "the Moon", "EVENT", "LOCATION"),
        ("Armstrong", "walked on", "the Moon", "PERSON", "LOCATION"),
        ("Edwin Aldrin", "flew on", "Apollo 11", "PERSON", "EVENT"),
        ("the Moon", "orbits", "Earth", "LOCATION", "LOCATION"),
    ]:
        ids[f"{s}|{p}|{o}"] = tools.relate(s, p, o, st, ot)["memory_id"]
    return ids


def test_relate_stores_a_fact_memory_with_graph_metadata_and_dedupes_by_key(tools, live_client):
    first = tools.relate("payments-api", "owned by", "Dana", "SERVICE", "person", scope="own")
    assert first["status"] == "active"
    assert first["key"] == "payments-api|owned by|dana"
    assert first["triple"] == "(payments-api) --[owned by]--> (Dana)"
    assert not first["deduplicated"]
    row = q(
        live_client,
        "SELECT type, key, content, metadata::JSON::VARCHAR FROM memories "
        f"WHERE memory_id = {sql_literal(first['memory_id'])}",
    )[0]
    assert row[:3] == ("fact", "payments-api|owned by|dana", "payments-api owned by Dana")
    graph = json.loads(row[3])["graph"]
    assert graph["subject_type"] == "SERVICE"
    assert graph["object_type"] == "PERSON"
    assert graph["object_norm"] == "dana"

    # Same edge, different surface forms and prose: confirmed, not duplicated.
    again = tools.relate(
        "Payments API".replace("Payments API", "PAYMENTS-API"),
        "Owned By",
        "dana!",
        content="Dana is the on-call owner of payments-api",
        scope="own",
    )
    assert again["deduplicated"]
    assert again["memory_id"] == first["memory_id"]
    assert q(
        live_client,
        "SELECT action, reason FROM memory_actions "
        f"WHERE memory_id = {sql_literal(first['memory_id'])}",
    ) == [("confirm", "relate() with the same subject, predicate, and object")]

    quarantined = tools.relate("x-lib", "depends on", "y-lib", trust_level="tool", scope="own")
    assert quarantined["status"] == "quarantined"
    with pytest.raises(ToolError):
        tools.relate("same", "is", "SAME")
    with pytest.raises(ToolError):
        tools.relate("a", "b", "c", trust_level="imported")


def test_relate_redacts_entity_names_and_content(tools, live_client):
    # Built at runtime so no token-shaped literal sits in the source.
    fake_token = "ghp_" + "x" * 36
    stored = tools.relate("deploy job", "uses token", fake_token, scope="secret")
    row = q(
        live_client,
        "SELECT key, content, metadata::JSON::VARCHAR FROM memories "
        f"WHERE memory_id = {sql_literal(stored['memory_id'])}",
    )[0]
    assert all("ghp_" not in cell for cell in row)


def test_graph_walks_hops_logs_the_retrieval_and_misses_unresolved_aliases(tools, live_client):
    ids = _seed(tools)
    one = tools.graph("NEIL ARMSTRONG", hops=1)
    assert one["strategy"] == "graph_hop_v1"
    assert one["triples"] == ["(Neil Armstrong) --[commanded]--> (Apollo 11)"]
    assert one["entities"] == ["Apollo 11", "Neil Armstrong"]

    two = tools.graph("neil armstrong", hops=2)
    keys = {i["key"]: i["depth"] for i in two["items"]}
    assert keys == {
        "neil armstrong|commanded|apollo 11": 0,
        "apollo 11|landed on|the moon": 1,
        "edwin aldrin|flew on|apollo 11": 1,
    }
    # "Armstrong" is a different node until it is merged, and Earth is three hops out.
    assert "armstrong|walked on|the moon" not in keys
    assert "the moon|orbits|earth" not in keys
    assert two["candidates"] == 3
    assert "status" not in two["items"][0]
    assert two["items"][0]["score"] == 1.0
    assert two["items"][-1]["score_components"] == {"depth": 1.0}

    retrieval = q(
        live_client,
        "SELECT strategy, query_text, candidates, returned, params::JSON::VARCHAR, sql_used "
        f"FROM retrievals WHERE retrieval_id = {sql_literal(two['retrieval_id'])}",
    )[0]
    assert retrieval[:4] == ("graph_hop_v1", "neil armstrong", 3, 3)
    assert json.loads(retrieval[4])["hops"] == 2
    assert "WITH RECURSIVE" in retrieval[5]
    items = q(
        live_client,
        "SELECT rank, memory_id::VARCHAR, score, status_at_recall FROM retrieval_items "
        f"WHERE retrieval_id = {sql_literal(two['retrieval_id'])} ORDER BY rank",
    )
    assert items[0] == (1, ids["Neil Armstrong|commanded|Apollo 11"], 1.0, "active")
    assert len(items) == 3
    event = q(
        live_client,
        "SELECT kind, content, payload::JSON::VARCHAR FROM events "
        f"WHERE event_id = {sql_literal(two['request_event_id'])}",
    )[0]
    assert event[:2] == ("memory_recall", "neil armstrong")
    assert json.loads(event[2])["graph"] == {"entity_norm": "neil armstrong", "hops": 2}

    cited = tools.cite(retrieval_id=two["retrieval_id"], memory_ids=[items[0][1]])
    why = q(
        live_client,
        "SELECT via, memory_id::VARCHAR FROM why_answer "
        f"WHERE answer_event_id = {sql_literal(cited['answer_event_id'])}",
    )
    assert ("cited_memory", items[0][1]) in why

    nothing = tools.graph("nobody here", hops=3)
    assert nothing["items"] == []
    assert nothing["candidates"] == 0
    assert nothing["retrieval_id"] is not None  # misses are logged too
    with pytest.raises(ToolError):
        tools.graph("   ")


def test_merge_entity_supersedes_rewrites_and_retires_as_history(tools, live_client):
    ids = _seed(tools)
    recall = tools.recall("anything")
    source = recall["request_event_id"]
    # A tool-derived relation that was approved must stay approved after the merge.
    approved = tools.relate(
        "Armstrong",
        "trained at",
        "Edwards AFB",
        "PERSON",
        "LOCATION",
        trust_level="tool",
        source_event_ids=[source],
    )
    tools.review("approve", approved["memory_id"], "checked")

    result = tools.merge_entity(
        "armstrong", "Neil Armstrong", "same astronaut", canonical_type="PERSON"
    )
    assert not result["degraded"]
    superseded = {s["memory_id"]: s for s in result["superseded"]}
    assert set(superseded) == {ids["Armstrong|walked on|the Moon"], approved["memory_id"]}
    assert result["retired"] == []

    walked = superseded[ids["Armstrong|walked on|the Moon"]]
    assert walked["key"] == "neil armstrong|walked on|the moon"
    new_row = q(
        live_client,
        "SELECT content, supersedes::VARCHAR, trust_level, metadata::JSON::VARCHAR, "
        "ms.status FROM memories m JOIN memory_status ms USING (memory_id) "
        f"WHERE memory_id = {sql_literal(walked['by'])}",
    )[0]
    assert new_row[0] == "Neil Armstrong walked on the Moon"
    assert new_row[1] == ids["Armstrong|walked on|the Moon"]
    assert json.loads(new_row[3])["graph"]["aliases"] == ["Armstrong"]
    assert new_row[4] == "active"
    statuses = dict(
        q(
            live_client,
            "SELECT memory_id::VARCHAR, status FROM memory_status WHERE memory_id IN "
            f"({sql_literal(ids['Armstrong|walked on|the Moon'])}, "
            f"{sql_literal(approved['memory_id'])})",
        )
    )
    assert set(statuses.values()) == {"superseded"}

    carried = superseded[approved["memory_id"]]["by"]
    assert q(
        live_client,
        f"SELECT trust_level, ms.status FROM memories m JOIN memory_status ms USING (memory_id) "
        f"WHERE memory_id = {sql_literal(carried)}",
    ) == [("tool", "active")]
    sources = q(
        live_client,
        "SELECT event_id::VARCHAR, relation FROM memory_sources "
        f"WHERE memory_id = {sql_literal(carried)} ORDER BY event_id",
    )
    assert (source, "derived_from") in sources
    assert (result["event_id"], "derived_from") in sources
    assert len(sources) == 3  # original write event, the source, and the merge event

    # The graph is now connected through the canonical node.
    two = tools.graph("Neil Armstrong", hops=1)
    assert sorted(i["key"] for i in two["items"]) == [
        "neil armstrong|commanded|apollo 11",
        "neil armstrong|trained at|edwards afb",
        "neil armstrong|walked on|the moon",
    ]

    # Merging into a relation that already exists retires the duplicate instead.
    buzz = tools.relate("Buzz", "flew on", "Apollo 11", "PERSON", "EVENT")
    second = tools.merge_entity("Buzz", "Edwin Aldrin", "nickname")
    assert second["superseded"] == []
    assert second["retired"] == [
        {"memory_id": buzz["memory_id"], "into": ids["Edwin Aldrin|flew on|Apollo 11"]}
    ]
    action = q(
        live_client,
        "SELECT action, related_memory_id::VARCHAR, actor FROM memory_actions "
        f"WHERE memory_id = {sql_literal(buzz['memory_id'])}",
    )
    assert action == [("retire", ids["Edwin Aldrin|flew on|Apollo 11"], "factory-droid")]

    assert tools.merge_entity("Nobody", "Someone", "no such entity")["superseded"] == []
    with pytest.raises(ToolError):
        tools.merge_entity("Same", "same", "x")
    with pytest.raises(ToolError):
        tools.merge_entity("a", "b", "  ")


def test_graph_sql_passes_the_policy_through_quack(live_client):
    # Both statements classify as SELECT-style after literal stripping.
    assert live_client.query(hop_sql("x|y", 3, ["global"], ["fact"]))[1] == []
    assert live_client.query(relations_sql(["global"], ["x"], ["fact"]))[1] == []


# live: CLI


def test_mem_graph_cli_entities_show_and_resolve(live_cfg, monkeypatch, capsys):
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    seed = MemoryTools(live_cfg, "oh-my-pi", harness_session_id="cli-seed", project="cli")
    try:
        seed.relate("Alice Chen", "owns", "billing-svc", "PERSON", "SERVICE")
        seed.relate("A. Chen", "reviews", "billing-svc", "PERSON", "SERVICE")
        seed.relate("billing-svc", "depends on", "ledger-db", "SERVICE", "DATABASE")
    finally:
        seed.close()

    def mem(*args: str) -> tuple[int, str]:
        code = main(["--home", str(live_cfg.home), "--port", str(live_cfg.port), "graph", *args])
        return code, capsys.readouterr().out

    code, out = mem("entities", "--scope", "cli")
    assert code == 0
    assert "Alice Chen" in out
    assert "A. Chen" in out
    assert "4 entities" in out

    code, out = mem("show", "billing-svc", "--hops", "1", "--scope", "cli")
    assert code == 0
    assert "(Alice Chen) --[owns]--> (billing-svc)" in out
    assert "3 of 3 edges" in out

    def post(url, body, **_):
        prompt = body["messages"][0]["content"]
        if "Alice Chen" in prompt:
            cluster = {"canonical": "Alice Chen", "aliases": ["Alice Chen", "A. Chen"]}
            content = {"clusters": [cluster]}
        else:
            content = {"clusters": []}
        return {"message": {"content": json.dumps(content)}}, 1.0

    monkeypatch.setattr(endpoints, "post_json", post)
    code, out = mem("resolve", "--scope", "cli", "--model", "stub")
    assert code == 0
    assert "A. Chen  ->  Alice Chen  (PERSON)" in out
    assert "proposal only" in out

    code, out = mem("resolve", "--scope", "cli", "--model", "stub", "--apply")
    assert code == 0
    assert "merged 'A. Chen' into 'Alice Chen': 1 superseded, 0 retired" in out

    code, out = mem("show", "Alice Chen", "--hops", "1", "--scope", "cli")
    assert "(Alice Chen) --[reviews]--> (billing-svc)" in out
    assert "(Alice Chen) --[owns]--> (billing-svc)" in out

    code, out = mem("resolve", "--scope", "cli", "--model", "stub")
    assert code == 0
    assert "proposes no merges" in out

    code, out = mem("entities", "--scope", "nothing-here")
    assert code == 0
    assert "no relations" in out
