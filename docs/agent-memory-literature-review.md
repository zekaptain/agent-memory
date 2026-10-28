# Memory for LLM Agents: A Critical Review of Approaches, Evaluation, and Governance (2023 to 2026)

Prepared for the agent-memory project, 2026-10-25.

## Abstract

Large language models are stateless functions of their input. Everything an agent "remembers" is therefore an engineering decision about what to put back into the next prompt, what to fold into the model's weights or latent state, and what to keep in storage the model can reach through tools. This review surveys the approaches that have emerged for this problem between 2023 and late 2026. It organises them along three axes that recur in the literature: the substrate memory lives in (parametric, latent, or token-level), the function it serves (working, episodic, semantic, procedural), and the control policy that decides what is written, consolidated, retrieved, and forgotten. Five mechanism families are examined: parametric and latent memory; context-resident memory and compaction; external retrieval stores, from flat vector indexes through temporal knowledge graphs to filesystems and agent-maintained wikis; experiential and procedural memory that distils strategies from past trajectories; and learned memory-management policies trained with reinforcement learning. The review then examines evaluation, where a credibility problem has become acute: headline scores on LoCoMo range from 53 to 92.5 depending on who ran the system, with which answer model, judge, and ingestion path, and a plain filesystem agent has outscored several dedicated memory products. It closes with the security, temporal-validity, provenance, and deletion literature that arrived in force in 2026, which argues that memory must be governed at write time with provenance, versioning, and policy-aware retention rather than filtered only at retrieval. The main findings are: (1) no mechanism family dominates; the harness that decides what loads, compacts, and persists often matters more than the store; (2) recall benchmarks are near saturation and poorly predict agentic use of memory, where current systems score far lower; (3) write-time summarisation creates a "write-before-query" information barrier that only retention of raw evidence with links can remove; (4) similarity search cannot detect contradiction or staleness, so temporal validity must be modelled explicitly; (5) deletion on request is essentially unsolved; and (6) memory is a persistent attack surface that compounds across sessions and agents. The final section maps these findings onto the design decisions of an append-only, auditable, multi-harness memory ledger.

## 1. Introduction

### 1.1 Why memory, and why now

A transformer language model conditions its output on a bounded sequence of tokens. Across calls it retains nothing. Early chat applications hid this by replaying the conversation, and early "agents" extended that by replaying tool outputs. By 2023 three pressures made this untenable. Conversations and tasks outgrew the window. Users expected assistants to remember preferences and prior work across sessions. And agents began to run for hours or days, where failing to recall a constraint stated earlier (an allergy, a renamed API, a prior failed attempt) produces visible, costly errors (Zhao et al., 2026; Wu et al., 2026b).

The response has been a research and product field that now has its own surveys, benchmarks, security literature, and vendor ecosystem. Between 2024 and 2026 at least seven broad surveys appeared (Zhang et al., 2024; Du et al., 2025; Wu et al., 2025b; Hu et al., 2025b; Du, 2026; Jiang et al., 2026; Luo et al., 2026), each proposing a taxonomy, which is itself evidence that the field has not converged on vocabulary. Hu et al. (2025b), a 47-author survey, open by noting that "the proliferation of loosely defined memory terminologies has further obscured conceptual clarity" and that the long-term/short-term split "has proven insufficient."

### 1.2 What this review covers

This review asks four questions:

1. What approaches exist for giving LLM agents memory, and what are the mechanisms behind each?
2. What does the evidence say about their relative effectiveness, and how trustworthy is that evidence?
3. What failure modes have been documented, including security, staleness, and deletion?
4. What do these findings imply for the design of a memory system intended to be auditable and shared across agent harnesses?

Scope is memory for LLM-based agents: systems where a model takes actions over multiple steps or sessions and must carry information across them. Three adjacent topics are treated only where they bear on agent memory: long-context modelling as such, retrieval-augmented generation (RAG) over static corpora, and continual pretraining. Following Hu et al. (2025b), the distinction adopted here is that agent memory is writable by the agent during operation, evolves over time, and is read back to influence future behaviour; RAG over a fixed corpus and a long context window are ingredients, not memory systems.

### 1.3 Method and evidence grading

Sources were gathered from arXiv, conference proceedings (NeurIPS, ICML, ICLR, ACL, EMNLP, EACL), and vendor publications, with searches conducted on 2026-10-25 and snowballing from the surveys above. Where a claim rests on a vendor's own benchmark run, that is stated. Three grades of evidence are distinguished in the text:

- **Peer-reviewed**: accepted at a venue with review (noted where known).
- **Preprint**: arXiv only at time of writing.
- **Vendor or practitioner report**: blog posts, READMEs, product documentation. Useful for what was shipped; weak for comparative performance.

The field moves fast and much of the strongest 2026 work is still preprint. This is noted rather than treated as disqualifying, but readers should weight claims accordingly.

## 2. Conceptual Foundations and Taxonomies

### 2.1 Borrowings from cognitive science

Almost every agent memory paper borrows a human-memory frame. Three recur.

**Multi-store models.** Atkinson and Shiffrin's sensory, short-term, long-term stores motivate systems such as LightMem (Fang et al., 2025), which filters input through a "sensory" stage, groups it into topic-segmented short-term memory, and consolidates into long-term memory offline. MemoryBank (Zhong et al., 2023) borrowed Ebbinghaus's forgetting curve to decay memories that are not reinforced by retrieval.

**Tulving's episodic/semantic distinction, extended with procedural memory.** Sumers et al. (2023) formalised this for language agents in the CoALA framework: working memory (the active context), episodic memory (records of past experience), semantic memory (facts about the world and the agent), and procedural memory (skills, including the agent's own code and prompts). This four-way split has become the lingua franca of production systems; MIRIX (Wang and Chen, 2025) adds "resource" and "knowledge vault" components, and practitioner summaries in 2026 describe exactly this four-tier vocabulary as the industry default (Kumar, 2026).

**Complementary learning systems.** The hippocampal/neocortical fast/slow learning story motivates systems that record episodes quickly and consolidate them into abstractions slowly. HippoRAG (Gutiérrez et al., 2024) is the most explicit, modelling a knowledge graph plus Personalized PageRank as hippocampal pattern completion over a neocortical (LLM) store. Pink et al. (2025) argue, in a position paper, that episodic memory specifically is "the missing piece" for long-term agents, and define it by five properties: long-term, explicit, single-shot (acquired from one exposure), instance-specific, and contextual (bound to when and where). This definition is useful because it separates what a ledger of events provides from what a store of extracted facts provides.

These analogies are productive as design heuristics. They are weaker as evidence: no paper surveyed here shows that fidelity to a cognitive model predicts task performance, and the simplest systems in Section 5.4 ignore the analogies entirely while performing competitively.

### 2.2 Survey taxonomies

The surveys propose overlapping but distinct cuts. Table 1 summarises them.

| Survey | Primary axes | Notable contribution |
|---|---|---|
| Zhang et al. (2024) | Sources, forms, operations (writing, management, reading) | First memory-specific survey; framed memory as what gives agents a "self-evolving capability" |
| Du et al. (2025) | Representation (parametric vs contextual) × six atomic operations: consolidation, updating, indexing, forgetting, retrieval, compression | Operation-centric view; maps operations to long-term, long-context, parametric-modification, and multi-source topics |
| Wu et al. (2025b) | Human memory to AI memory mapping | Cognitive grounding |
| Hu et al. (2025b) | Forms (token-level, parametric, latent) × functions (factual, experiential, working) × dynamics (formation, evolution, retrieval) | Explicitly delimits agent memory from RAG and context engineering; compiles benchmarks and frameworks |
| Du (2026) | Write–manage–read loop; temporal scope × substrate × control policy; five mechanism families | Mechanism families used in this review's structure |
| Jiang et al. (2026) | Four structures: lightweight semantic; entity-centric and personalised; episodic and reflective; structured and hierarchical | Empirical analysis of evaluation pathologies and system costs |
| Luo et al. (2026) | Evolutionary stages: Storage (trajectory preservation), Reflection (trajectory refinement), Experience (trajectory abstraction) | Historical and teleological view; ACL 2026 Findings |
| Lin et al. (2026) | Six lifecycle phases × four security objectives | Security-specific; EMNLP 2026 |

Table 1. Taxonomies proposed by the major surveys.

### 2.3 Working taxonomy for this review

To compare systems, this review uses three axes and one lifecycle.

**Substrate.** Where the information physically lives:
- *Parametric*: in model weights (fine-tuning, knowledge editing, memory layers).
- *Latent*: in continuous state that is not weights and not tokens (KV caches, recurrent states, memory tokens, test-time-trained modules).
- *Token-level*: as text (or structured data serialised to text) the model reads through its context window.

**Function.** What the information is for: working (the current task's state), episodic (what happened, when), semantic (what is true), procedural (how to act).

**Control.** Who decides what to write and read, and when: a fixed pipeline (an extraction LLM runs on every turn), the agent itself through tools (MemGPT's self-directed edits), a separate memory agent, a learned policy (RL-trained managers), or a human (AGENTS.md files).

**Lifecycle.** Du et al.'s (2025) six operations, consolidation, updating, indexing, forgetting, retrieval, compression, plus the write-path filtering and provenance capture that the 2026 security literature adds at the front (Lin et al., 2026).

The sections that follow walk through the substrate axis from inside the model outward, then treat experiential memory and learned control as cross-cutting families, then evaluation and governance.

## 3. Parametric and Latent Memory

### 3.1 Extending the window from inside the architecture

The earliest "memory" work changed the model. Compressive Transformers (Rae et al., 2019) compressed old activations into a secondary memory. Memorizing Transformers (Wu et al., 2022) added approximate k-nearest-neighbour lookup over past key-value pairs, giving a 262k-token effective context without retraining the base layers. The Recurrent Memory Transformer (Bulatov et al., 2022) passed memory tokens between segments. Infini-attention (Munkhdalai et al., 2024) combined a compressive memory with local attention in a single block, claiming unbounded context at bounded cost. Memory3 (Yang et al., 2024) externalised explicit memory as sparse key-value pairs to be loaded alongside parameters.

These approaches extend what the model can attend to, but they inherit a known weakness of attention itself: performance degrades with input length even when the information is nominally in range. Liu et al. (2023) documented the "lost in the middle" effect. Chroma's July 2025 technical report (Hong et al., 2025) tested 18 frontier models and found that every one degraded as input length grew, including on tasks as simple as repeated-word copying; the effect is now commonly called "context rot." RULER (Hsieh et al., 2024) and NoLiMa (Modarressi et al., 2025) showed that effective context lengths are far below advertised ones when the task requires anything beyond literal matching. These results matter for the whole field: they are the reason "just use a bigger window" is not an answer, and they are why memory systems that reduce the number of tokens presented to the model can beat full-context baselines even when every needed fact is in the full context (Rasmussen et al., 2025; Chhikara et al., 2025).

### 3.2 Test-time learning and neural long-term memory

A second line makes part of the network learn during inference. Test-Time Training layers (Sun et al., 2024) treat the hidden state as a small model updated by gradient steps on the incoming sequence. Titans (Behrouz et al., 2024) adds a "neural long-term memory" module trained at test time with a surprise-based update rule (gradient of the loss with respect to the input, with momentum and adaptive forgetting), reporting better scaling than Transformers and modern linear recurrent models to context windows beyond 2M tokens on needle-in-haystack tasks. Nested Learning (Behrouz et al., 2025, NeurIPS 2025) generalises this to a "continuum memory system" in which different components update at different frequencies, and proposes the HOPE architecture as a proof of concept aimed at continual learning without catastrophic forgetting.

These are genuine memory mechanisms in the sense that they accumulate information across a stream. Their limitation for agent systems is that the memory is a tensor. It cannot be inspected, cited, corrected, exported, or shared with a different model. No agent memory survey reviewed here reports a production agent using them, and the agentic benchmarks of Section 9 do not include them.

### 3.3 Memory layers and latent memory tokens

Meta's Memory Layers at Scale (Berges et al., 2024) add a trainable key-value lookup with product-key indexing, adding up to 128B memory parameters without increasing FLOPs, and show gains on factual tasks over dense and MoE baselines at matched compute. MemoryLLM (Wang et al., 2024) and its extension M+ (Wang et al., 2025, ICML 2025) attach a pool of latent memory tokens to a frozen model and train an update mechanism; M+ adds a long-term pool with a co-trained retriever and reports retention beyond 160k tokens. Larimar (Das et al., 2024, ICML 2024) uses an episodic memory controller for one-shot knowledge updates and selective forgetting.

A distinct and practical idea is the Cartridge (Eyuboglu et al., 2025, Stanford Hazy Research): train a small KV cache offline on a corpus using a "self-study" recipe (the model generates questions about the corpus and trains the cache to answer them), then load that cache at inference. The authors report 38.6× memory reduction and 26.4× throughput improvement compared with placing the corpus in context, at matched quality. This is parametric memory for a document set, trained per corpus. It is relevant to agent memory as a possible consolidation target: a nightly job could compile a project's stable knowledge into a cartridge. No agent system surveyed here has yet done so.

### 3.4 Knowledge editing and fine-tuning

Knowledge editing methods (ROME, Meng et al., 2022; MEMIT, Meng et al., 2023) locate and modify the weights that encode a fact. They are precise for single edits but degrade under sequential editing (Gupta et al., 2024) and do not propagate consequences of an edit to related facts. Fine-tuning on interaction logs is the obvious way to make an agent "learn," and the literature on continual learning for LLMs is large, but within the agent memory literature it is treated mainly as a baseline. Memento (Zhou et al., 2025a) is titled "Fine-tuning LLM Agents without Fine-tuning LLMs" precisely because the authors regard weight updates as too costly, too slow, and too opaque for continual adaptation.

### 3.5 Assessment

Parametric and latent memory is the only family that can make a model faster or cheaper at using what it has learned, because the information does not re-enter the context window each time. It is also the only family that offers nothing to an auditor: there is no record of what was stored, no provenance, no way to quarantine a suspect memory, and deletion means retraining or editing with uncertain side effects. Hu et al. (2025b) list parametric memory as one of three forms; in practice the surveyed agent systems treat it as future work. The honest summary is that parametric memory is where the long-run efficiency gains lie and token-level memory is where everything deployable lives today.

## 4. Context-Resident Memory and Compaction

### 4.1 The window as working memory

In every agent the context window is the working memory. What distinguishes systems is how they manage it as it fills. Anthropic's September 2025 guidance (Anthropic, 2025a) names three strategies that have become standard vocabulary: *compaction* (summarise the conversation and restart with the summary), *structured note-taking* (write notes to storage outside the window and pull them back when needed), and *sub-agent architectures* (delegate a sub-task to a fresh context and return a short summary). The same month Anthropic shipped a server-side context-editing API that clears old tool results automatically and a file-based memory tool (Anthropic, 2025b), and Claude Code's "auto memory" (Anthropic, 2026) loads the first 200 lines or 25 KB of agent-written notes at the start of every session. These are vendor documents, but they describe the baseline against which research systems are now compared.

### 4.2 Hierarchical virtual context: MemGPT

MemGPT (Packer et al., 2023) is the founding paper of this family and the most cited design in the production literature. It treats the context window as RAM and external storage as disk. The "main context" holds system instructions, a small editable "working context," and a FIFO message queue; "external context" comprises recall storage (the full message history, searchable) and archival storage (arbitrary documents). The key move is that the LLM itself issues function calls to page information in and out, edit its working context, and search external storage, and the system raises "interrupts" (for example, a warning when the queue is nearly full) that prompt the model to act. MemGPT outperformed fixed-context baselines on a deep-memory-retrieval (DMR) benchmark and on document QA; Letta is its productised successor. Kumar (2026) observes that the vocabulary every 2026 vendor uses is MemGPT's.

The idea's durability comes from a property that later work keeps rediscovering: a capable model given simple, well-known tools and told to manage its own memory does well, and the harness's rules about what loads and when matter as much as the store (Letta, 2025; Wooders, 2026).

### 4.3 Compaction and its failure modes

Compaction is lossy by construction, and three papers quantify the damage.

ACE (Zhang et al., 2025b, "Agentic Context Engineering") identifies two pathologies of iteratively rewritten contexts: *brevity bias*, where summarisers drop domain-specific heuristics in favour of concise generalities, and *context collapse*, where repeated rewriting erodes detail over time. ACE's remedy is a generator/reflector/curator loop that makes incremental, itemised updates to a structured "playbook" rather than rewriting it wholesale, reporting +10.6% on agent benchmarks and +8.6% on finance tasks over strong baselines.

ACON (Kang et al., 2025, Microsoft) treats the compression prompt as something to optimise: it analyses cases where a compressed context led to failure while the full context succeeded, refines natural-language compression guidelines accordingly, and distils the resulting compressor into a smaller model. It reports 26 to 54% peak-token reduction while preserving task performance on AppWorld, OfficeBench, and multi-objective QA.

TierMem (Zhu et al., 2026) names the structural problem: a *write-before-query barrier*. Summaries are produced before the system knows what a future query will hinge on, so decisive constraints can be dropped and the omission is unverifiable. Their design keeps summaries as the fast path, keeps the raw log immutable, routes to the raw log only when a sufficiency check fails, and writes verified findings back as new summary units linked to their raw sources. On LoCoMo this recovers nearly all of raw-only accuracy (0.851 vs 0.873) at 54% fewer input tokens.

### 4.4 Learned compaction

Rather than prompt a summariser, several systems train the model to maintain a compact state. MEM1 (Zhou et al., 2025b, ICLR 2026) trains an agent with RL to update a shared internal state at every turn that integrates memory and reasoning, discarding the rest, so context size stays near constant over multi-objective tasks; it reports a 3.5× memory reduction and a 3.7× performance improvement over Qwen2.5-14B-Instruct at 16-objective tasks. MemAgent (Yu et al., 2025) trains an overwritable token-level memory for reading documents far longer than the window, with multi-conversation RL. Both are compaction policies learned end to end, and both are evaluated on tasks where the needed information is bounded and the reward is dense.

### 4.5 Assessment

Context-resident memory is unavoidable and well understood. The literature's consensus points are that compaction must be anchored (keep structure and raw evidence reachable), incremental (edit rather than rewrite), and verifiable (link summaries to sources), and that the harness, not the model, should own the policy for when compaction and note-taking happen. The write-before-query barrier is the clearest argument for retaining the full event log: it is the only way to make summaries correctable after the fact.

## 5. External Retrieval-Based Memory

This is the largest family and the one where most products live. Jiang et al.'s (2026) four structures organise it well: lightweight semantic, entity-centric, episodic and reflective, and structured and hierarchical. A fifth, filesystems and wikis, has become prominent enough since 2025 to treat separately.

### 5.1 Lightweight semantic stores

The simplest design embeds text units and retrieves the top-k by cosine similarity. Generative Agents (Park et al., 2023) is the canonical early example and introduced a retrieval score that combines *recency*, *importance* (an LLM-rated salience), and *relevance*, plus periodic *reflection* that synthesises higher-level observations from recent memories. These three scoring signals and the reflection step have been reused in dozens of systems since.

Mem0 (Chhikara et al., 2025) is the most widely deployed modern instance. Its pipeline has two phases: an extraction LLM reads each new message pair with a rolling summary and recent messages and proposes candidate facts; an update LLM compares each candidate against similar existing memories and chooses ADD, UPDATE, DELETE, or NOOP. The authors' own LoCoMo results (gpt-4o-mini as answer model, LLM judge) were 66.9 for Mem0 and 68.4 for the graph variant Mem0g, against 72.9 for the full conversation in context; the pitch was cost, with Mem0 using roughly 1.8k tokens and 1.4 s p95 per query against about 26k tokens and 17 s for full context (Chhikara et al., 2025; figures as reported in Kumar, 2026). The paper is a preprint; the comparative numbers in it were disputed, as Section 9 discusses.

Strengths of this structure are simplicity and generality. Weaknesses are documented repeatedly: no relational structure, so multi-hop questions require the model to reassemble facts; no model of time, so stale and current facts compete with near-identical similarity; and dependence on the extraction prompt for what is remembered at all. Jiang et al. (2026) find this structure the most backbone-sensitive.

### 5.2 Entity-centric, graph, and temporal stores

Knowledge-graph memory addresses multi-hop and temporal questions by storing entities and typed relations.

HippoRAG (Gutiérrez et al., 2024, NeurIPS 2024) builds an open-information-extraction KG over passages and retrieves by Personalized PageRank seeded from query entities, reporting large gains on multi-hop QA at lower cost than iterative retrieval. HippoRAG 2 (Gutiérrez et al., 2025, ICML 2025), subtitled "From RAG to Memory," adds deeper passage integration and online LLM use, and positions the system as non-parametric continual learning that handles factual, sense-making, and associative tasks in one framework. Microsoft's GraphRAG (Edge et al., 2024) adds community detection and pre-computed community summaries for global, sense-making questions.

Zep (Rasmussen et al., 2025) is the production system most explicit about time. Its Graphiti engine maintains a *bi-temporal* graph: each edge carries the time the fact was true in the world and the time it was recorded or invalidated in the system, so a contradiction does not delete the old edge but closes its validity interval. The graph has episode, semantic-entity, and community layers. On the DMR benchmark Zep reported 94.8% against MemGPT's 93.4%; on LongMemEval with gpt-4o it reported 71.2% against 60.2% for full context, with latency falling from 28.9 s to 2.6 s and prompt size from about 115k tokens to 1.6k. The same table shows where graph memory loses: on single-session questions about what the assistant itself said, accuracy fell from 94.6% to 80.4%, because extraction can miss the assistant's own statements (Rasmussen et al., 2025; Kumar, 2026).

AMA-Agent (Zhao et al., 2026) extends the graph idea to agent trajectories rather than dialogue, constructing a *causality graph* over states, actions, and observations with tool-augmented retrieval. Its motivation, from the AMA-Bench study, is that existing systems "fail to capture causal and objective information and rely heavily on lossy similarity-based retrieval." Kumiho (Jang, 2026) goes furthest in formalisation: a property-graph memory with immutable revisions, typed dependency edges, and URI addressing, whose operational semantics are shown to satisfy the AGM belief-revision postulates; it also introduces *prospective indexing* (indexing LLM-generated implications of a memory at write time so that implicit constraints can be found later).

The consistent finding is that graphs win on temporal and multi-hop questions and on explainability (edges can be cited), and cost more to build and maintain; Jiang et al. (2026) measure the ingestion latency and throughput overhead and find it is often omitted from papers.

### 5.3 Episodic, reflective, and hierarchical stores

A third structure keeps records of what happened and layers abstractions over them.

MemoryBank (Zhong et al., 2023) stored dialogue, daily summaries, and a user-personality profile, with Ebbinghaus-style decay. A-MEM (Xu et al., 2025, NeurIPS 2025) borrows the Zettelkasten method: each memory is a note with LLM-generated keywords, tags, and a contextual description; on insertion the system retrieves related notes, creates links, and may *evolve* existing notes' descriptions in light of the new one. The memory thus reorganises itself over time rather than only growing. HiMem (Zhang et al., 2026) segments the interaction stream into episodes using topic shifts and "surprise," extracts stable "note memory" from them, links the two layers, and performs conflict-aware reconsolidation driven by retrieval feedback. LightMem (Fang et al., 2025) pushes the consolidation step offline ("sleep-time update") and reports large token and call reductions relative to Mem0-style per-turn pipelines.

Hindsight (Vectorize, Mariani et al., 2025) organises memory into four logical networks that distinguish *world facts*, *agent experiences*, *synthesised entity summaries*, and *evolving beliefs*, with retain/recall/reflect operations, and emphasises that current systems "blur the line between evidence and inference." The separation of what was observed from what was inferred is the design point most relevant to auditability, and it is rare in the literature. Hindsight reports 91.4% on LongMemEval and 89.6% on LoCoMo with a large backbone; these are the authors' own runs.

MIRIX (Wang and Chen, 2025) and MemOS (Li et al., 2025) are the maximalist designs. MIRIX has six memory components (core, episodic, semantic, procedural, resource, knowledge vault) each managed by a dedicated agent under a meta-manager, with active retrieval, and extends to screenshots. MemOS proposes a "MemCube" abstraction unifying plaintext, activation (KV), and parametric memory with lifecycle scheduling and governance, presented as a memory operating system. Both are preprints with author-run evaluations; their contribution is architectural vocabulary more than measured gains.

### 5.4 Filesystems, notes, and agent-maintained wikis

The most important empirical finding of 2025 in this family came from Letta (2025). Lacking a native way to ingest LoCoMo, they placed each conversation into a file, gave a gpt-4o-mini agent `grep`, semantic `search_files`, `open`, and `close`, constrained it to search before answering, and scored 74.0%, above Mem0's then-reported 68.4% for its best variant and above the full-context baseline. Their explanation: models post-trained on coding use filesystem tools fluently, generate their own queries, and iterate until they find the evidence, whereas purpose-built memory tools designed for single-hop retrieval are less effective. Kumar (2026) reports a related result from 2026 evaluations where BM25 over raw transcripts outperformed every dedicated memory product tested. Wooders (2026) generalises the point in an essay arguing that "memory isn't a plugin" because the harness already controls what loads, compacts, and persists.

Vendors moved the same way. Anthropic's memory tool is a directory of files the model reads and writes (Anthropic, 2025b); Claude Code's auto memory is per-repository notes the agent writes from user corrections, distinct from the human-written CLAUDE.md/AGENTS.md instruction files that act as procedural memory (Anthropic, 2026). Cursor, Codex, and other coding harnesses have equivalents.

Karpathy's April 2026 "LLM Wiki" gist (Karpathy, 2026) articulates the strongest version. Rather than RAG over raw documents, in which "the LLM is rediscovering knowledge from scratch on every question," the agent incrementally builds and maintains a wiki of interlinked markdown pages. There are three layers: immutable raw sources (the source of truth, never modified by the LLM), the wiki (owned entirely by the LLM), and a schema file (CLAUDE.md or AGENTS.md) that defines conventions and workflows. Three operations: *ingest* a source (touching 10 to 15 pages), *query* (with the insight that good answers should be filed back as pages so explorations compound), and *lint* (find contradictions, stale claims, orphan pages, missing cross-references). The pattern was adopted by several teams within months (Mem0, 2026b; akitaonrails, 2026).

Why this works deserves a careful statement, because it cuts against the research trend toward complex stores. Three reasons are supported by the evidence above. First, capability is in the model: iterative search with self-generated queries over raw text recovers from retrieval misses, while one-shot top-k cannot. Second, human-readable artifacts can be reviewed, corrected, and version-controlled with existing tools. Third, the wiki's "compile once, keep current" discipline directly attacks the staleness and contradiction problems that similarity stores cannot see (Section 10.3). The costs are equally clear: a wiki is only as good as its maintainer's lint discipline, there is no retrieval log unless one is added, and nothing in the pattern prevents a poisoned source from being compiled into a confident page.

### 5.5 Assessment

Across the four structures the evidence supports a layered view rather than a winner. Raw evidence (transcripts, events) must be retained for grounding and correction; summaries and extracted facts are the fast path; entities and time-aware relations are needed when questions are temporal or multi-hop; and agent-maintained synthesis (reflections, wiki pages) is where accumulated understanding lives. The systems that perform best on the harder 2026 benchmarks combine these (Hindsight, AMA-Agent, TierMem, Kumiho) and keep links between layers. Systems that keep only one layer, especially only extracted facts, show the documented failures: missed assistant statements, stale facts, unverifiable omissions.

## 6. Experiential and Procedural Memory

Luo et al. (2026) frame the field's evolution as Storage (keep trajectories), Reflection (refine them), Experience (abstract across them). This section covers the Reflection and Experience stages: memory whose content is *lessons* rather than facts.

### 6.1 Verbal reinforcement and insight extraction

Reflexion (Shinn et al., 2023, NeurIPS 2023) had an agent write a natural-language self-reflection after each failed attempt and store it in an episodic buffer that is prepended on the next attempt, improving HumanEval pass@1 to 91% and ALFWorld success substantially. ExpeL (Zhao et al., 2023, AAAI 2024) generalised from within-task to cross-task: it collects trajectories, compares successful and failed ones, and extracts "insights" that are added, upvoted, downvoted, or edited in a persistent list. Voyager (Wang et al., 2023) built a skill library of verified executable code for Minecraft, retrieved by embedding of the skill's description, which is procedural memory in CoALA's sense.

### 6.2 Workflows, cheatsheets, and strategy banks

Agent Workflow Memory (Wang et al., 2024) induces reusable sub-routines ("workflows") from past trajectories and adds them to the agent's memory, either offline from training examples or online from the agent's own successes; on WebArena and Mind2Web it improved success by 24.6% and 51.1% relative, and generalised better than baselines as the train-test gap widened. Dynamic Cheatsheet (Suzgun et al., 2025) maintains a persistent, curated memory of strategies and code snippets that the model updates at test time, more than doubling accuracy on AIME 2024 for Claude 3.5 Sonnet and giving large gains on Game of 24. ReasoningBank (Ouyang et al., 2025, Google, ICLR 2026) distils generalisable *reasoning strategies* from both successes and failures (self-judged), retrieves them for new tasks, and adds *memory-aware test-time scaling* (MaTTS), in which parallel or sequential attempts at the same task generate richer contrast for memory synthesis; it reports up to 34.2% relative gains in effectiveness and 16% fewer steps on WebArena, Mind2Web, and SWE-Bench-Verified.

### 6.3 Case-based reasoning and curated contexts

Memento (Zhou et al., 2025a) frames continual adaptation as case-based reasoning over a memory-based MDP: a case bank of past (state, action, reward) triples, a learned Q-function that selects which cases to retrieve, and no update to the LLM. It reports top-1 on GAIA validation (87.9%) and strong results on DeepResearcher and SimpleQA. ACE (Section 4.3) belongs here as well: its playbook is procedural memory, and its contribution is a curation discipline that avoids collapse.

### 6.4 Assessment

Experiential memory is where the "self-evolving agent" promise lives, and the measured gains on repeated-task distributions are real and large. Three cautions recur. First, the gains are evaluated on task streams drawn from the same distribution as the experiences; transfer to new domains is weaker (Wang et al., 2024 measure it; most do not). Second, failure-derived lessons require a judge of failure, and self-judgement is noisy; ReasoningBank uses an LLM judge and notes this. Third, a wrong lesson is a persistent wrong lesson: Du (2026) lists "trustworthy reflection" as an open challenge, and the security literature (Section 10.1) treats experiential stores as a prime injection target because one planted "lesson" influences every future task. Provenance for lessons (which trajectories they came from, who judged them) is almost never recorded in these systems.

## 7. Learned Memory-Management Policies

The newest family trains the memory controller rather than prompting it. Du (2026) calls it "policy-learned management" and Hu et al. (2025b) list RL integration as a frontier.

Memory-R1 (Yan et al., 2025, ACL 2026) trains two components: a Memory Manager that chooses ADD, UPDATE, DELETE, or NOOP for each extracted fact, and an Answer Agent that filters retrieved memories before answering, using PPO or GRPO with outcome rewards. It reports state-of-the-art results on LoCoMo from only 152 training question-answer pairs, with transfer across backbones. Mem-α (Wang et al., 2025b) trains an agent to construct a multi-component memory (core, episodic, semantic) with RL and reports generalisation from 30k-token training instances to 400k-token evaluations. AgeMem (Yu et al., 2026, ACL 2026 SAC Highlight) unifies long- and short-term management as tool actions (store, retrieve, update, summarise, discard) within the agent's own policy, trained by a three-stage curriculum with step-wise GRPO to handle sparse rewards. MEM1 and MemAgent (Section 4.4) are the context-compaction instances.

The Proactive Memory Agent (Wu et al., 2026b) is a different shape: a separate memory agent runs alongside an unmodified action agent, maintains a structured memory bank from the trajectory, and decides whether to inject a reminder or stay silent. It names the failure mode it targets, "behavioral state decay," and reports +8.3 pp pass@1 on Terminal-Bench 2.0 and +6.8 pp on τ²-Bench, with ablations showing that *selective* intervention beats always-on injection and passive exposure of the bank. The authors also train an open-weight memory policy (Qwen3.5-27B) with SFT and GRPO, with partial transfer.

Sleep-time compute (Lin et al., 2025, Letta) is complementary: it uses idle time between interactions to pre-process context into a learned context, reporting about 5× reduction in test-time compute at matched accuracy on Stateful GSM-Symbolic and AIME, and amortisation across queries about the same context. LightMem's offline consolidation and Karpathy's lint step are the same idea in different clothing.

### 7.1 Assessment

Learned controllers address a real weakness of prompted pipelines, which make the same kind of errors regardless of outcome. The evidence base is young: most results are on dialogue QA benchmarks with dense rewards; data-efficiency claims (152 examples) deserve replication; and the trained policy inherits the benchmark's notion of what is worth remembering. The Proactive Memory Agent is notable for evaluating on agentic benchmarks and for ablating against the obvious alternatives. Jiang et al. (2026) warn that RL-optimised compression tends to leave memory "largely unstructured and textual," which trades auditability for efficiency.

## 8. Multi-Agent and Shared Memory

When several agents, or several harnesses driven by one user, share memory, new problems appear. G-Memory (Zhang et al., 2025a) organises multi-agent memory as a three-tier graph (insight, query, interaction) inspired by organisational memory theory, reporting gains on embodied and knowledge tasks. MIRIX's six managers are themselves a multi-agent memory system. MemTX (Li et al., 2026) makes the governance argument directly: in shared memory "one agent's write becomes another agent's premise, and eventually a tool call with real side effects," so a write should not be a belief commit. It stages writes in snapshot-isolated transactions with evidence, permissions, provenance, and validity, gates irreversible tool calls on in-flight belief state, and cascades repair when a belief is retracted; two invariants are checked by property-based testing over 5.5 million protocol states. Portable Agent Memory (Ravindran, 2026) proposes a cross-vendor transfer protocol with content-addressed entries in a Merkle-DAG provenance graph, capability-scoped disclosure, and an injection-resistant "rehydration" step for moving memory between GPT, Claude, Gemini, and Llama agents.

These are preprints, but they converge on a view the security literature shares: shared memory needs transactional semantics, provenance, and trust levels per writer, because the blast radius of a bad write is every reader.

## 9. Evaluation

### 9.1 Benchmarks

Table 2 lists the benchmarks that appear most in the literature.

| Benchmark | Year, venue | What it tests | Horizon | Notes |
|---|---|---|---|---|
| DMR (MemGPT) | 2023 | Recall of a fact from a prior session | Short | Near-saturated (Zep 94.8%) |
| LoCoMo (Maharana et al.) | 2024, ACL | Single-hop, multi-hop, temporal, open-domain, adversarial QA over 10 multi-session conversations | Tens of sessions; each conversation fits in a 2026 context window | Category 5 lacks usable ground truth; speaker-attribution errors (Zep, 2025) |
| LongMemEval (Wu et al.) | 2024/ICLR 2025 | Information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention; 500 questions | ~115k tokens (S) to ~500 sessions (M) | Headline: assistants drop ~30% over sustained interaction |
| MemoryAgentBench (Hu et al.) | 2025/ICLR 2026 | Accurate retrieval, test-time learning, long-range understanding, conflict resolution, under incremental multi-turn input | Long | All systems under 45 on retrieval split (Kumar, 2026) |
| BEAM (Taheri et al.) | 2025 | Recall and reasoning over conversations up to 10M tokens | Extreme | Introduces LIGHT framework; high evaluation cost |
| LoCoMo-Plus (Li et al.) | 2026 | Implicit constraints under cue-trigger semantic disconnect | Long | Shows string-match metrics misaligned |
| AMA-Bench (Zhao et al.) | 2026 | Memory over agent trajectories (states, actions, observations, tool outputs), real and synthetic to any length | Any | Best system 57.2% |
| MemoryArena (He et al.) | 2026, ICML | Interdependent multi-session agentic tasks: web, planning, search, formal reasoning | Multi-session | Systems near-saturated on LoCoMo "perform poorly" |
| Bemula (2026) | 2026 | Memory unlearning: 135 forget requests, 1,742 questions, paired control dialogues | Multi-session | No system passes >1.5% of requests end to end |
| Letta Leaderboard | 2025 | Model's memory-management ability with framework and tools fixed | Varies | Vendor-run |
| Terminal-Bench 2.0, τ²-Bench, SWE-Bench Verified | 2025 | Downstream agentic tasks used to measure memory's effect on action | Single task | Used by ReasoningBank, Proactive Memory Agent |

Table 2. Benchmarks used to evaluate agent memory.

Two shifts are visible. From recall to action: MemoryArena, AMA-Bench, and the use of Terminal-Bench test whether memory improves decisions, not whether a fact can be found, and current systems score far lower there. From storage to lifecycle: LongMemEval's knowledge-update and abstention categories, MemoryAgentBench's conflict resolution, and Bemula's forgetting test the operations that Du et al. (2025) enumerated and that most 2024 systems ignored.

### 9.2 The credibility problem

The comparative evidence in this field is unusually weak, and the weakness is documented by the participants themselves.

Mem0's April 2025 paper reported LoCoMo scores for competitors. Zep's rebuttal (Zep, 2025) identified three errors in how Zep had been run: both speakers ingested as the user, timestamps appended to message text rather than set in the `created_at` field, and sequential rather than parallel searches; rerun, Zep scored 75.1 against the 66.0 Mem0 had reported for it. Letta (2025) stated that the MemGPT numbers in the same paper could not have been produced without a code path that did not exist, and that Mem0 did not respond to requests for the modified implementation. By 2026, Mem0's managed platform reported 92.5 on LoCoMo and 94.4 on LongMemEval (Mem0, 2026a), Hindsight reported 89.6, and MemOS 88.8; none of these used the same answer model, judge, or question subset as the 2025 rows, and only Zep's and Letta's numbers were produced to check someone else's (Kumar, 2026). Jiang et al. (2026) formalise the pathologies: benchmark saturation, judge sensitivity and metric validity (LoCoMo-Plus shows string matching misaligned with the construct), backbone-dependent accuracy, and omitted ingestion cost. A 2026 essay titled "The Benchmark Theatre" (Bloo-mind, 2026) argues that almost no published memory score should be compared with another.

The practical consequence is stated plainly by Letta (2025): "Comparing agent frameworks and agent memory tools is like comparing apples to oranges, as you can always mix and match frameworks and tools (and, of course, models)." A defensible comparison fixes the model, the judge, the ingestion path, and the question set, and reports token and latency cost alongside accuracy. Few papers do all four.

### 9.3 What good evaluation looks like

From the methodological papers (Hu et al., 2025a; Jiang et al., 2026; He et al., 2026; Zhao et al., 2026; the Bemula authors), a checklist emerges:

- Incremental ingestion, not a pre-loaded corpus, so the write path is tested.
- Interleaved memory and action, so use of memory is tested, not only recall.
- Knowledge updates and contradictions in the data, with questions whose correct answer changed.
- Abstention items, so hallucinated recall is penalised.
- Forget requests with paired controls.
- Cost reported: tokens per query, ingestion latency, storage growth.
- Fixed backbone and judge across compared systems, with judge validated against human labels.
- Several backbones, because rankings change with the model.

## 10. Security, Temporal Validity, Provenance, and Forgetting

The 2026 literature's most important development is the recognition that memory is a governance problem. Lin et al. (2026, EMNLP 2026) characterise the threat landscape by three properties that distinguish it from prompt-level security: *persistence* (a bad write survives the session), *statefulness* (it changes future behaviour), and *propagation* (it spreads to other agents and sessions). Their Memory Lifecycle Framework crosses six phases (Write, Store, Retrieve, Execute, Share & Propagate, Forget & Rollback) with four objectives (Integrity, Confidentiality, Availability, Governance) and concludes that security "cannot be retrofitted at retrieval or execution time alone, but must be anchored in storage-time provenance, versioning, and policy-aware retention from the outset." They propose Verifiable Memory Governance as five architectural primitives for auditable, recoverable control of memory state.

### 10.1 Poisoning

AgentPoison (Chen et al., 2024, NeurIPS 2024) showed that a small number of planted entries in an agent's RAG store or memory can act as a backdoor triggered by optimised query patterns, with high success at under 0.1% poisoning rate; it assumed write access to the store. MINJA (Dong et al., 2025) removed that assumption: through ordinary user queries alone, an attacker induces the agent to write "indication prompts" into its own memory that later steer other users' sessions, reporting over 95% injection success and 70% attack success under idealised conditions. Sunil et al. (2026) replicate MINJA on EHR agents over MIMIC-III with three models and find that realistic conditions, a store already populated with legitimate memories, "dramatically reduce attack effectiveness"; they evaluate two defences, composite trust scoring at input/output and memory sanitisation with trust-aware retrieval and temporal decay, and find that thresholds must be calibrated to avoid both rejecting everything and missing subtle attacks. The SSGM framework (Lam, 2026) proposes decoupling memory evolution from execution, with consistency verification, temporal decay, and access control enforced before consolidation.

The design implication accepted across this literature is that writes should carry a trust level determined by their source (user statement, agent inference, tool output, import), that tool-derived content is the main injection channel and should be quarantined pending review, and that retrieval should be able to filter on trust and status.

### 10.2 Confidentiality and leakage

Lam (2026) names "topology-induced knowledge leakage," where sensitive context from one setting is consolidated into long-term memory and resurfaces in another. Hu et al. (2025b) list trustworthiness and privacy as frontier issues. Production guidance converges on redaction before storage (API keys, credentials, personal data), scoping memory by project and user, and recording what was redacted so audits can account for gaps.

### 10.3 Temporal validity and contradiction

The clearest negative result in the field concerns similarity search. Hancock (2026, MemStrata) shows on a calibrated dataset that cosine similarity distinguishes a *contradicting* fact from a *rephrased duplicate* with AUROC 0.59, near chance, because contradictions are often more similar to the original than paraphrases are. The consequence: a vector store presented with "the API is now `v2`" and "the API is `v1`" retrieves both with near-identical scores, and when forced to answer serves the stale value 15 to 40% of the time. MemStrata's remedy is a deterministic (subject, relation, object) supersession rule into a *bi-temporal ledger*, with no similarity threshold and no LLM call, reducing stale-fact errors to about zero at retrieval latency. Zep's Graphiti reached the same bi-temporal design from engineering practice (Rasmussen et al., 2025), and MemTX types contradiction-resolution heuristics as bitemporal write-time operators (Li et al., 2026). HiMem's reconsolidation and Karpathy's lint step attack staleness from the agent side.

The general lesson is that staleness is a modelling problem, not a retrieval-tuning problem. The store must know that a fact has a validity interval and that a later fact can close it, and must keep the closed fact for audit.

### 10.4 Provenance and belief revision

Several 2026 systems treat provenance as a first-class structure. TierMem links every summary unit to its immutable raw sources and escalates to them when summaries are insufficient (Zhu et al., 2026). Kumiho gives every belief a URI, a revision history, provenance edges to evidence, and an immutable audit trail, and proves AGM postulate satisfaction for its revision operations (Jang, 2026). Hindsight separates evidence from inference in distinct networks (Mariani et al., 2025). MemTX requires evidence, permissions, provenance, and validity on every record, and gates side effects on commit state (Li et al., 2026). Portable Agent Memory uses content addressing and Merkle-DAG parent hashes for tamper evidence (Ravindran, 2026). W3C PROV-style provenance for agent traces is proposed in Chen et al. (2026). These are preprints, and their benchmark scores are not the point. Their shared claim, supported by the failure analyses in Sections 4.3 and 10.3, is that a memory system without provenance cannot explain, correct, or safely share what it remembers.

### 10.5 Forgetting and deletion

Forgetting appears in the surveys as an atomic operation (Du et al., 2025) and in early systems as decay (MemoryBank). Deletion on request is different: it is a correctness requirement with legal weight. The Bemula benchmark (Nboxff et al., 2026) tests it directly with 135 forget requests of six types across 50 dialogues, paired with control dialogues so that only systems that stored the target can score, and five suites for storage, removal, preservation of unrelated memories, relearning, and extraction resistance. Across six mainstream memory systems and three backbones, "no system passes more than 1.5% of forget requests end to end, and most pass none at all": systems either leave the information extractable under simple attacks or delete far more than requested. Agentic Unlearning (Zhou et al., 2026) identifies "backflow," where content deleted from explicit memory is reintroduced from the model's parameters, and Can Agents Truly Forget? (Wang et al., 2026) extends unlearning to agent settings. Deletion is, on this evidence, unsolved.

An append-only design interacts with this in a specific way. It cannot delete; it can retire, supersede, and exclude from retrieval, which satisfies "stop using this" but not "erase this." Content erasure requires an out-of-band administrative operation, and the audit log must then record that a redaction occurred without recording what was redacted. Lin et al.'s "Forget & Rollback" phase names exactly this tension.

## 11. Cross-Cutting Analysis

### 11.1 Harness versus store

The single most consequential empirical pattern across Sections 4, 5, and 9 is that the harness matters more than the store. Letta's filesystem result, the BM25 results reported by Kumar (2026), the ablations in the Proactive Memory Agent showing that *when* to inject matters more than *what* is stored, and the vendors' convergence on file-based memory all point the same way. MemGPT's original insight, that a capable model managing simple tools does well, has held for three years. This does not mean stores are irrelevant: on temporal and multi-hop questions graphs win, and at 10M-token scale indexing is unavoidable. It means that evaluations which vary the store while holding the harness fixed, or vice versa, are the only ones that isolate anything.

### 11.2 Lossy versus lossless

Every system makes a lossy choice somewhere: extraction drops what the prompt did not ask for, summarisation drops detail, embeddings drop structure, decay drops the unreinforced. The write-before-query barrier (Zhu et al., 2026) explains why this is not fixable by better prompts. The systems that recover are those that keep raw evidence immutable and reachable (TierMem, Kumiho, Karpathy's raw sources layer, Hindsight's experience network) and treat everything else as a cache over it. Full-context baselines keep winning on recall benchmarks for the same reason; memory systems win on cost and on long horizons where full context is impossible or rots.

### 11.3 Who writes

Control policy is the least evaluated axis. Pipelines that run an extraction LLM on every turn (Mem0, Zep) are consistent and expensive and remember what the prompt was written to notice. Agent-directed writes (MemGPT, memory tools, auto memory) are cheaper and adaptive and depend on the model's judgement of salience, which Wu et al. (2026b) show decays over long trajectories. Learned policies (Section 7) are promising and young. Human-written procedural memory (AGENTS.md) is the most reliable and the least scalable. No paper compares all four on the same task.

### 11.4 Cost

Jiang et al. (2026) find that ingestion latency and throughput are routinely omitted from papers and can dominate in production. The per-turn extraction pipelines are the most expensive; LightMem reports two orders of magnitude fewer LLM calls by consolidating offline. Sleep-time compute and nightly lint are the field's answer: do the expensive consolidation when no one is waiting.

### 11.5 Strength of evidence by claim

- *Memory systems reduce tokens and latency versus full context by an order of magnitude*: strong, replicated across vendors and independent runs.
- *A specific memory product beats others on recall*: weak; see Section 9.2.
- *Graph or temporal structure helps temporal and multi-hop questions*: moderate; consistent across Zep, HippoRAG, AMA-Agent, Kumiho, with author-run evaluations.
- *Experiential memory improves repeated-task performance*: strong within distribution (Reflexion, AWM, ReasoningBank, Dynamic Cheatsheet, Memento); weak out of distribution.
- *Similarity search cannot detect contradiction*: strong, with a calibrated measurement (AUROC 0.59).
- *Deletion is unsolved*: strong, one benchmark, negative across all systems tested.
- *Memory poisoning is practical*: strong for idealised conditions; moderate evidence that realistic stores reduce it.
- *Provenance-based governance improves outcomes*: argued, formalised, and partially verified (MemTX property tests); not yet shown on shared agentic benchmarks.

## 12. Open Problems

1. **Causally grounded retrieval.** AMA-Bench's diagnosis is that systems retrieve by similarity when they need causal and objective structure over trajectories. Causality graphs are an early answer (Zhao et al., 2026); Du (2026) lists this as a frontier.
2. **Continual consolidation without collapse.** Consolidation that abstracts without the brevity bias and context collapse ACE documents, with lossless links back to evidence.
3. **Trustworthy reflection.** Lessons derived from self-judged outcomes need provenance, confidence, and a way to be retired when later evidence contradicts them.
4. **Learned forgetting and verified deletion.** Bemula's negative result stands; backflow from parameters means explicit-memory deletion is necessary but not sufficient.
5. **Temporal reasoning.** LongMemEval's temporal category and the staleness result show that explicit validity intervals are needed; most stores still lack them.
6. **Evaluation that isolates components.** Fixed-harness comparisons of stores and fixed-store comparisons of harnesses, with cost, on agentic tasks, with validated judges.
7. **Shared and portable memory.** Transactional semantics, per-writer trust, and injection-resistant transfer across vendors are specified in preprints and untested at scale.
8. **Parametric consolidation.** Compiling stable knowledge into weights or caches (Cartridges, memory layers) while keeping the token-level ledger as the auditable source; nobody has closed this loop in an agent.
9. **Multimodal and embodied memory.** MIRIX's screenshots and the embodied agents in G-Memory are early; most of the field is text.

## 13. Implications for an Append-Only, Auditable, Multi-Harness Ledger

This section maps the review's findings onto a design in which several agent harnesses write to one local append-only database with events, curated memories, retrieval logs, and context links, as in this project's PLAN.md.

**Keep the raw event log; it is the layer the literature says cannot be reconstructed.** The write-before-query barrier (Section 4.3), the Letta and BM25 results (Section 5.4), and TierMem's design all argue that an immutable record of what happened, with tool calls and results linked, is the foundation. Everything else, extracted memories, summaries, wiki pages, is a cache that must link back to it. The planned `memory_sources` table and `blob_ref` storage for large content are the right primitives; the review suggests making the raw-log path a first-class recall strategy (grep or FTS over `events`), not only a provenance pointer, because iterative search over raw text is the most robust retrieval behaviour measured.

**Trust levels and quarantine for tool-derived memories are supported by the security literature, with one refinement.** MINJA and AgentPoison make tool output and query-induced writes the main channels; trust-aware retrieval with calibrated thresholds is the defence Sunil et al. (2026) evaluate. The refinement is MemTX's: a write should not become an actionable belief until it is admitted, and irreversible actions should be gated on belief state. The planned `memory_status` derivation (quarantined by default for `tool` trust) implements admission; exposing status to the recall API so an agent can see that a memory is unreviewed implements the gate on the reading side.

**Model temporal validity explicitly, do not infer it from similarity.** The AUROC 0.59 result means dedupe and contradiction detection by embedding similarity, as planned for Phase 4, will not distinguish a correction from a paraphrase. The literature's answer is a bi-temporal record: `valid_from`/`valid_to` in world time and `recorded_at` in system time, with `supersedes` closing the earlier interval. The planned `supersedes` link and validity columns are the right shape; deterministic supersession on a (scope, key) pair, as MemStrata does on (subject, relation), should be the primary mechanism and similarity a secondary signal for review queues.

**Separate evidence from inference.** Hindsight's four networks and Kumiho's belief graph distinguish what was observed from what was concluded. The `memory_sources` relation types (`user_stated`, `derived_from`, `supports`, `contradicts`) encode this; the review suggests surfacing it in the Markdown projection so a reader can see whether a page states an observation or an inference, which is also what Karpathy's lint step needs.

**Log retrieval, not only storage.** No surveyed system records what was retrieved, with what scores, and whether it was injected, which is why the "why did the agent say that" question is unanswerable in most of them. The planned `retrievals`, `retrieval_items`, and `context_links` tables are unusual in the literature; `why_answer` reporting status at recall and status now addresses exactly the audit case the governance papers describe.

**Prefer offline consolidation.** LightMem, sleep-time compute, and the wiki lint step all move expensive synthesis out of the interaction loop. The Phase 5 Markdown projection and Phase 4 quality work fit a nightly job over the Parquet export rather than per-turn LLM calls.

**Plan for the deletion gap honestly.** Append-only storage can retire and exclude but not erase. The review's evidence is that no system deletes correctly today, so the project is not behind the field, but the administrative redaction path should exist, should be logged as an event with counts rather than content, and should be tested with paired controls in the style of Bemula.

**Evaluate on your own traffic.** Given the credibility problem, external benchmark scores for the extraction or recall components will not be comparable. The ledger itself enables the evaluation the methodological papers ask for: incremental ingestion is the capture path; interleaved action is the agents' real work; and `retrieval_items.injected` with later `outcome` events gives a direct measure of whether recalled memories changed behaviour. That is a stronger signal than any published number.

## References

Where a work has a peer-reviewed venue it is given; otherwise the arXiv identifier is given and the work should be read as a preprint. Vendor and practitioner sources are marked.

- akitaonrails (2026). *ai-memory: research notes on Karpathy's LLM Wiki and the 2026 landscape.* GitHub repository. Practitioner.
- Anthropic (2025a). *Effective context engineering for AI agents.* Engineering blog, 29 September 2025. Vendor.
- Anthropic (2025b). *Managing context on the Claude Developer Platform* (context editing and memory tool). 29 September 2025. Vendor.
- Anthropic (2026). *How Claude remembers your project* (CLAUDE.md, AGENTS.md, auto memory). Claude Code documentation. Vendor.
- Behrouz, A., Zhong, P., Mirrokni, V. (2024). Titans: Learning to Memorize at Test Time. arXiv:2501.00663.
- Behrouz, A. et al. (2025). Nested Learning: The Illusion of Deep Learning Architectures. NeurIPS 2025; arXiv:2512.24695.
- Berges, V.-P. et al. (2024). Memory Layers at Scale. arXiv:2412.09764.
- Bloo-mind (2026). *The Benchmark Theatre: why almost nothing you've read about agent memory scores is comparable.* Essay. Practitioner.
- Bulatov, A., Kuratov, Y., Burtsev, M. (2022). Recurrent Memory Transformer. NeurIPS 2022.
- Chen, Z. et al. (2024). AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases. NeurIPS 2024.
- Chen et al. (2026). From Agent Traces to Trust: Evidence Tracing and Execution Provenance. arXiv:2606.04990.
- Chhikara, P., Khant, D., Aryan, S., Singh, T., Yadav, D. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.
- Das, P. et al. (2024). Larimar: Large Language Models with Episodic Memory Control. ICML 2024.
- Dong, S. et al. (2025). A Practical Memory Injection Attack against LLM Agents (MINJA). arXiv:2503.03704.
- Du, P. (2026). Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers. arXiv:2603.07670.
- Du, Y., Huang, W., Zheng, D., Wang, S., Montella, S., Lapata, M., Wong, K.-F., Pan, J. Z. (2025). Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions. arXiv:2505.00675.
- Edge, D. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.
- Eyuboglu, S. et al. (2025). Cartridges: Lightweight and general-purpose long context representations via self-study. arXiv:2506.06266.
- Fang, J. et al. (2025). LightMem: Lightweight and Efficient Memory-Augmented Generation. arXiv:2510.18866.
- Gupta, A., Rao, A., Anumanchipalli, G. (2024). Model Editing at Scale leads to Gradual and Catastrophic Forgetting. ACL Findings 2024.
- Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024.
- Gutiérrez, B. J., Shu, Y., Qi, W., Zhou, S., Su, Y. (2025). From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2). ICML 2025.
- Hancock (2026). Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge (MemStrata). arXiv:2606.26511.
- He, Z. et al. (2026). MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks. ICML 2026; arXiv:2602.16313.
- Hong, K., Troynikov, A., Huber, J. (2025). *Context Rot: How Increasing Input Tokens Impacts LLM Performance.* Chroma technical report, July 2025. Vendor research.
- Hsieh, C.-P. et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models? COLM 2024.
- Hu, Y. et al. (2025a). Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench). ICLR 2026; arXiv:2507.05257.
- Hu, Y., Zhang, G. et al. (2025b). Memory in the Age of AI Agents. arXiv:2512.13564.
- Jang (2026). Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures (Kumiho). arXiv:2603.17244.
- Jiang, D. et al. (2026). Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations. arXiv:2602.19320.
- Kang, M. et al. (2025). ACON: Optimizing Context Compression for Long-horizon LLM Agents. arXiv:2510.00615.
- Karpathy, A. (2026). *LLM Wiki* (idea file). GitHub gist, April 2026. Practitioner.
- Kumar, A. (2026). *Agent Memory in 2026: Vendors Claim 92.5 on LoCoMo, a Plain Filesystem Scored 74, and Every Memory Product Lost to BM25.* Blog, 5 October 2026. Practitioner.
- Lam, C. (2026). Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework. arXiv:2603.11768.
- Letta (2025). *Benchmarking AI Agent Memory: Is a Filesystem All You Need?* 12 August 2025. Vendor research.
- Li, X. et al. (2026). MemTX: Transactional Belief Commit for Stateful Agent Memory. arXiv:2607.23929.
- Li, Y. et al. (2026). LoCoMo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents. arXiv:2602.10715.
- Li, Z. et al. (2025). MemOS: A Memory OS for AI System. arXiv:2507.03724.
- Lin, K., Snell, C., Wang, Y., Packer, C., Wooders, S., Stoica, I., Gonzalez, J. E. (2025). Sleep-time Compute: Beyond Inference Scaling at Test-time. arXiv:2504.13171.
- Lin, Z. et al. (2026). A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle. EMNLP 2026; arXiv:2604.16548.
- Liu, N. F. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. TACL 2024.
- Luo, J. et al. (2026). From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms. ACL 2026 Findings; arXiv:2605.06716.
- Maharana, A. et al. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo). ACL 2024.
- Mariani et al. (2025). Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects. arXiv:2512.12818. Vendor research (Vectorize).
- Mem0 (2026a). *State of AI Agent Memory 2026: Benchmarks & Trends.* Blog, updated 10 October 2026. Vendor.
- Mem0 (2026b). *The State of Agent Wikis.* Article, July 2026. Vendor.
- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT (ROME). NeurIPS 2022.
- Meng, K. et al. (2023). Mass-Editing Memory in a Transformer (MEMIT). ICLR 2023.
- Modarressi, A. et al. (2025). NoLiMa: Long-Context Evaluation Beyond Literal Matching. arXiv:2502.05167.
- Munkhdalai, T., Faruqui, M., Gopal, S. (2024). Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention. arXiv:2404.07143.
- Nboxff et al. (2026). Can Agents Forget When Asked? A Benchmark for Memory Unlearning in LLM Agents (Bemula). arXiv:2610.18052.
- Ouyang, S. et al. (2025). ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. ICLR 2026; arXiv:2509.25140.
- Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST 2023.
- Pink, M. et al. (2025). Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents. arXiv:2502.06975.
- Rae, J. W. et al. (2019). Compressive Transformers for Long-Range Sequence Modelling. ICLR 2020.
- Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., Chalef, D. (2025). Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956. Vendor research.
- Ravindran, S. K. (2026). Portable Agent Memory: A Protocol for Cryptographically-Verified Memory Transfer Across Heterogeneous AI Agents. arXiv:2605.11032.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
- Sumers, T., Yao, S., Narasimhan, K., Griffiths, T. (2023). Cognitive Architectures for Language Agents (CoALA). TMLR 2024; arXiv:2309.02427.
- Sun, Y. et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620.
- Sunil, B. D. et al. (2026). Memory Poisoning Attack and Defense on Memory Based LLM-Agents. arXiv:2601.05504.
- Suzgun, M., Yuksekgonul, M., Bianchi, F., Jurafsky, D., Zou, J. (2025). Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory. arXiv:2504.07952.
- Taheri, M. et al. (2025). Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs (BEAM). arXiv:2510.27246.
- Wang, G. et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. TMLR 2024.
- Wang, Y., Chen, X. (2025). MIRIX: Multi-Agent Memory System for LLM-Based Agents. arXiv:2507.07957.
- Wang, Y. et al. (2024). MemoryLLM: Towards Self-Updatable Large Language Models. ICML 2024.
- Wang, Y. et al. (2025a). M+: Extending MemoryLLM with Scalable Long-Term Memory. ICML 2025.
- Wang, Y. et al. (2025b). Mem-α: Learning Memory Construction via Reinforcement Learning. arXiv:2509.25911.
- Wang, Z. Z., Mao, J., Fried, D., Neubig, G. (2024). Agent Workflow Memory. arXiv:2409.07429.
- Wang et al. (2026). Can Agents Truly Forget? Explore Machine Unlearning for LLM Agents. arXiv:2610.17659.
- Wooders, S. (2026). *Why memory isn't a plugin.* Letta blog, April 2026. Vendor.
- Wu, D. et al. (2024). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. ICLR 2025.
- Wu, Y., Rabe, M. N., Hutchins, D., Szegedy, C. (2022). Memorizing Transformers. ICLR 2022.
- Wu, Y. et al. (2025b). From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs. arXiv:2504.15965.
- Wu, Y. et al. (2026b). Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents. arXiv:2607.08716.
- Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. (2025). A-MEM: Agentic Memory for LLM Agents. NeurIPS 2025; arXiv:2502.12110.
- Yan, S. et al. (2025). Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning. ACL 2026; arXiv:2508.19828.
- Yang, H. et al. (2024). Memory3: Language Modeling with Explicit Memory. arXiv:2407.01178.
- Yu, H. et al. (2025). MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent. arXiv:2507.02259.
- Yu, Y. et al. (2026). Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents (AgeMem). ACL 2026; arXiv:2601.01885.
- Zep (2025). *Lies, Damn Lies, and Statistics: Is Mem0 Really SOTA in Agent Memory?* Blog, 6 May 2025. Vendor.
- Zhang, G. et al. (2025a). G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems. arXiv:2506.07398.
- Zhang, N. et al. (2026). HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents. arXiv:2601.06377.
- Zhang, Q. et al. (2025b). Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models (ACE). arXiv:2510.04618.
- Zhang, Z. et al. (2024). A Survey on the Memory Mechanism of Large Language Model based Agents. arXiv:2404.13501; later published by ACM (2025).
- Zhao, A. et al. (2023). ExpeL: LLM Agents Are Experiential Learners. AAAI 2024.
- Zhao, Y. et al. (2026). AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications. arXiv:2602.22769.
- Zhong, W., Guo, L., Gao, Q., Ye, H., Wang, Y. (2023). MemoryBank: Enhancing Large Language Models with Long-Term Memory. AAAI 2024.
- Zhou, H. et al. (2025a). Memento: Fine-tuning LLM Agents without Fine-tuning LLMs. arXiv:2508.16153.
- Zhou, Z. et al. (2025b). MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents. ICLR 2026; arXiv:2506.15841.
- Zhou et al. (2026). Agentic Unlearning: When LLM Agent Meets Machine Unlearning. arXiv:2602.17692.
- Zhu, Q., Chen, S., Yu, R., Wu, Z., Wang, B. (2026). From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents (TierMem). arXiv:2602.17913.
