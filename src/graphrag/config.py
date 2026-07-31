from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

logger = logging.getLogger("graphrag")


class OUTPUT_TONE(enum.Enum):
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"
    FORMAL = "formal"


class OUTPUT_COMPLEXITY(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class AgentConfig:
    query: str | None = None
    entity: str | None = None
    entity_a: str | None = None
    entity_b: str | None = None
    hops: int = 1
    max_depth: int = 6
    nodes_limit: int = 10
    triples_limit: int = 20
    neighbors_limit: int = 25
    subgraph_limit: int = 200
    labels: tuple[str, ...] = ()
    relationship_types: tuple[str, ...] = ()
    include_nodes: bool = True
    include_triples: bool = True
    include_neighbors: bool = True
    include_subgraph: bool = True
    include_shortest_path: bool = True
    answer_prompt: str = ""
    rewrite_prompt: str = ""
    kg_reasoning_prompt: str = ""
    decomposition_prompt: str = ""
    reflection_prompt: str = ""
    adaptive_router_prompt: str = ""
    llm_warmup: bool = False
    enable_decomposition_step: bool = False
    enable_adaptive_routing_step: bool = False
    enable_cache: bool = True
    cache_maxsize: int = 128
    recursion_limit: int = 50
    # 6000 fits comfortably in Qwen2.5-32B's 32k window with prompt + answer;
    # 1000 truncated most multi-channel retrievals (head/tail compression cut
    # the mid-context evidence and inflated "insufficient context" answers).
    max_content_tokens: int = 6000
    token_estimator_ratio: float = 0.25  # tokens-per-char (~4 chars/token)
    tone: OUTPUT_TONE = OUTPUT_TONE.TECHNICAL
    complexity: OUTPUT_COMPLEXITY = OUTPUT_COMPLEXITY.MEDIUM
    target_audience: str = "domain_expert"
    use_structured_response: bool = False
    rank_triples: bool = True
    # When decomposition produces multiple retrieval queries, merged results
    # preserve arrival order by default; enable to re-rank them globally by
    # score. Off by default to keep existing baselines unchanged.
    rerank_merged_results: bool = False
    # The answer prompt asks for a 'Limits and confidence' section only when
    # context is sparse; enable to request it on every answer (demo UX). Off
    # by default to keep existing baselines unchanged.
    always_include_limits: bool = False
    # WP1 (docs/demo_quality_plan_2026-07.md): retrieved evidence is rendered as
    # numbered blocks carrying document and page, the answer prompt asks for
    # [S1]/[T1] tags on specific claims, and a post-generation gate checks every
    # tag against the index. Off by default so gold runs and experiment
    # baselines keep the previous prompt and context format.
    cite_evidence: bool = False
    # What to do with a reference tag the model invented: "mark" flags it in
    # place, "strip" deletes it. Marking is the default because deleting leaves
    # an unsupported claim looking like ordinary prose.
    citation_policy: str = "mark"
    evidence_max_text_items: int = 12
    evidence_max_triple_items: int = 30
    # How verified references are shown to the reader. "id" keeps [S1]/[T3],
    # which is what experiment artifacts and the citation metrics parse;
    # "label" rewrites them as "[SEeD for Change, p. 3]" after the gate has run,
    # because a reader cannot check "S3" against anything.
    citation_display: str = "id"
    # WP5 (docs/demo_quality_plan_2026-07.md): the language detected on the
    # question becomes an explicit constraint in the answer prompt, written in
    # the target language, and a single retry fires when the answer comes back
    # in the other language. Off by default: it changes the rendered prompt, so
    # gold runs and experiment baselines opt in explicitly.
    enforce_language: bool = False
    # Triples carry no per-edge confidence yet (see KG-side item B8), so the
    # confidence weight is 0.0 and lexical/mention absorb it. Keeping the field
    # lets a future confidence signal be re-enabled without code changes.
    ranker_weight_lexical: float = 0.70
    ranker_weight_mention: float = 0.30
    ranker_weight_confidence: float = 0.0
    ranker_system_link_penalty: float = 0.5
    adaptive_hops: bool = True
    min_subgraph_triples: int = 10
    max_hops: int = 4
    include_triple_metadata: bool = True
    # WP3 (docs/demo_quality_plan_2026-07.md §5): definitional questions get the
    # chunk carrying the verbatim definition ranked first and an answer that
    # quotes before it paraphrases. Off by default so gold runs and experiment
    # baselines keep the previous ranking and prompt.
    prefer_verbatim_definitions: bool = False
    # Weight of the definitional signal when reordering already-retrieved
    # chunks. It reorders, it never fetches: the worst case is the order the
    # retriever would have produced anyway.
    definition_boost_weight: float = 1.0
    # Checks that every «...» passage occurs in the retrieved text, and drops
    # the guillemets when it does not. Independent of the WP3 prompt because a
    # model can quote unprompted, and a fabricated quote carrying a valid [S2]
    # is the one failure the citation gate cannot see.
    verify_quoted_passages: bool = True
    use_text_retriever: bool = False
    text_retriever_top_k: int = 5
    # WP4 (docs/demo_quality_plan_2026-07.md §6): source diversification.
    # MMR trades a little query similarity for coverage; the per-document cap
    # is the part that actually stops one PDF from filling the context, since
    # two pages of the same document can be far apart in embedding space and
    # still both be selected. Both off by default.
    text_retriever_mmr: bool = False
    text_retriever_mmr_lambda: float = 0.7
    # 0 disables the cap. Enumerative questions get twice this budget: their
    # answer is usually one list on contiguous pages of a single document, and
    # capping that document truncates the list.
    text_retriever_max_per_doc: int = 0
    # Candidate pool the cap and the definitional boost choose from. 0 means
    # ``4 * text_retriever_top_k``.
    text_retriever_fetch_k: int = 0
    text_retriever_backend: str = "tfidf"  # "tfidf" | "dense"
    dense_embedding_model: str = "intfloat/multilingual-e5-base"
    dense_query_prefix: str = "query: "
    dense_passage_prefix: str = "passage: "
    dense_normalize: bool = True
    dense_device: str = "auto"  # "auto" | "cpu" | "cuda"
    vector_index_dir: str = "artifacts/vector_index"
    # P1 (exp_results/KG_VS_RETRIEVAL.md): the full-text query is a flat OR of
    # every term the question yields, all weighted alike, so a generic token
    # outvotes the specific phrase by matching more nodes. "What are the three
    # C's of the Circular Economy for Food framework?" retrieved 41 nodes, all
    # "…framework" variants, and none of the three C's — which are in the graph.
    # Enabling this drops query tokens whose node-name document frequency
    # exceeds the ratio below and boosts the surviving terms by rarity.
    # Off by default so existing baselines keep the previous term selection.
    lexical_specificity: bool = False
    # A token in more than this share of node names carries no discriminative
    # power. 0.01 of 14 520 nodes ≈ 145 names; "framework" is well above it.
    lexical_df_max_ratio: float = 0.01
    # Multi-word candidates are the reliable anchors, single tokens the risky
    # ones: weight the phrase query so it survives alongside common tokens.
    lexical_phrase_boost: float = 4.0
    # Ceiling on the rarity boost given to a surviving single token, so one
    # hapax cannot monopolise the result set.
    lexical_max_token_boost: float = 3.0
    lexical_df_cache_path: str = "artifacts/kg_token_df.json"
    # P1, second half: the anchor for the neighbour, subgraph and shortest-path
    # channels was the first *search term* — a raw word from the question, like
    # "valuable" or "implementation", which matches no node. Enabling this ranks
    # node names the index actually returned ahead of query words. Off by
    # default because it changes which subgraph those three channels expand.
    seed_from_retrieved: bool = False
    # P0 (exp_results/KG_VS_RETRIEVAL.md): retrieval is purely lexical, but the
    # graph is largely Italian and the questions are English — 44 % of the gold
    # entities exist in the graph *only* under an Italian surface form, so no
    # lexical query can reach them. A multilingual encoder puts both in one
    # space: "the three C's of the Circular Economy for Food" retrieves the node
    # "3 C dell'Economia Circolare per l'Alimentazione". The vector channel is
    # added to the lexical one, never replaces it — exact surface matches are
    # still the most precise signal available.
    # Requires scripts/kg_vector_index.py and a running embedding endpoint; off
    # by default, and degrades to lexical-only if either is missing.
    vector_retrieval: bool = False
    vector_index: str = "node_embedding"
    # Nodes pulled from the vector channel per query. Kept near nodes_limit so
    # the two channels contribute comparably instead of one drowning the other.
    vector_nodes_limit: int = 10
    vector_triples_limit: int = 10
    # Nearest nodes expanded into triples. Small on purpose: the graph is 72 %
    # leaves, so expanding many weak seeds adds edges, not answers.
    vector_seed_limit: int = 5
    # Cosine floor. e5 scores short names in a narrow high band, so a hard
    # threshold mostly removes the tail; ranking does the real work.
    vector_min_score: float = 0.0
    # P2 (exp_results/KG_VS_RETRIEVAL.md): the answer prompt's "use ONLY the
    # provided context" suppresses the model's own knowledge even when
    # retrieval missed, which the campaign measured as a net loss — graph
    # context destroyed 12 answers Qwen2.5-32B produced correctly with no
    # context at all. Enabling this authorises a fallback, but only marked as
    # such, so groundedness stays measurable. Off by default: it changes the
    # rendered prompt, so baselines opt in explicitly.
    allow_parametric_fallback: bool = False
    # Predicates that carry no answerable content. RELATED_TO alone is 20 % of
    # the graph's edges and 19 % of what retrieval returns; the bibliographic
    # pair is another 17 %. Dropping them frees context budget for triples that
    # can actually support a claim. Empty tuple keeps every predicate.
    drop_predicates: tuple[str, ...] = ()
    # Check that the anchor matches a node before expanding neighbours, the
    # subgraph and the shortest path. Those three channels each scan the graph
    # when the anchor matches nothing, which on the thesis gold set cost up to
    # 34 s on a single question and returned no evidence at all. On by default:
    # it changes latency, never the evidence, since a seed that matches no node
    # could not have produced any.
    verify_anchor_exists: bool = True
    # How many anchors the subgraph channel expands from. Anchoring on retrieved
    # nodes made every seed accurate, which also made the 2-hop neighbourhood
    # narrower: subgraph_2hop was the only strategy to lose recall. Expanding
    # from the top few anchors, each with a share of the triple budget, restores
    # breadth without reverting to question-word seeds. 1 keeps the old shape.
    subgraph_seed_count: int = 1
    # The retrieval fixes raised answer recall by +0.036 and dropped precision by
    # -0.033: a richer context yields a more discursive answer that names more
    # entities, and entities belonging to other questions count against it. This
    # asks for the same grounding with a narrower scope — answer what was asked
    # and leave out related material the evidence happens to carry. Off by
    # default; it changes the rendered prompt.
    focused_answer: bool = False

    def __post_init__(self) -> None:
        if self.rank_triples:
            weight_sum = (
                self.ranker_weight_lexical
                + self.ranker_weight_mention
                + self.ranker_weight_confidence
            )
            if abs(weight_sum - 1.0) > 0.01:
                logger.warning(
                    "Ranker weights sum to %.3f instead of 1.0 "
                    "(lexical=%.2f mention=%.2f confidence=%.2f): triple scores "
                    "will not be comparable across configurations",
                    weight_sum,
                    self.ranker_weight_lexical,
                    self.ranker_weight_mention,
                    self.ranker_weight_confidence,
                )


@dataclass(slots=True)
class KGConfig:
    url: str
    username: str
    password: str
    database: str | None = None
    node_name_properties: tuple[str, ...] = (
        "name",
        "title",
        "label",
        "id",
        "uuid",
        "entity",
    )
    default_limit: int = 50


def build_kg_config_from_env(
    url_env: str = "NEO4J_URL",
    username_env: str = "NEO4J_USERNAME",
    password_env: str = "NEO4J_PASSWORD",
    database_env: str = "NEO4J_DATABASE",
) -> KGConfig:
    url = os.getenv(url_env)
    username = os.getenv(username_env)
    password = os.getenv(password_env)
    database = os.getenv(database_env)

    missing = [
        key
        for key, value in (
            (url_env, url),
            (username_env, username),
            (password_env, password),
        )
        if not value
    ]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_csv}")

    return KGConfig(
        url=url,
        username=username,
        password=password,
        database=database,
    )
