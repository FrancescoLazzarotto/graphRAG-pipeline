from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig, DEFAULT_MODEL_ID, build_kg_config_from_env
from graphrag.experiments import ExperimentRunner, Question
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.strategies import apply_strategy
from graphrag.text_rag.factory import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_CHUNK_CHARS,
    make_text_pipeline,
)
from graphrag.text_rag.pipeline import StandardTextRAGPipeline

logger = logging.getLogger("graphrag.cli")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GraphRAG demo pipeline")
    parser.add_argument(
        "--question",
        default="Quali sono gli obiettivi della strategia Farm to Fork?",
    )
    parser.add_argument(
        "--entity",
        default="",
        help="Optional entity seed for graph traversal (leave empty for auto-seeding)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--llm", action="store_true", help="Enable LLM generation")
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use a vLLM OpenAI-compatible endpoint instead of local Hugging Face loading",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible API",
    )
    parser.add_argument(
        "--llm-warmup", action="store_true", help="Preload model at startup"
    )
    parser.add_argument(
        "--enable-decomposition-step",
        action="store_true",
        help="Enable LLM decomposition step before retrieval",
    )
    parser.add_argument(
        "--enable-adaptive-routing-step",
        action="store_true",
        help="Enable LLM adaptive routing step before retrieval",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per response",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=6000,
        help="Maximum tokens for compressed context before generation",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="Maximum LangGraph recursion steps before aborting",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.92,
        help="Fraction of each GPU memory reserved for model placement (0,1]",
    )
    parser.add_argument(
        "--allow-large-model-fp16-fallback",
        action="store_true",
        help="Allow fp16 fallback for large models when 4-bit quantized loading fails",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run batch experiments and persist outputs",
    )
    parser.add_argument(
        "--questions-file",
        help="Questions to run. .txt: one question per line (optionally "
             "'Q01<TAB>question' to carry the gold id); .json: the gold's "
             "{'queries': [{'query_id','query'}]} shape; .jsonl: one such object "
             "per line; .csv: query_id + query columns. Declaring ids lets the "
             "evaluator join results to the gold by query_id instead of by text.",
    )
    parser.add_argument(
        "--strategies", default="default", help="Comma-separated strategy presets"
    )
    parser.add_argument("--runs-per-strategy", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument(
        "--text-docs-dir",
        default="",
        help="Directory of documents (PDF/txt/md) to index for text_only standard RAG. "
             "If omitted, auto-discovers from the latest KG pipeline stage0 artifacts.",
    )
    parser.add_argument(
        "--text-retriever-backend",
        default="tfidf",
        choices=("tfidf", "dense"),
        help="Retrieval backend for standard RAG: 'tfidf' (lexical, default) or 'dense' (cosine/FAISS).",
    )
    parser.add_argument(
        "--dense-embedding-model",
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model ID for dense retrieval (ignored for tfidf).",
    )
    parser.add_argument(
        "--vector-index-dir",
        default="artifacts/vector_index",
        help="Directory for persisted FAISS index cache (ignored for tfidf).",
    )
    return parser


def _build_llm_manager(args: argparse.Namespace, warmup: bool) -> LLMManager | None:
    if not args.llm:
        return None

    return LLMManager(
        model_id=args.model_id,
        warmup=warmup,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_fraction=args.gpu_memory_fraction,
        allow_large_model_fp16_fallback=args.allow_large_model_fp16_fallback,
        use_vllm=args.vllm,
        vllm_base_url=args.vllm_base_url,
    )


def _build_base_config(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        query=args.question,
        entity=args.entity,
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        include_subgraph=True,
        include_shortest_path=True,
        llm_warmup=args.llm_warmup,
        enable_decomposition_step=args.enable_decomposition_step,
        enable_adaptive_routing_step=args.enable_adaptive_routing_step,
        recursion_limit=args.recursion_limit,
        max_content_tokens=args.max_context_tokens,
    )


def _build_text_pipeline(args: argparse.Namespace) -> StandardTextRAGPipeline | None:
    logger = logging.getLogger("graphrag.cli")
    backend = getattr(args, "text_retriever_backend", "tfidf")
    pipeline = make_text_pipeline(
        backend=backend,
        embedding_model=getattr(args, "dense_embedding_model", "intfloat/multilingual-e5-base"),
        vector_index_dir=getattr(args, "vector_index_dir", "artifacts/vector_index"),
    )

    docs_dir = (args.text_docs_dir or "").strip()
    if docs_dir:
        target = Path(docs_dir)
        if not target.exists():
            logger.warning("--text-docs-dir %s not found; text retrieval disabled", docs_dir)
            return None
        n = pipeline.index_directory(target)
        logger.info("Text pipeline: indexed %d chunks from %s", n, docs_dir)
        return pipeline

    # Auto-discover from the latest KG stage0 artifacts.
    kg_artifacts = Path("kg_pipeline/artifacts")
    if kg_artifacts.exists():
        run_dirs = sorted(
            [p for p in kg_artifacts.iterdir() if p.is_dir() and p.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs:
            stage0 = run_dir / "stage0_documents.json"
            if not stage0.exists():
                continue
            try:
                import json as _json
                from graphrag.text_rag.manager import TextChunk
                docs = _json.loads(stage0.read_text(encoding="utf-8"))
                if not isinstance(docs, list):
                    docs = []
                chunks: list[TextChunk] = []
                for doc_idx, doc in enumerate(docs, start=1):
                    text = str(doc.get("markdown_text", "") or "").strip()
                    filename = str(doc.get("filename", f"doc_{doc_idx}"))
                    if not text:
                        continue
                    step = DEFAULT_CHUNK_SIZE - DEFAULT_CHUNK_OVERLAP
                    for c_idx, start in enumerate(range(0, len(text), step), start=1):
                        fragment = text[start : start + DEFAULT_CHUNK_SIZE].strip()
                        if len(fragment) >= DEFAULT_MIN_CHUNK_CHARS:
                            chunks.append(TextChunk(
                                chunk_id=f"d{doc_idx:04d}-c{c_idx:04d}",
                                content=fragment,
                                source=filename,
                            ))
                if chunks:
                    pipeline.retriever.add_chunks(chunks)
                    logger.info(
                        "Text pipeline: indexed %d chunks from %s (stage0)",
                        len(chunks), run_dir.name,
                    )
                    return pipeline
            except Exception as exc:
                logger.warning("Failed to load stage0 from %s: %s", run_dir, exc)
                continue

    logger.warning("No text documents found; text_only strategy will have empty context")
    return None


def _question_from_obj(obj: dict, where: str) -> Question:
    """Build a Question from a JSON/JSONL/CSV entry.

    Args:
        obj: Mapping with a question field and an optional id field.
        where: Human-readable location, for error messages.

    Returns:
        The parsed Question.

    Raises:
        ValueError: If no question text field is present.
    """
    text = str(
        obj.get("query") or obj.get("question") or obj.get("text") or ""
    ).strip()
    if not text:
        raise ValueError(f"{where}: entry has no 'query'/'question'/'text' field")
    query_id = str(obj.get("query_id") or obj.get("id") or "").strip()
    return Question(text=text, query_id=query_id)


def _questions_from_text(path: Path) -> list[Question]:
    """Parse the plain-text format: one question per line.

    A line may optionally carry its gold id as ``Q01<TAB>question text``. Lines
    without a TAB keep their legacy meaning — the whole line is the question and
    the run emits no id for it.
    """
    questions: list[Question] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        query_id = ""
        text = line
        if "\t" in line:
            head, _, tail = line.partition("\t")
            if head.strip() and tail.strip():
                query_id, text = head.strip(), tail.strip()
            else:
                text = line.replace("\t", " ").strip()
        questions.append(Question(text=text, query_id=query_id))
    return questions


def _questions_from_jsonl(path: Path) -> list[Question]:
    """Parse a JSONL questions file: one {query_id, query} object per line."""
    questions: list[Question] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        questions.append(_question_from_obj(obj, f"{path}:{lineno}"))
    return questions


def _questions_from_json(path: Path) -> list[Question]:
    """Parse a JSON questions file.

    Accepts the gold's own shape (``{"queries": [{"query_id", "query"}, ...]}``),
    so a gold file can be handed straight to --questions-file and the run is
    guaranteed to emit ids that join to it. A bare list of objects or of plain
    strings also works.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if isinstance(payload, dict):
        raw_entries = payload.get("queries")
        if raw_entries is None:
            raise ValueError(f"{path}: JSON object has no 'queries' array")
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ValueError(f"{path}: expected a JSON object with 'queries' or a list")

    if not isinstance(raw_entries, list):
        raise ValueError(f"{path}: 'queries' must be a list")

    questions: list[Question] = []
    for idx, entry in enumerate(raw_entries):
        if isinstance(entry, str):
            if entry.strip():
                questions.append(Question(text=entry.strip()))
        elif isinstance(entry, dict):
            questions.append(_question_from_obj(entry, f"{path}[{idx}]"))
        else:
            raise ValueError(f"{path}[{idx}]: expected an object or a string")
    return questions


def _questions_from_csv(path: Path) -> list[Question]:
    """Parse a CSV questions file with a query/question column and optional query_id."""
    questions: list[Question] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for lineno, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            questions.append(_question_from_obj(dict(row), f"{path}:{lineno}"))
    return questions


def _load_questions(args: argparse.Namespace) -> list[Question]:
    """Load the questions to run, with their gold ids when the file declares them.

    Supported --questions-file formats, picked by suffix:
      * ``.txt`` / anything else: one question per line (legacy), optionally
        ``Q01<TAB>question text``;
      * ``.json``: the gold's ``{"queries": [...]}`` shape, or a bare list;
      * ``.jsonl``: one ``{"query_id", "query"}`` object per line;
      * ``.csv``: a ``query_id`` column plus ``query`` or ``question``.

    Returns:
        The questions in file order.

    Raises:
        FileNotFoundError: If the questions file does not exist.
        ValueError: If the file is empty, malformed, or repeats a query_id.
    """
    logger = logging.getLogger("graphrag.cli")

    if not args.questions_file:
        return [Question(text=args.question)]

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    suffix = questions_path.suffix.lower()
    if suffix == ".json":
        questions = _questions_from_json(questions_path)
    elif suffix == ".jsonl":
        questions = _questions_from_jsonl(questions_path)
    elif suffix == ".csv":
        questions = _questions_from_csv(questions_path)
    else:
        questions = _questions_from_text(questions_path)

    if not questions:
        raise ValueError(f"Questions file is empty: {questions_path}")

    ids = [q.query_id for q in questions if q.query_id]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        # Duplicated ids would make the evaluator's join ambiguous, and it joins
        # on exactly this field.
        raise ValueError(
            f"{questions_path}: duplicate query_id(s) {duplicates}"
        )

    if not ids:
        logger.warning(
            "Questions file %s declares no query_id: results.jsonl will carry an "
            "empty query_id and the evaluator must fall back to joining on "
            "question TEXT. Use a gold .json/.jsonl/.csv, or 'Q01<TAB>question' "
            "lines, to join by id.",
            questions_path,
        )
    elif len(ids) < len(questions):
        logger.warning(
            "Questions file %s declares query_id for only %d/%d questions; the "
            "rest will fall back to a text join at evaluation time.",
            questions_path,
            len(ids),
            len(questions),
        )
    else:
        logger.info(
            "Loaded %d questions with query_id from %s", len(questions), questions_path
        )
    return questions


def _run_experiments(
    args: argparse.Namespace, kg_manager: KnowledgeGraphManager
) -> None:
    if args.runs_per_strategy < 1:
        raise ValueError("--runs-per-strategy must be >= 1")

    questions = _load_questions(args)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not strategies:
        raise ValueError("--strategies must include at least one strategy")

    base_config = _build_base_config(args)
    llm_manager = _build_llm_manager(args=args, warmup=args.llm_warmup)
    runner = ExperimentRunner(questions=questions)

    # Build the text pipeline if ANY selected strategy resolves to a config that
    # uses raw-text retrieval (text_only, hybrid, ...). Deriving this from the
    # resolved config instead of a hardcoded name list keeps new text-using
    # strategies working automatically.
    needs_text = any(
        apply_strategy(base_config, s).use_text_retriever for s in strategies
    )
    text_pipeline = _build_text_pipeline(args) if needs_text else None

    strategy_configs: dict[str, dict] = {}
    for strategy in strategies:
        for run_index in range(1, args.runs_per_strategy + 1):
            config = apply_strategy(base_config, strategy)
            if strategy not in strategy_configs:
                strategy_configs[strategy] = dataclasses.asdict(config)
            retriever = KGRetriever(
                kg_store=kg_manager,
                config=config,
                text_pipeline=text_pipeline if config.use_text_retriever else None,
            )
            agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
            runner.run_agent(
                agent=agent,
                label=strategy,
                run_metadata={
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "enable_decomposition_step": args.enable_decomposition_step,
                    "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                },
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.experiment_tag.strip() or "batch"
    output_dir = Path(args.output_dir) / f"{timestamp}_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"
    config_json_path = output_dir / "config.json"

    config_json_path.write_text(
        json.dumps(
            {
                "cli_args": {k: v for k, v in vars(args).items()},
                "strategy_configs": strategy_configs,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    runner.export_jsonl(str(jsonl_path))
    runner.export_csv(str(csv_path))
    summary_text = runner.summary()
    summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "tag": tag,
                "questions_count": len(questions),
                # Lets a reader tell at a glance whether this run can be joined
                # to the gold by id or only by question text.
                "questions_with_query_id": sum(1 for q in questions if q.query_id),
                "strategies": strategies,
                "runs_per_strategy": args.runs_per_strategy,
                "llm": {
                    "enabled": args.llm,
                    "model_id": args.model_id if args.llm else "none",
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                },
                "agent_pipeline": {
                    "enable_decomposition_step": args.enable_decomposition_step,
                    "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                },
                "stats": runner.summary_stats(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("Experiment completed. Output directory: %s", output_dir)
    for path in (jsonl_path, csv_path, summary_txt_path, summary_json_path, config_json_path):
        logger.info("Output file: %s", path)
    logger.info("Summary:\n%s", summary_text)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load local .env if present while preserving variables already set by system/scheduler.
    load_dotenv(override=False)

    if args.llm_warmup and not args.llm:
        parser.error("--llm-warmup requires --llm")
    if args.vllm and not args.llm:
        parser.error("--vllm requires --llm")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be >= 1")
    if args.recursion_limit < 1:
        parser.error("--recursion-limit must be >= 1")
    if args.gpu_memory_fraction <= 0 or args.gpu_memory_fraction > 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    kg_config = build_kg_config_from_env()
    kg_manager = KnowledgeGraphManager(kg_config)

    logger.info("Graph Schema: %s", kg_manager.refresh_schema())

    if args.experiment:
        _run_experiments(args=args, kg_manager=kg_manager)
        return

    config = _build_base_config(args)

    text_pipeline = _build_text_pipeline(args) if config.use_text_retriever else None
    retriever = KGRetriever(kg_store=kg_manager, config=config, text_pipeline=text_pipeline)
    llm_manager = _build_llm_manager(args=args, warmup=False)

    agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
    result = agent.invoke(args.question)

    print("\nAgent answer:")
    print(result.get("answer", ""))
    print("\nLatency (ms):", f"{result.get('latency_ms', 0.0):.2f}")


if __name__ == "__main__":
    main()
