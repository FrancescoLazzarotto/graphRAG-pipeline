from __future__ import annotations

import re
import unicodedata
from typing import Any

from evalkit.models import EvalRow


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ─── Deterministic text metrics ─────────────────────────────────────────────

def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalized strings are identical, else 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def partial_match(prediction: str, reference: str) -> float:
    """Ratio of reference tokens found anywhere in prediction."""
    pred_norm = _normalize(prediction)
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0
    hits = sum(1 for token in ref_tokens if token in pred_norm)
    return hits / len(ref_tokens)


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set: dict[str, int] = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1

    ref_set: dict[str, int] = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1

    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L based on longest common subsequence at token level."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Corpus-style BLEU-N (1..max_n average) with brevity penalty.

    Uses sacrebleu when available, falls back to a simple sentence-BLEU.
    """
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU  # type: ignore

        metric = SacreBLEU(effective_order=True)
        result = metric.sentence_score(prediction, [reference])
        return result.score / 100.0
    except ImportError:
        pass

    return _simple_bleu(prediction, reference, max_n)


def _ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _simple_bleu(prediction: str, reference: str, max_n: int) -> float:
    import math

    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0

    precisions = []
    for n in range(1, max_n + 1):
        if len(pred_tokens) < n:
            precisions.append(0.0)
            continue
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        clipped = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append(clipped / total if total else 0.0)

    if all(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions if p > 0) / max_n
    return bp * math.exp(log_avg)


def bertscore_f1(prediction: str, reference: str, model: str = "distilbert-base-uncased") -> float | None:
    """BERTScore F1 (returns None if bert-score is not installed)."""
    try:
        from bert_score import score as _bs  # type: ignore

        P, R, F1 = _bs([prediction], [reference], model_type=model, verbose=False)
        return float(F1[0])
    except ImportError:
        return None


# ─── Variant-aware scoring ───────────────────────────────────────────────────

def best_variant_score(
    metric_fn: Any,
    prediction: str,
    ground_truth: str,
    answer_variants: list[str],
) -> float:
    """Return the best score across ground_truth and answer_variants."""
    references = [ground_truth] + [v for v in answer_variants if v.strip()]
    scores = [metric_fn(prediction, ref) for ref in references if ref.strip()]
    return max(scores) if scores else 0.0


# ─── Row-level computation ───────────────────────────────────────────────────

def compute_text_row(row: EvalRow, bertscore: bool = False) -> dict[str, Any]:
    """Compute text similarity metrics for a single EvalRow.

    Returns None for all metrics when skip_reason is set or ground_truth is empty.
    """
    base = {
        "run_dir": row.run_dir,
        "model_id": row.model_id,
        "framework": row.framework,
        "strategy": row.strategy,
        "question": row.question,
        "question_type": row.question_type,
        "skip_reason": row.skip_reason,
        "exact_match": None,
        "partial_match": None,
        "token_f1": None,
        "rouge_l": None,
        "bleu": None,
        "bertscore_f1": None,
    }

    if row.skip_reason or not row.ground_truth.strip():
        return base

    pred = row.answer
    gt = row.ground_truth
    variants = row.answer_variants

    base["exact_match"] = best_variant_score(exact_match, pred, gt, variants)
    base["partial_match"] = best_variant_score(partial_match, pred, gt, variants)
    base["token_f1"] = best_variant_score(token_f1, pred, gt, variants)
    base["rouge_l"] = best_variant_score(rouge_l, pred, gt, variants)
    base["bleu"] = best_variant_score(bleu, pred, gt, variants)

    if bertscore:
        bs = best_variant_score(bertscore_f1, pred, gt, variants)
        base["bertscore_f1"] = bs if bs is not None else None

    return base


TEXT_METRICS = ["exact_match", "partial_match", "token_f1", "rouge_l", "bleu", "bertscore_f1"]
