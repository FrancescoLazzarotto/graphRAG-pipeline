#!/usr/bin/env python3
"""Measure an LLM topic gate on the same questions the score threshold failed on.

`calibrate_domain_gate.py` showed the dense top-1 similarity does not separate
in-domain from out-of-domain questions: the in-domain minimum is 0.7996 and the
out-of-domain maximum is 0.8314, so the ranges overlap and no single threshold
works. e5 compresses everything into a narrow band, and an SQL question lands
above a third of the gold set.

This evaluates the alternative: one short classification call to the serving
model before retrieval, at temperature 0, answering a single token. The gate is
only worth wiring if it clears both error classes on the same 50 questions:

* false refusal — an in-domain question judged out of domain. Expensive: the
  expert asks something legitimate and the demo stonewalls. Must be 0.
* false accept — an out-of-domain question judged in domain. Cheap by
  comparison: the answer still gets generated, marked as ungrounded.

Usage:
    conda run -n graphllm python scripts/domain_gate/eval_domain_gate_llm.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "domain_gate"))

from calibrate_domain_gate import IT_IN_DOMAIN, OUT_OF_DOMAIN, load_gold_questions  # noqa: E402

VLLM_URL = "http://localhost:8000/v1/chat/completions"

# The scope description is the whole gate: it must name the domain widely enough
# to cover what the corpus actually holds (circular economy, food systems,
# agri-food by-products, sustainability policy, territorial projects) without
# turning into "anything to do with food", which would let a recipe through.
GATE_SYSTEM = (
    "You classify whether a question can be answered from a document collection "
    "about the circular economy of food. The collection covers: circular economy "
    "principles and frameworks applied to food, food systems and supply chains, "
    "agri-food by-products and residues and their valorisation — including their "
    "chemical composition and their pharmaceutical, nutraceutical, cosmetic, "
    "energy and material uses — food waste, food and beverage packaging and its "
    "materials, sustainability indicators and policy, and territorial or regional "
    "food projects.\n\n"
    "Answer with exactly one word:\n"
    "IN — the question is about that domain\n"
    "OUT — the question is about something else (programming, mathematics, "
    "geography, entertainment, general knowledge, or any other field)\n\n"
    "A question is IN whenever its subject is a food, a crop, a food-industry "
    "residue or by-product, or a food supply chain — whatever is being asked "
    "about it. Asking what compounds rice bran contains, or what a food package "
    "is made of, is a question about the domain, not about pharmacology or "
    "materials science.\n\n"
    "A question is also IN when it asks about the theoretical vocabulary of the "
    "Circular Economy for Food framework itself, even when it never mentions "
    "food: the three C's (Capital, Cyclicality, Co-evolution), metabolisation "
    "and its implementation cycles, extension, cascading, ecodesign, industrial "
    "symbiosis, and the relations between these concepts.\n\n"
    "Answer IN whenever the question plausibly belongs to the domain, even if "
    "you doubt the collection holds the specific detail asked for: the retrieval "
    "step decides that, not you. Answer with the single word only."
)


def classify(question: str, model_id: str) -> tuple[str, float]:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": GATE_SYSTEM},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "max_tokens": 4,
    }
    request = urllib.request.Request(
        VLLM_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - start
    verdict = body["choices"][0]["message"]["content"].strip().upper()
    return ("OUT" if verdict.startswith("OUT") else "IN"), elapsed


def served_model() -> str:
    with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5) as response:
        return json.load(response)["data"][0]["id"]


def run(label: str, questions: list[str], expected: str, model_id: str) -> tuple[int, list[str]]:
    errors: list[str] = []
    latencies: list[float] = []
    for question in questions:
        verdict, elapsed = classify(question, model_id)
        latencies.append(elapsed)
        if verdict != expected:
            errors.append(f"   {verdict} (atteso {expected})  {question}")
    mean_latency = sum(latencies) / max(1, len(latencies))
    ok = len(questions) - len(errors)
    print(f"\n## {label}: {ok}/{len(questions)} corrette  (latenza media {mean_latency:.2f}s)")
    for line in errors:
        print(line)
    return len(errors), errors


def main() -> int:
    model_id = served_model()
    print(f"modello: {model_id}")

    gold = load_gold_questions(ROOT / "evaluation" / "gold" / "gold_circular_v1.json")

    false_refusals_en, _ = run("IN DOMINIO — gold EN (30)", gold, "IN", model_id)
    false_refusals_it, _ = run("IN DOMINIO — probe IT (10)", IT_IN_DOMAIN, "IN", model_id)
    false_accepts, _ = run("FUORI DOMINIO — probe (10)", OUT_OF_DOMAIN, "OUT", model_id)

    false_refusals = false_refusals_en + false_refusals_it
    print("\n" + "=" * 78)
    print(f"rifiuti falsi (in dominio giudicato OUT):  {false_refusals} / 40")
    print(f"accettazioni false (fuori dominio → IN):   {false_accepts} / 10")
    if false_refusals == 0 and false_accepts == 0:
        print("\nSEPARAZIONE PERFETTA sui 50 casi. Il gate è cablabile.")
    elif false_refusals == 0:
        print("\nNESSUN RIFIUTO FALSO. Il gate è cablabile: le accettazioni false")
        print("cadono nel percorso 'rispondi ma marca', che è il comportamento")
        print("previsto per il retrieval debole.")
    else:
        print("\nRIFIUTI FALSI PRESENTI: il gate bloccherebbe domande legittime.")
        print("Non cablare finché il prompt di scope non li azzera.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
