"""Structured evaluation of the conversational recommendation layer.

Calls api.chat.chat() directly (not via HTTP) using a fake app_state.
Tests 25 scenarios across 6 categories: simple search, multi-turn,
budget-constrained, cross-merchant, store/location, and edge cases.

Multi-turn scenarios properly simulate the conversation by making two
sequential API calls, with the first response injected as the assistant
turn before the second user message.

Vague queries (where asking for clarification is correct behavior) are
flagged with clarification_acceptable=True. Success = products returned
OR clarification asked (for ambiguous inputs).

Requires OPENAI_API_KEY in environment (loaded from .env).

Results saved to data/chat_eval_results.json.

Usage:
    python -m scripts.evaluate_chat
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import open_clip

from pipelines.embed import CLIP_MODEL, CLIP_PRETRAINED, DB_PATH, pick_device
from pipelines.search import load_index

OUT_PATH = ROOT / "data" / "chat_eval_results.json"

# ── Scenario definitions ──────────────────────────────────────────────────────
# clarification_acceptable=True: if the LLM asks for preferences instead of
# immediately returning products, that counts as "appropriate behavior".
# multi_turn=True: the messages list contains turn_1 and turn_2 entries;
# the runner makes two sequential chat calls (simulating real conversation).

SCENARIOS = [
    # ── A. Simple / specific product search (8 scenarios) ────────────────
    # Specific queries → products expected immediately; no clarification needed
    {
        "id": 1, "category": "simple_search",
        "messages": [{"role": "user", "content": "show me dry cat food"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    {
        "id": 2, "category": "simple_search",
        "messages": [{"role": "user", "content": "dog food for puppies"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    {
        "id": 3, "category": "simple_search",
        "messages": [{"role": "user", "content": "cat litter"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    {
        "id": 4, "category": "simple_search",
        "messages": [{"role": "user", "content": "dog leash and collar"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    # Vague fashion queries → clarification is valid; but products also valid
    {
        "id": 5, "category": "simple_search",
        "messages": [{"role": "user", "content": "I'm looking for a summer dress"}],
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": True,
    },
    {
        "id": 6, "category": "simple_search",
        "messages": [{"role": "user", "content": "find me a leather bag"}],
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": True,
    },
    {
        "id": 7, "category": "simple_search",
        "messages": [{"role": "user", "content": "I want sandals or heels"}],
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": True,
    },
    {
        "id": 8, "category": "simple_search",
        "messages": [{"role": "user", "content": "show me dog toys"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },

    # ── B. Multi-turn refinement (5 scenarios) ───────────────────────────
    # Two actual turns: turn_1 first, assistant responds, then turn_2.
    # Success evaluated on the FINAL (second-turn) response.
    {
        "id": 9, "category": "multi_turn",
        "turn_1": "I'm looking for shoes",
        "turn_2": "I prefer heels under 200 euros",
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": False,
    },
    {
        "id": 10, "category": "multi_turn",
        "turn_1": "something for my cat",
        "turn_2": "food, specifically dry cat food",
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    {
        "id": 11, "category": "multi_turn",
        "turn_1": "I want a dress",
        "turn_2": "something casual for summer, not too formal",
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": False,
    },
    {
        "id": 12, "category": "multi_turn",
        "turn_1": "dog accessories",
        "turn_2": "I need a harness for a small dog",
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    {
        "id": 13, "category": "multi_turn",
        "turn_1": "I need a gift for a friend",
        "turn_2": "she has a cat and loves fashion",
        "criteria": {"products_returned": True},
        "clarification_acceptable": False,
    },

    # ── C. Budget-constrained search (4 scenarios) ───────────────────────
    {
        "id": 14, "category": "budget_constrained",
        "messages": [{"role": "user", "content": "show me dresses under 100 euros"}],
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": False,
    },
    {
        "id": 15, "category": "budget_constrained",
        "messages": [{"role": "user", "content": "cat food under 30 euros"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    {
        "id": 16, "category": "budget_constrained",
        "messages": [{"role": "user", "content": "dog snacks under 10 euros"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": False,
    },
    {
        "id": 17, "category": "budget_constrained",
        "messages": [{"role": "user", "content": "sweaters or knitwear under 150 euros"}],
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": False,
    },

    # ── D. Cross-merchant handling (3 scenarios) ─────────────────────────
    {
        "id": 18, "category": "cross_merchant",
        "messages": [{"role": "user", "content": "I want to buy a dress for myself and dry cat food for my cat"}],
        "criteria": {"products_returned": True, "multi_merchant": True},
        "clarification_acceptable": False,
    },
    {
        "id": 19, "category": "cross_merchant",
        "messages": [{"role": "user", "content": "show me something for a dog and a fashion item for a woman"}],
        "criteria": {"products_returned": True},
        "clarification_acceptable": False,
    },
    {
        "id": 20, "category": "cross_merchant",
        "messages": [{"role": "user", "content": "I need dog food and also a skirt or dress"}],
        "criteria": {"products_returned": True},
        "clarification_acceptable": False,
    },

    # ── E. Store / location request (3 scenarios) ────────────────────────
    {
        "id": 21, "category": "store_location",
        "messages": [{"role": "user", "content": "where are the fashion stores located? I'm in Como"}],
        "criteria": {"stores_returned": True},
        "clarification_acceptable": False,
    },
    {
        "id": 22, "category": "store_location",
        "messages": [{"role": "user", "content": "find stores in Milan"}],
        "criteria": {"stores_returned": True},
        "clarification_acceptable": False,
    },
    {
        "id": 23, "category": "store_location",
        "messages": [{"role": "user", "content": "show me all pet supply store locations"}],
        "criteria": {"stores_returned": True},
        "clarification_acceptable": False,
    },

    # ── F. Edge cases: multilingual (2 scenarios) ────────────────────────
    {
        "id": 24, "category": "edge_cases",
        "messages": [{"role": "user", "content": "Hola, busco un vestido de verano bonito"}],
        "criteria": {"products_returned": True, "correct_merchant": "twinset"},
        "clarification_acceptable": True,
    },
    {
        "id": 25, "category": "edge_cases",
        "messages": [{"role": "user", "content": "こんにちは、猫のご飯を探しています"}],
        "criteria": {"products_returned": True, "correct_merchant": "arcaplanet"},
        "clarification_acceptable": True,
    },
]


def _parse_price(price_str) -> float | None:
    if not price_str:
        return None
    try:
        return float(str(price_str).replace(",", ".").strip().split()[0])
    except (ValueError, IndexError):
        return None


def evaluate_scenario(scenario: dict, response: dict, is_clarification: bool) -> dict:
    """Evaluate a scenario's criteria against the chat response.

    is_clarification: True if the assistant asked follow-up questions instead
    of returning products. Only counts as success if clarification_acceptable.
    """
    criteria = scenario["criteria"]
    clar_ok = scenario.get("clarification_acceptable", False)
    products = response.get("products") or []
    stores = response.get("stores") or []
    follow_ups = response.get("follow_up_questions") or []

    results = {}
    passed = []

    # products_returned
    if "products_returned" in criteria:
        got_products = len(products) > 0
        got_clarification = len(follow_ups) > 0 and clar_ok
        ok = got_products or got_clarification
        results["products_returned"] = got_products
        results["clarification_returned"] = len(follow_ups) > 0
        results["clarification_acceptable"] = clar_ok
        passed.append(ok)

    # stores_returned
    if "stores_returned" in criteria:
        got = len(stores) > 0
        results["stores_returned"] = got
        passed.append(got == criteria["stores_returned"])

    # correct_merchant: ≥80% of products from expected merchant
    if "correct_merchant" in criteria and products:
        exp = criteria["correct_merchant"]
        matching = sum(1 for p in products if p.get("merchant", "") == exp)
        ratio = matching / len(products)
        ok = ratio >= 0.80
        results["correct_merchant"] = ok
        results["correct_merchant_ratio"] = round(ratio, 3)
        passed.append(ok)

    # multi_merchant: products from both merchants
    if "multi_merchant" in criteria and products:
        merchants_seen = {p.get("merchant", "") for p in products}
        ok = len(merchants_seen) >= 2
        results["multi_merchant"] = ok
        results["merchants_seen"] = sorted(merchants_seen)
        passed.append(ok)

    success = all(passed) if passed else False
    return {"success": success, "criteria_results": results}


async def run_single_turn(messages: list, app_state) -> dict:
    """Run one chat turn and return the response dict."""
    from api.chat import chat
    return await chat(messages=messages, image_bytes=None, app_state=app_state)


async def run_scenario(scenario: dict, app_state) -> dict:
    """Run a scenario (single or multi-turn) and return result dict."""
    is_multi = "turn_1" in scenario

    t0 = time.time()
    try:
        if is_multi:
            # Turn 1: send first user message
            turn1_msg = scenario["turn_1"]
            messages_t1 = [{"role": "user", "content": turn1_msg}]
            response1 = await run_single_turn(messages_t1, app_state)

            # Build turn 2: inject assistant reply, append second user message
            assistant_reply = response1.get("message", "")
            messages_t2 = [
                {"role": "user", "content": turn1_msg},
                {"role": "assistant", "content": assistant_reply},
                {"role": "user", "content": scenario["turn_2"]},
            ]
            response = await run_single_turn(messages_t2, app_state)
            display_msg = scenario["turn_1"]  # for display
        else:
            messages = scenario.get("messages", [])
            response = await run_single_turn(messages, app_state)
            display_msg = messages[0]["content"] if messages else ""

        elapsed = round(time.time() - t0, 2)
        follow_ups = response.get("follow_up_questions") or []
        is_clarification = len(follow_ups) > 0 and len(response.get("products") or []) == 0

        eval_result = evaluate_scenario(scenario, response, is_clarification)

        return {
            "id": scenario["id"],
            "category": scenario["category"],
            "display_query": display_msg[:80],
            "multi_turn": is_multi,
            "clarification_acceptable": scenario.get("clarification_acceptable", False),
            "criteria": scenario["criteria"],
            "response_message": response.get("message", "")[:300],
            "n_products_returned": len(response.get("products") or []),
            "n_stores_returned": len(response.get("stores") or []),
            "n_follow_up_questions": len(follow_ups),
            "elapsed_s": elapsed,
            "success": eval_result["success"],
            "criteria_results": eval_result["criteria_results"],
            "error": None,
        }

    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        print(f"  ERROR in scenario {scenario['id']}: {exc}", flush=True)
        return {
            "id": scenario["id"],
            "category": scenario["category"],
            "display_query": scenario.get("turn_1", (scenario.get("messages") or [{}])[0].get("content", ""))[:80],
            "multi_turn": is_multi,
            "clarification_acceptable": scenario.get("clarification_acceptable", False),
            "criteria": scenario["criteria"],
            "response_message": "",
            "n_products_returned": 0,
            "n_stores_returned": 0,
            "n_follow_up_questions": 0,
            "elapsed_s": elapsed,
            "success": False,
            "criteria_results": {},
            "error": str(exc),
        }


async def main_async():
    t_start = time.time()
    device = pick_device(None)
    print(f"Device: {device}", flush=True)

    print("Loading CLIP model...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()

    print("Loading product index...", flush=True)
    matrix, meta = load_index()
    print(f"  {matrix.shape[0]} products x {matrix.shape[1]} dims", flush=True)

    conn = sqlite3.connect(str(DB_PATH))

    # Build product_cities lookup (same logic as api/main.py startup)
    product_cities: dict[str, set[str]] = {}
    try:
        rows = conn.execute(
            "SELECT DISTINCT t.product_id, sp.city FROM transactions t "
            "JOIN store_profiles sp ON t.store_id = sp.store_id "
            "WHERE sp.city IS NOT NULL"
        ).fetchall()
        for pid, city in rows:
            product_cities.setdefault(pid, set()).add(city.strip().title())
    except Exception:
        pass

    app_state = types.SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        matrix=matrix,
        meta=meta,
        device=device,
        conn=conn,
        product_cities=product_cities,
    )

    print(f"\nRunning {len(SCENARIOS)} scenarios...\n", flush=True)
    scenario_results = []

    for scenario in SCENARIOS:
        disp = scenario.get("turn_1", (scenario.get("messages") or [{}])[0].get("content", ""))
        multi = "(2-turn)" if "turn_1" in scenario else ""
        clar = "[clar ok]" if scenario.get("clarification_acceptable") else ""
        print(f"  [{scenario['id']:02d}/{len(SCENARIOS)}] {scenario['category']} {multi}{clar}: "
              f"{disp[:55]!r}", flush=True)
        result = await run_scenario(scenario, app_state)
        status = "PASS" if result["success"] else "FAIL"
        fup = f" fup={result['n_follow_up_questions']}" if result["n_follow_up_questions"] else ""
        print(f"         {status} | prods={result['n_products_returned']} "
              f"stores={result['n_stores_returned']}{fup} ({result['elapsed_s']}s)", flush=True)
        if result["error"]:
            print(f"         error: {result['error']}", flush=True)
        scenario_results.append(result)

    conn.close()

    # ── Aggregate ─────────────────────────────────────────────────────────
    categories = ["simple_search", "multi_turn", "budget_constrained",
                  "cross_merchant", "store_location", "edge_cases"]
    by_category = {}
    for cat in categories:
        cat_results = [r for r in scenario_results if r["category"] == cat]
        if cat_results:
            n = len(cat_results)
            n_success = sum(1 for r in cat_results if r["success"])
            by_category[cat] = {
                "n": n,
                "n_success": n_success,
                "success_rate": round(n_success / n, 4),
            }

    total = len(scenario_results)
    total_success = sum(1 for r in scenario_results if r["success"])
    overall_rate = round(total_success / total, 4) if total > 0 else 0.0

    # Also count scenarios where products returned (strict, ignoring clarification credit)
    strict_success = sum(1 for r in scenario_results
                         if r["n_products_returned"] > 0 or r["n_stores_returned"] > 0)

    output = {
        "n_scenarios": total,
        "n_success": total_success,
        "overall_task_success_rate": overall_rate,
        "strict_product_or_store_rate": round(strict_success / total, 4),
        "elapsed_total_s": round(time.time() - t_start, 1),
        "notes": (
            "success=True if: products returned, OR (clarification_acceptable AND "
            "follow_up_questions returned). Multi-turn scenarios do 2 sequential API calls."
        ),
        "by_category": by_category,
        "scenarios": scenario_results,
    }

    print(f"\n{'='*60}")
    print(f"Overall success rate : {total_success}/{total} = {overall_rate:.4f}")
    print(f"Strict (prod/store)  : {strict_success}/{total} = {output['strict_product_or_store_rate']:.4f}")
    for cat, stats in by_category.items():
        print(f"  {cat:25s}: {stats['n_success']}/{stats['n']} = {stats['success_rate']:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
