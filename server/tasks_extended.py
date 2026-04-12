"""
server/tasks_extended.py — Extended Task Definitions for RL Training Loop.

Tasks 4-10 for the DataCleaner RL environment.
Each task includes: input data, gold answer, deterministic grader, max_steps.

Task 4  (Medium)  : Currency normalization (multi-currency → USD)
Task 5  (Medium)  : Address standardization (free-text → structured)
Task 6  (Easy)    : Multi-format date normalization → ISO 8601
Task 7  (Medium)  : Phone number normalization → E.164
Task 8  (Hard)    : Product taxonomy mapping (L1 > L2 > L3)
Task 9  (Hard)    : Null/missing value imputation (statistical rules)
Task 10 (Hard)    : Unit conversion + data type coercion (imperial → metric)

Score contract: all scores strictly in (0.01, 0.99) via _clamp().
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

_EPS: float = 0.01


def _clamp(score: float) -> float:
    return round(max(_EPS, min(float(score), 1.0 - _EPS)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — Currency Normalization (Multi-currency → USD)
# ─────────────────────────────────────────────────────────────────────────────

TASK4_EXCHANGE_RATES = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "CAD": 0.74}

TASK4_INPUT = [
    {"product_id": "P001", "name": "Laptop Pro",     "price": 1299.99, "currency": "USD"},
    {"product_id": "P002", "name": "Wireless Kbd",   "price": 89.00,   "currency": "EUR"},
    {"product_id": "P003", "name": "USB-C Hub",      "price": 45.50,   "currency": "GBP"},
    {"product_id": "P004", "name": "Webcam HD",      "price": 12800,   "currency": "JPY"},
    {"product_id": "P005", "name": "Desk Lamp LED",  "price": 65.00,   "currency": "CAD"},
    {"product_id": "P006", "name": "Mouse Pad XL",   "price": 24.99,   "currency": "USD"},
    {"product_id": "P007", "name": "Monitor 27in",   "price": 349.00,  "currency": "EUR"},
    {"product_id": "P008", "name": "Headset Pro",    "price": 159.00,  "currency": "GBP"},
]

TASK4_GOLD = [
    {"product_id": "P001", "name": "Laptop Pro",     "price_usd": 1299.99, "currency": "USD"},
    {"product_id": "P002", "name": "Wireless Kbd",   "price_usd": round(89.00   * 1.08, 2), "currency": "USD"},
    {"product_id": "P003", "name": "USB-C Hub",      "price_usd": round(45.50   * 1.27, 2), "currency": "USD"},
    {"product_id": "P004", "name": "Webcam HD",      "price_usd": round(12800   * 0.0067, 2), "currency": "USD"},
    {"product_id": "P005", "name": "Desk Lamp LED",  "price_usd": round(65.00   * 0.74, 2), "currency": "USD"},
    {"product_id": "P006", "name": "Mouse Pad XL",   "price_usd": 24.99, "currency": "USD"},
    {"product_id": "P007", "name": "Monitor 27in",   "price_usd": round(349.00  * 1.08, 2), "currency": "USD"},
    {"product_id": "P008", "name": "Headset Pro",    "price_usd": round(159.00  * 1.27, 2), "currency": "USD"},
]


def grade_task4(payload: Any) -> Tuple[float, str]:
    records = payload if isinstance(payload, list) else payload.get("records", [])
    gold_map = {r["product_id"]: r for r in TASK4_GOLD}
    n = len(TASK4_GOLD)
    correct = 0
    feedback_parts = []

    for agent_rec in records:
        pid  = agent_rec.get("product_id", "")
        gold = gold_map.get(pid)
        if not gold:
            continue
        currency_ok = str(agent_rec.get("currency", "")).upper() == "USD"
        agent_price = agent_rec.get("price_usd") or agent_rec.get("price")
        try:
            price_ok = abs(float(agent_price) - gold["price_usd"]) <= 0.05
        except Exception:
            price_ok = False
        if currency_ok and price_ok:
            correct += 1
        else:
            feedback_parts.append(
                f"✗ {pid}: got price={agent_price} currency={agent_rec.get('currency')} "
                f"expected price_usd={gold['price_usd']} currency=USD"
            )

    score = _clamp(correct / n)
    feedback = (
        f"Task4 score={score:.2f} | correct={correct}/{n}\n"
        + "\n".join(feedback_parts[:6])
    )
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — Address Standardization (free-text → structured components)
# ─────────────────────────────────────────────────────────────────────────────

TASK5_INPUT = [
    {"id": "A001", "raw": "350 Fifth Avenue, New York, NY 10118, United States"},
    {"id": "A002", "raw": "1600 Pennsylvania Ave NW Washington DC 20500 US"},
    {"id": "A003", "raw": "1 Infinite Loop, Cupertino, California, 95014"},
    {"id": "A004", "raw": "410 Terry Ave North  Seattle  WA  98109"},
    {"id": "A005", "raw": "1355 Market St Suite 900 San Francisco CA 94103 USA"},
    {"id": "A006", "raw": "500 W 2nd St, Austin Texas 78701, United States"},
]

TASK5_GOLD = [
    {"id": "A001", "street": "350 Fifth Avenue",      "city": "New York",     "state": "NY", "zip": "10118", "country": "US"},
    {"id": "A002", "street": "1600 Pennsylvania Ave NW", "city": "Washington", "state": "DC", "zip": "20500", "country": "US"},
    {"id": "A003", "street": "1 Infinite Loop",       "city": "Cupertino",    "state": "CA", "zip": "95014", "country": "US"},
    {"id": "A004", "street": "410 Terry Ave North",   "city": "Seattle",      "state": "WA", "zip": "98109", "country": "US"},
    {"id": "A005", "street": "1355 Market St Suite 900", "city": "San Francisco", "state": "CA", "zip": "94103", "country": "US"},
    {"id": "A006", "street": "500 W 2nd St",          "city": "Austin",       "state": "TX", "zip": "78701", "country": "US"},
]

_STATE_ABBR = {
    "california": "CA", "new york": "NY", "washington": "WA",
    "texas": "TX", "illinois": "IL", "florida": "FL", "district of columbia": "DC",
}


def grade_task5(payload: Any) -> Tuple[float, str]:
    records = payload if isinstance(payload, list) else payload.get("addresses", payload.get("records", []))
    gold_map = {r["id"]: r for r in TASK5_GOLD}
    fields = ["street", "city", "state", "zip", "country"]
    n = len(TASK5_GOLD)
    total = n * len(fields)
    correct = 0
    feedback_parts = []

    for agent_rec in records:
        aid  = agent_rec.get("id", "")
        gold = gold_map.get(aid)
        if not gold:
            continue
        for f in fields:
            av = str(agent_rec.get(f, "") or "").strip()
            gv = str(gold.get(f, "")).strip()
            # Normalize state full names → abbreviations
            if f == "state":
                av_norm = _STATE_ABBR.get(av.lower(), av.upper())
                match = av_norm == gv.upper()
            elif f == "country":
                av_norm = av.upper().replace("UNITED STATES", "US").replace("USA", "US")
                match = av_norm in ("US", "USA")
            else:
                match = av.lower() == gv.lower()
            if match:
                correct += 1
            else:
                feedback_parts.append(f"✗ {aid}.{f}: got={av!r} expected={gv!r}")

    score = _clamp(correct / total)
    feedback = (
        f"Task5 score={score:.2f} | correct={correct}/{total}\n"
        + "\n".join(feedback_parts[:8])
    )
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — Multi-Format Date Normalization → ISO 8601 (YYYY-MM-DD)
# ─────────────────────────────────────────────────────────────────────────────

TASK6_INPUT = [
    {"id": "D01", "raw_date": "03/15/2024"},
    {"id": "D02", "raw_date": "15-03-2024"},
    {"id": "D03", "raw_date": "March 15, 2024"},
    {"id": "D04", "raw_date": "15 Mar 2024"},
    {"id": "D05", "raw_date": "2024.03.15"},
    {"id": "D06", "raw_date": "20240315"},
    {"id": "D07", "raw_date": "Mar 15 2024"},
    {"id": "D08", "raw_date": "15/03/24"},
    {"id": "D09", "raw_date": "3-15-24"},
    {"id": "D10", "raw_date": "2024-03-15T09:30:00Z"},
]

TASK6_GOLD = [
    {"id": "D01", "normalized_date": "2024-03-15"},
    {"id": "D02", "normalized_date": "2024-03-15"},
    {"id": "D03", "normalized_date": "2024-03-15"},
    {"id": "D04", "normalized_date": "2024-03-15"},
    {"id": "D05", "normalized_date": "2024-03-15"},
    {"id": "D06", "normalized_date": "2024-03-15"},
    {"id": "D07", "normalized_date": "2024-03-15"},
    {"id": "D08", "normalized_date": "2024-03-15"},
    {"id": "D09", "normalized_date": "2024-03-15"},
    {"id": "D10", "normalized_date": "2024-03-15"},
]

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def grade_task6(payload: Any) -> Tuple[float, str]:
    records = payload if isinstance(payload, list) else payload.get("dates", payload.get("records", []))
    gold_map = {r["id"]: r["normalized_date"] for r in TASK6_GOLD}
    n = len(TASK6_GOLD)
    correct = 0
    feedback_parts = []

    for agent_rec in records:
        did   = agent_rec.get("id", "")
        gold  = gold_map.get(did, "")
        agent_date = str(agent_rec.get("normalized_date", "") or "").strip()
        # Accept truncated ISO from datetime strings
        if "T" in agent_date:
            agent_date = agent_date.split("T")[0]
        if agent_date == gold:
            correct += 1
        else:
            feedback_parts.append(f"✗ {did}: got={agent_date!r} expected={gold!r}")

    score = _clamp(correct / n)
    feedback = (
        f"Task6 score={score:.2f} | correct={correct}/{n}\n"
        + "\n".join(feedback_parts[:6])
    )
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 — Phone Number Normalization → E.164 (+1XXXXXXXXXX)
# ─────────────────────────────────────────────────────────────────────────────

TASK7_INPUT = [
    {"id": "PH01", "raw_phone": "(212) 555-1234"},
    {"id": "PH02", "raw_phone": "212.555.5678"},
    {"id": "PH03", "raw_phone": "212-555-9012"},
    {"id": "PH04", "raw_phone": "+1 (415) 555-3456"},
    {"id": "PH05", "raw_phone": "1-800-555-7890"},
    {"id": "PH06", "raw_phone": "5553214567"},
    {"id": "PH07", "raw_phone": "+14155552671"},
    {"id": "PH08", "raw_phone": "312 555 0011"},
]

TASK7_GOLD = [
    {"id": "PH01", "e164": "+12125551234"},
    {"id": "PH02", "e164": "+12125555678"},
    {"id": "PH03", "e164": "+12125559012"},
    {"id": "PH04", "e164": "+14155553456"},
    {"id": "PH05", "e164": "+18005557890"},
    {"id": "PH06", "e164": "+15553214567"},
    {"id": "PH07", "e164": "+14155552671"},
    {"id": "PH08", "e164": "+13125550011"},
]


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", str(s))


def grade_task7(payload: Any) -> Tuple[float, str]:
    records = payload if isinstance(payload, list) else payload.get("phones", payload.get("records", []))
    gold_map = {r["id"]: r["e164"] for r in TASK7_GOLD}
    n = len(TASK7_GOLD)
    correct = 0
    feedback_parts = []

    for agent_rec in records:
        pid       = agent_rec.get("id", "")
        gold_e164 = gold_map.get(pid, "")
        agent_e164 = str(agent_rec.get("e164", "") or "").strip()
        # Normalize: strip spaces, ensure + prefix, then compare digits
        gold_digits  = _digits_only(gold_e164)
        agent_digits = _digits_only(agent_e164)
        # Accept if digits match exactly (with or without leading 1)
        match = (agent_e164 == gold_e164) or (agent_digits == gold_digits and agent_e164.startswith("+"))
        if match:
            correct += 1
        else:
            feedback_parts.append(f"✗ {pid}: got={agent_e164!r} expected={gold_e164!r}")

    score = _clamp(correct / n)
    feedback = (
        f"Task7 score={score:.2f} | correct={correct}/{n}\n"
        + "\n".join(feedback_parts[:6])
    )
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# TASK 8 — Product Taxonomy Mapping (L1 > L2 > L3)
# ─────────────────────────────────────────────────────────────────────────────

TASK8_INPUT = [
    {"product_id": "T001", "name": "Wireless Mouse",     "raw_category": "computer accessories"},
    {"product_id": "T002", "name": "Running Shoes",       "raw_category": "athletic footwear"},
    {"product_id": "T003", "name": "4K Monitor",          "raw_category": "display screens"},
    {"product_id": "T004", "name": "Protein Powder",      "raw_category": "supplements"},
    {"product_id": "T005", "name": "Yoga Mat",            "raw_category": "fitness gear"},
    {"product_id": "T006", "name": "USB-C Cable",         "raw_category": "cables and adapters"},
    {"product_id": "T007", "name": "Coffee Maker",        "raw_category": "kitchen appliances"},
    {"product_id": "T008", "name": "Python Programming Book", "raw_category": "tech books"},
]

TASK8_GOLD = [
    {"product_id": "T001", "l1": "Electronics", "l2": "Computers & Accessories", "l3": "Input Devices"},
    {"product_id": "T002", "l1": "Sports",       "l2": "Footwear",                "l3": "Running"},
    {"product_id": "T003", "l1": "Electronics", "l2": "Computers & Accessories", "l3": "Monitors & Displays"},
    {"product_id": "T004", "l1": "Health",       "l2": "Sports Nutrition",        "l3": "Protein Supplements"},
    {"product_id": "T005", "l1": "Sports",       "l2": "Exercise & Fitness",      "l3": "Yoga"},
    {"product_id": "T006", "l1": "Electronics", "l2": "Cables & Accessories",     "l3": "USB Cables"},
    {"product_id": "T007", "l1": "Home",         "l2": "Kitchen & Dining",        "l3": "Coffee Makers"},
    {"product_id": "T008", "l1": "Books",        "l2": "Computers & Technology",  "l3": "Programming"},
]


def grade_task8(payload: Any) -> Tuple[float, str]:
    records = payload if isinstance(payload, list) else payload.get("taxonomy", payload.get("records", []))
    gold_map = {r["product_id"]: r for r in TASK8_GOLD}
    n = len(TASK8_GOLD)
    total = n * 3
    correct = 0
    feedback_parts = []

    for agent_rec in records:
        pid  = agent_rec.get("product_id", "")
        gold = gold_map.get(pid)
        if not gold:
            continue
        for level in ["l1", "l2", "l3"]:
            av = str(agent_rec.get(level, "") or "").strip().lower()
            gv = str(gold.get(level, "")).strip().lower()
            if av == gv:
                correct += 1
            else:
                feedback_parts.append(f"✗ {pid}.{level}: got={av!r} expected={gv!r}")

    score = _clamp(correct / total)
    feedback = (
        f"Task8 score={score:.2f} | correct={correct}/{total}\n"
        + "\n".join(feedback_parts[:8])
    )
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# TASK 9 — Null / Missing Value Imputation (statistical rules)
# ─────────────────────────────────────────────────────────────────────────────

TASK9_INPUT = [
    {"id": "R01", "category": "Electronics", "price": 299.99, "rating": None,  "stock": 50},
    {"id": "R02", "category": "Electronics", "price": 149.50, "rating": 4.2,   "stock": None},
    {"id": "R03", "category": "Electronics", "price": None,   "rating": 3.8,   "stock": 75},
    {"id": "R04", "category": "Clothing",    "price": 49.99,  "rating": None,  "stock": 200},
    {"id": "R05", "category": "Clothing",    "price": 79.95,  "rating": 4.5,   "stock": None},
    {"id": "R06", "category": "Clothing",    "price": None,   "rating": 4.0,   "stock": 150},
    {"id": "R07", "category": "Books",       "price": 24.99,  "rating": 4.7,   "stock": None},
    {"id": "R08", "category": "Books",       "price": 19.99,  "rating": None,  "stock": 300},
    {"id": "R09", "category": "Books",       "price": None,   "rating": 4.3,   "stock": 180},
]

# Gold: median by category for numerics
TASK9_GOLD = [
    {"id": "R01", "category": "Electronics", "price": 299.99, "rating": 4.0,  "stock": 50},    # rating: median(4.2,3.8)=4.0
    {"id": "R02", "category": "Electronics", "price": 149.50, "rating": 4.2,  "stock": 63},    # stock: median(50,75)=62.5→63
    {"id": "R03", "category": "Electronics", "price": 224.75, "rating": 3.8,  "stock": 75},    # price: median(299.99,149.50)=224.74→224.75
    {"id": "R04", "category": "Clothing",    "price": 49.99,  "rating": 4.25, "stock": 200},   # rating: median(4.5,4.0)=4.25
    {"id": "R05", "category": "Clothing",    "price": 79.95,  "rating": 4.5,  "stock": 175},   # stock: median(200,150)=175
    {"id": "R06", "category": "Clothing",    "price": 64.97,  "rating": 4.0,  "stock": 150},   # price: median(49.99,79.95)=64.97
    {"id": "R07", "category": "Books",       "price": 24.99,  "rating": 4.7,  "stock": 240},   # stock: median(300,180)=240
    {"id": "R08", "category": "Books",       "price": 19.99,  "rating": 4.5,  "stock": 300},   # rating: median(4.7,4.3)=4.5
    {"id": "R09", "category": "Books",       "price": 22.49,  "rating": 4.3,  "stock": 180},   # price: median(24.99,19.99)=22.49
]


def grade_task9(payload: Any) -> Tuple[float, str]:
    records = payload if isinstance(payload, list) else payload.get("records", [])
    gold_map = {r["id"]: r for r in TASK9_GOLD}
    fields = ["price", "rating", "stock"]
    n = len(TASK9_GOLD)
    total = n * len(fields)
    correct = 0
    feedback_parts = []

    for agent_rec in records:
        rid  = agent_rec.get("id", "")
        gold = gold_map.get(rid)
        if not gold:
            continue
        for f in fields:
            av = agent_rec.get(f)
            gv = gold.get(f)
            if av is None or gv is None:
                feedback_parts.append(f"✗ {rid}.{f}: got=None expected={gv}")
                continue
            try:
                # 5% tolerance for numeric fields
                match = abs(float(av) - float(gv)) / (abs(float(gv)) + 1e-9) <= 0.05
            except Exception:
                match = False
            if match:
                correct += 1
            else:
                feedback_parts.append(f"✗ {rid}.{f}: got={av} expected={gv}")

    score = _clamp(correct / total)
    feedback = (
        f"Task9 score={score:.2f} | correct={correct}/{total}\n"
        + "\n".join(feedback_parts[:8])
    )
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# TASK 10 — Unit Conversion + Data Type Coercion (imperial → metric)
# ─────────────────────────────────────────────────────────────────────────────

TASK10_INPUT = [
    {"id": "U01", "name": "Laptop",       "weight": "4.5 lbs",   "height": "0.7 in",    "storage": "512 GB",    "price": "1299"},
    {"id": "U02", "name": "Monitor",      "weight": "12.3 lbs",  "diagonal": "27 in",   "power": "85 watts",    "price": "449.99"},
    {"id": "U03", "name": "Coffee Maker", "weight": "2200 g",    "capacity": "1.5 L",   "wattage": "1200 W",    "price": "89"},
    {"id": "U04", "name": "Running Shoes","weight": "310 g",     "length": "11.5 in",   "width": "4.2 in",      "price": "129.95"},
    {"id": "U05", "name": "Protein Pack", "weight": "5 lbs",     "serving_ml": "250 ml","calories": "200",       "price": "54.99"},
    {"id": "U06", "name": "Yoga Mat",     "weight": "1.8 kg",    "length": "72 in",     "width": "24 in",        "price": "45"},
]

# Conversion: 1 lb = 0.4536 kg, 1 in = 2.54 cm
TASK10_GOLD = [
    {"id": "U01", "name": "Laptop",       "weight_kg": 2.04,  "price": 1299.0,  "storage_gb": 512},
    {"id": "U02", "name": "Monitor",      "weight_kg": 5.58,  "price": 449.99,  "diagonal_cm": 68.58},
    {"id": "U03", "name": "Coffee Maker", "weight_kg": 2.2,   "price": 89.0,    "capacity_l": 1.5},
    {"id": "U04", "name": "Running Shoes","weight_kg": 0.31,  "price": 129.95,  "length_cm": 29.21},
    {"id": "U05", "name": "Protein Pack", "weight_kg": 2.27,  "price": 54.99,   "serving_ml": 250},
    {"id": "U06", "name": "Yoga Mat",     "weight_kg": 1.8,   "price": 45.0,    "length_cm": 182.88},
]


def grade_task10(payload: Any) -> Tuple[float, str]:
    records = payload if isinstance(payload, list) else payload.get("records", [])
    gold_map = {r["id"]: r for r in TASK10_GOLD}
    n = len(TASK10_GOLD)
    # Score per-record: weight_kg + price are mandatory (2 fields each)
    total = n * 2
    correct = 0
    feedback_parts = []

    for agent_rec in records:
        uid  = agent_rec.get("id", "")
        gold = gold_map.get(uid)
        if not gold:
            continue
        for f in ["weight_kg", "price"]:
            av = agent_rec.get(f)
            gv = gold.get(f)
            if av is None or gv is None:
                feedback_parts.append(f"✗ {uid}.{f}: missing")
                continue
            try:
                tol = abs(float(gv)) * 0.02   # 2% tolerance
                match = abs(float(av) - float(gv)) <= max(tol, 0.01)
            except Exception:
                match = False
            if match:
                correct += 1
            else:
                feedback_parts.append(f"✗ {uid}.{f}: got={av} expected={gv:.3f}")

    score = _clamp(correct / total)
    feedback = (
        f"Task10 score={score:.2f} | correct={correct}/{total}\n"
        + "\n".join(feedback_parts[:8])
    )
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# Public TASKS_EXTENDED dict — merged with TASKS in environment
# ─────────────────────────────────────────────────────────────────────────────

TASKS_EXTENDED = {
    4: {
        "name": "currency-normalization",
        "instruction": (
            "You are given a list of product records with prices in different currencies "
            "(USD, EUR, GBP, JPY, CAD). Use the following exchange rates to convert all prices to USD: "
            "EUR=1.08, GBP=1.27, JPY=0.0067, CAD=0.74. "
            "Return a list of records with fields: product_id, name, price_usd (float, 2dp), currency='USD'."
        ),
        "input_data":  {"products": TASK4_INPUT, "exchange_rates": TASK4_EXCHANGE_RATES},
        "schema_hint": {"product_id": "str", "name": "str", "price_usd": "float (2dp)", "currency": "'USD'"},
        "grader":      grade_task4,
        "max_steps":   3,
    },
    5: {
        "name": "address-standardization",
        "instruction": (
            "Parse each raw address string into structured components: "
            "street (str), city (str), state (2-letter abbreviation), zip (5-digit str), country ('US'). "
            "Return a list of records with: id, street, city, state, zip, country."
        ),
        "input_data":  TASK5_INPUT,
        "schema_hint": {"id": "str", "street": "str", "city": "str", "state": "2-letter abbr", "zip": "5-digit str", "country": "'US'"},
        "grader":      grade_task5,
        "max_steps":   3,
    },
    6: {
        "name": "date-normalization",
        "instruction": (
            "Normalize all date strings to ISO 8601 format: YYYY-MM-DD. "
            "Input dates may be in formats: MM/DD/YYYY, DD-MM-YYYY, Month DD YYYY, DD Mon YYYY, "
            "YYYY.MM.DD, YYYYMMDD, MM/DD/YY, MM-DD-YY, or ISO with time (strip time component). "
            "Return a list of records with: id, normalized_date (YYYY-MM-DD)."
        ),
        "input_data":  TASK6_INPUT,
        "schema_hint": {"id": "str", "normalized_date": "YYYY-MM-DD"},
        "grader":      grade_task6,
        "max_steps":   2,
    },
    7: {
        "name": "phone-normalization",
        "instruction": (
            "Normalize all US phone numbers to E.164 format: +1XXXXXXXXXX (12 chars total). "
            "Strip all non-digit characters, ensure US country code (+1), "
            "and handle numbers already in E.164. "
            "Return a list of records with: id, e164 (E.164 formatted phone)."
        ),
        "input_data":  TASK7_INPUT,
        "schema_hint": {"id": "str", "e164": "+1XXXXXXXXXX"},
        "grader":      grade_task7,
        "max_steps":   2,
    },
    8: {
        "name": "taxonomy-mapping",
        "instruction": (
            "Map each product to a 3-level taxonomy: L1 (top-level), L2 (sub-category), L3 (leaf). "
            "Use standard e-commerce taxonomy (Electronics, Sports, Health, Home, Books, etc.). "
            "Return a list of records with: product_id, l1, l2, l3."
        ),
        "input_data":  TASK8_INPUT,
        "schema_hint": {"product_id": "str", "l1": "str", "l2": "str", "l3": "str"},
        "grader":      grade_task8,
        "max_steps":   3,
    },
    9: {
        "name": "null-imputation",
        "instruction": (
            "Impute all null values using these rules:\n"
            "  - Numeric fields (price, rating, stock): use MEDIAN of non-null values in the SAME CATEGORY.\n"
            "  - Round price to 2dp, rating to 2dp, stock to nearest integer.\n"
            "  - Never leave a field null after imputation.\n"
            "Return ALL records (including those with no nulls) with: id, category, price, rating, stock."
        ),
        "input_data":  TASK9_INPUT,
        "schema_hint": {"id": "str", "category": "str", "price": "float (2dp)", "rating": "float (2dp)", "stock": "int"},
        "grader":      grade_task9,
        "max_steps":   4,
    },
    10: {
        "name": "unit-conversion",
        "instruction": (
            "Convert all measurements to metric units and coerce all types:\n"
            "  - Weight: lbs → kg (1 lb = 0.4536 kg), grams → kg. Round to 2dp.\n"
            "  - Length: inches → cm (1 in = 2.54 cm). Round to 2dp.\n"
            "  - Price: string → float (2dp).\n"
            "  - Storage: keep as int GB.\n"
            "Return records with: id, name, weight_kg, price, and any converted dimension fields."
        ),
        "input_data":  TASK10_INPUT,
        "schema_hint": {"id": "str", "name": "str", "weight_kg": "float (2dp)", "price": "float (2dp)"},
        "grader":      grade_task10,
        "max_steps":   3,
    },
}

# Gold payloads for judge reference
TASK_GOLD_EXTENDED = {
    4:  TASK4_GOLD,
    5:  TASK5_GOLD,
    6:  TASK6_GOLD,
    7:  TASK7_GOLD,
    8:  TASK8_GOLD,
    9:  TASK9_GOLD,
    10: TASK10_GOLD,
}
