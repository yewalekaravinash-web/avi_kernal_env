"""
Task definitions, datasets, and graders for the Data Cleaner Environment.

Task 1 (Easy)  : Schema-aligned field extraction from a raw JSON blob.
Task 2 (Medium): Duplicate detection & merge on a 20-record CSV-like list.
Task 3 (Hard)  : Multi-source reconciliation with conflict resolution policy.

All graders are fully deterministic — no LLM-as-judge.
"""
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# TASK 1 — Easy: Field Extraction
# ---------------------------------------------------------------------------

TASK1_INPUT = {
    "prod_id": "P-00192",
    "ProductName": "  Wireless Mouse  ",
    "price_usd": "29.99",
    "listed_date": "03/15/2024",
    "qty_available": "142",
    "Category": "Electronics",
    "brand_name": "LogiTech",
    "Rating": "4.5",
    "disc_pct": "10",
}

TASK1_SCHEMA = {
    "product_id": "str",
    "name": "str (trimmed)",
    "price": "float",
    "listed_date": "YYYY-MM-DD",
    "quantity": "int",
    "category": "str (lowercase)",
    "brand": "str (lowercase)",
    "rating": "float",
    "discount_pct": "int",
}

TASK1_GOLD = {
    "product_id": "P-00192",
    "name": "Wireless Mouse",
    "price": 29.99,
    "listed_date": "2024-03-15",
    "quantity": 142,
    "category": "electronics",
    "brand": "logitech",
    "rating": 4.5,
    "discount_pct": 10,
}


def grade_task1(payload: Dict[str, Any]) -> Tuple[float, str]:
    """Field-level match score, averaged across 9 fields."""
    gold = TASK1_GOLD
    fields = list(gold.keys())
    correct = 0
    feedback_parts = []
    for f in fields:
        agent_val = payload.get(f)
        gold_val = gold[f]
        # Normalize comparison
        try:
            if isinstance(gold_val, float):
                match = abs(float(agent_val) - gold_val) < 0.001
            elif isinstance(gold_val, int):
                match = int(agent_val) == gold_val
            else:
                match = str(agent_val).strip() == str(gold_val).strip()
        except Exception:
            match = False
        if match:
            correct += 1
            feedback_parts.append(f"✓ {f}")
        else:
            feedback_parts.append(f"✗ {f} (got {agent_val!r}, expected {gold_val!r})")
    score = round(correct / len(fields), 4)
    feedback = f"Task1 score={score:.2f} [{correct}/{len(fields)} fields correct]\n" + "\n".join(feedback_parts)
    return score, feedback


# ---------------------------------------------------------------------------
# TASK 2 — Medium: Duplicate Detection & Merge
# ---------------------------------------------------------------------------

TASK2_RECORDS = [
    {"id": 1,  "company": "Acme Corp",        "address": "123 Main St, Boston, MA 02101", "phone": "617-555-0101", "email": "info@acme.com"},
    {"id": 2,  "company": "ACME Corporation", "address": "123 Main Street, Boston MA",    "phone": "6175550101",  "email": "info@acme.com"},
    {"id": 3,  "company": "Acme Corp.",        "address": "123 Main St Boston MA 02101",  "phone": "(617)555-0101","email": "contact@acme.com"},
    {"id": 4,  "company": "Beta LLC",          "address": "45 Oak Ave, Chicago, IL 60601", "phone": "312-555-0202","email": "hello@beta.io"},
    {"id": 5,  "company": "Beta, LLC",         "address": "45 Oak Avenue Chicago IL",      "phone": "3125550202",  "email": "hello@beta.io"},
    {"id": 6,  "company": "Gamma Inc",         "address": "7 Pine Rd, Seattle, WA 98101", "phone": "206-555-0303","email": "support@gamma.com"},
    {"id": 7,  "company": "GammaInc",          "address": "7 Pine Road Seattle WA",        "phone": "2065550303",  "email": "support@gamma.com"},
    {"id": 8,  "company": "Delta Partners",    "address": "99 River Blvd, Austin TX 78701","phone": "512-555-0404","email": "dp@delta.com"},
    {"id": 9,  "company": "Delta Partners LLC","address": "99 River Blvd Austin, TX",      "phone": "5125550404",  "email": "dp@delta.com"},
    {"id": 10, "company": "Epsilon Co",        "address": "200 Lake Dr, Miami FL 33101",   "phone": "305-555-0505","email": "info@epsilon.net"},
]

# Ground truth duplicate clusters (sets of ids)
TASK2_GOLD_CLUSTERS = [
    {1, 2, 3},   # Acme
    {4, 5},      # Beta
    {6, 7},      # Gamma
    {8, 9},      # Delta
    {10},        # Epsilon — singleton
]

# Gold merged records (canonical)
TASK2_GOLD_MERGED = [
    {"company": "Acme Corp",     "address": "123 Main St, Boston, MA 02101", "phone": "617-555-0101", "email": "info@acme.com"},
    {"company": "Beta LLC",      "address": "45 Oak Ave, Chicago, IL 60601", "phone": "312-555-0202", "email": "hello@beta.io"},
    {"company": "Gamma Inc",     "address": "7 Pine Rd, Seattle, WA 98101", "phone": "206-555-0303", "email": "support@gamma.com"},
    {"company": "Delta Partners","address": "99 River Blvd, Austin TX 78701","phone": "512-555-0404","email": "dp@delta.com"},
    {"company": "Epsilon Co",    "address": "200 Lake Dr, Miami FL 33101",   "phone": "305-555-0505","email": "info@epsilon.net"},
]


def _normalize_phone(p: str) -> str:
    return "".join(c for c in p if c.isdigit())


def grade_task2(payload: Dict[str, Any]) -> Tuple[float, str]:
    """
    Expected payload keys:
      clusters: list of lists of record ids  e.g. [[1,2,3],[4,5],...]
      merged:   list of canonical dicts      e.g. [{"company":...},...]
    """
    clusters_agent = payload.get("clusters", [])
    merged_agent = payload.get("merged", [])

    # --- Cluster score (precision/recall via exact set match) ---
    gold_sets = [frozenset(c) for c in TASK2_GOLD_CLUSTERS]
    agent_sets = [frozenset(c) for c in clusters_agent]

    cluster_hits = sum(1 for gs in gold_sets if gs in agent_sets)
    cluster_score = cluster_hits / len(gold_sets)

    # --- Merge score: per-field match for matched clusters ---
    merge_hits = 0
    merge_total = len(TASK2_GOLD_MERGED) * 4  # 4 fields each
    for gm in TASK2_GOLD_MERGED:
        # Find best match in agent merged list by company name similarity
        best = None
        best_sim = 0.0
        for am in merged_agent:
            s = _company_sim(str(am.get("company", "")), gm["company"])
            if s > best_sim:
                best_sim = s
                best = am
        if best and best_sim > 0.5:
            for f in ["address", "email"]:
                if str(best.get(f, "")).strip().lower() == gm[f].strip().lower():
                    merge_hits += 1
            if _normalize_phone(str(best.get("phone", ""))) == _normalize_phone(gm["phone"]):
                merge_hits += 1
            # company name partial credit
            if best_sim > 0.8:
                merge_hits += 1

    merge_score = merge_hits / merge_total if merge_total else 0.0
    score = round(0.5 * cluster_score + 0.5 * merge_score, 4)
    feedback = (
        f"Task2 score={score:.2f} | cluster_score={cluster_score:.2f} "
        f"[{cluster_hits}/{len(gold_sets)}] | merge_score={merge_score:.2f} "
        f"[{merge_hits}/{merge_total} field-matches]"
    )
    return score, feedback


def _company_sim(a: str, b: str) -> float:
    """Simple token overlap similarity."""
    ta = set(a.lower().replace(",", "").replace(".", "").split())
    tb = set(b.lower().replace(",", "").replace(".", "").split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# TASK 3 — Hard: Multi-Source Reconciliation
# ---------------------------------------------------------------------------

TASK3_SOURCES = {
    "crm": [
        {"entity_id": "E001", "name": "Alpha Solutions", "phone": "212-555-1001", "email": "sales@alpha.com",  "tier": "gold",   "revenue": 500000},
        {"entity_id": "E002", "name": "BetaTech",        "phone": "415-555-2002", "email": "info@betatech.io","tier": "silver", "revenue": 120000},
        {"entity_id": "E003", "name": "Gamma Global",    "phone": "312-555-3003", "email": None,              "tier": "bronze", "revenue": 45000},
    ],
    "billing": [
        {"entity_id": "E001", "name": "Alpha Solutions Ltd", "phone": "212-555-1001", "email": "billing@alpha.com", "tier": None,     "revenue": 512000},
        {"entity_id": "E002", "name": "BetaTech Inc",        "phone": "415-555-9999", "email": "info@betatech.io", "tier": "gold",   "revenue": 125000},
        {"entity_id": "E003", "name": "Gamma Global",        "phone": "312-555-3003", "email": "admin@gamma.com",  "tier": "bronze", "revenue": 46000},
    ],
    "marketing": [
        {"entity_id": "E001", "name": "Alpha Solutions", "phone": None,           "email": "marketing@alpha.com", "tier": "gold",   "revenue": None},
        {"entity_id": "E002", "name": "BetaTech",        "phone": "415-555-2002", "email": "info@betatech.io",   "tier": "silver", "revenue": 119000},
        {"entity_id": "E003", "name": "Gamma Global",    "phone": None,           "email": "admin@gamma.com",    "tier": "silver", "revenue": 44000},
    ],
}

TASK3_POLICY = """
Source priority policy (apply field-by-field):
  name:    crm > billing > marketing (prefer shorter, cleaner name)
  phone:   crm > marketing > billing (billing phone for E002 is known bad)
  email:   crm has sales email; billing has billing email; use crm unless null → then billing → then marketing
  tier:    crm > marketing > billing  (crm is authoritative for tier)
  revenue: billing > crm > marketing  (billing has latest invoice data)
  rule:    Never set a field to null if a non-null value exists in any source.
"""

TASK3_GOLD = [
    {"entity_id": "E001", "name": "Alpha Solutions",  "phone": "212-555-1001", "email": "sales@alpha.com",  "tier": "gold",   "revenue": 512000},
    {"entity_id": "E002", "name": "BetaTech",         "phone": "415-555-2002", "email": "info@betatech.io", "tier": "silver", "revenue": 125000},
    {"entity_id": "E003", "name": "Gamma Global",     "phone": "312-555-3003", "email": "admin@gamma.com",  "tier": "bronze", "revenue": 46000},
]


def grade_task3(payload: Dict[str, Any]) -> Tuple[float, str]:
    """
    Expected payload: {"records": [{"entity_id":..., "name":..., ...}, ...]}
    Grader checks field accuracy vs gold and penalizes null-when-value-exists.
    """
    records = payload.get("records", [])
    gold_map = {r["entity_id"]: r for r in TASK3_GOLD}
    fields = ["name", "phone", "email", "tier", "revenue"]
    total = len(TASK3_GOLD) * len(fields)
    correct = 0
    penalties = 0
    feedback_parts = []

    for agent_rec in records:
        eid = agent_rec.get("entity_id")
        gold_rec = gold_map.get(eid)
        if not gold_rec:
            continue
        for f in fields:
            av = agent_rec.get(f)
            gv = gold_rec[f]
            try:
                if isinstance(gv, int):
                    match = int(av) == gv
                else:
                    match = str(av).strip().lower() == str(gv).strip().lower()
            except Exception:
                match = False
            if match:
                correct += 1
            else:
                # Penalty: agent set null when gold is not null
                if av is None and gv is not None:
                    penalties += 1
                feedback_parts.append(f"✗ {eid}.{f} got={av!r} expected={gv!r}")

    base_score = correct / total if total else 0.0
    penalty_deduction = min(penalties * 0.05, 0.20)
    score = round(max(0.0, base_score - penalty_deduction), 4)
    feedback = (
        f"Task3 score={score:.2f} | field_accuracy={base_score:.2f} "
        f"[{correct}/{total}] | null_penalties={penalties} (-{penalty_deduction:.2f})\n"
        + "\n".join(feedback_parts[:10])
    )
    return score, feedback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASKS = {
    1: {
        "name": "field-extraction",
        "instruction": (
            "Extract and normalize the following raw JSON product record into the target schema. "
            "Apply: trim whitespace, lowercase category/brand, convert price→float, "
            "quantity→int, discount_pct→int, date→YYYY-MM-DD."
        ),
        "input_data": TASK1_INPUT,
        "schema_hint": TASK1_SCHEMA,
        "grader": grade_task1,
        "max_steps": 3,
    },
    2: {
        "name": "dedup-merge",
        "instruction": (
            "You are given 10 company records that contain duplicates. "
            "Identify duplicate clusters (by shared company identity) and produce one merged "
            "canonical record per cluster. Return:\n"
            "  clusters: list of lists of record ids\n"
            "  merged:   list of canonical records (company, address, phone, email)"
        ),
        "input_data": TASK2_RECORDS,
        "schema_hint": None,
        "grader": grade_task2,
        "max_steps": 5,
    },
    3: {
        "name": "multi-source-reconciliation",
        "instruction": (
            "Three source systems (crm, billing, marketing) contain conflicting records "
            "for the same entities. Apply the provided source-priority policy field-by-field "
            "to produce a golden master record for each entity_id. "
            "Never set a field to null when a non-null value exists in any source."
        ),
        "input_data": {**TASK3_SOURCES, "policy": TASK3_POLICY},
        "schema_hint": None,
        "grader": grade_task3,
        "max_steps": 6,
    },
}
