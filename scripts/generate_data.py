"""Generate the synthetic public-finance dataset. Everything here is fictional.

Outputs (under data/):
  policies/*.md         budget circulars and expense policies (RAG corpus)
  claims.csv            labelled expense claims (classification / fine-tuning)
  spend_monthly.csv     monthly spend, budget and commitments per department (feature store + overrun model)

Run: python scripts/generate_data.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
SEED = 42

POLICIES: dict[str, str] = {
    "travel-policy": """# Official Travel Policy (Circular FIN-07)

## Eligibility
Business travel must be pre-approved by the department director before tickets are booked.
Travel for training requires an approved training request in addition to travel approval.

## Air travel
Employees at grade 1 to 3 may travel business class on flights longer than 4 hours.
All other employees travel economy class. Upgrades at personal cost are allowed but not reimbursed.

## Per diem
The daily allowance is AED 600 for GCC destinations and AED 900 for all other destinations.
Per diem covers meals and local transport; receipts are not required for per diem.

## Hotels
Hotel costs are reimbursed up to AED 1,200 per night with an itemised invoice.
""",
    "expense-claims": """# Expense Claims Procedure (Circular FIN-11)

## Submission deadline
Claims must be submitted within 30 days of the expense date. Late claims require director approval.

## Receipts
An itemised receipt or tax invoice is mandatory for every expense above AED 100.
Credit-card statements alone are not accepted as proof of payment.

## Approval levels
Claims up to AED 5,000 are approved by the line manager.
Claims above AED 5,000 and up to AED 25,000 require the department director.
Claims above AED 25,000 require the finance director.

## Payment
Approved claims are paid with the next monthly payroll cycle.
""",
    "hospitality": """# Hospitality and Entertainment (Circular FIN-14)

## Allowed hospitality
Business meals with external stakeholders are allowed up to AED 250 per attendee.
Alcohol is never reimbursed.

## Gifts
Gifts to external parties must not exceed AED 500 in value and must be logged in the gifts register.

## Internal events
Team events require a budget line approved in advance; the cap is AED 150 per employee per event.
""",
    "budget-management": """# Budget Management Guidelines (Circular BUD-02)

## Budget transfers
Transfers between budget lines within a department up to AED 200,000 are approved by the department director.
Transfers above AED 200,000 or between departments require approval from the Budget Committee.

## Commitments
Purchase orders create commitments that reduce available budget immediately, before invoices are paid.

## Year-end
Unspent operating budget does not roll over to the next fiscal year.
Departments forecasting an overrun must submit a variance report by the end of the third quarter.

## Monitoring
Finance reviews burn rate monthly. A department spending more than 110 percent of its pro-rata budget is flagged.
""",
    "it-procurement": """# IT Equipment Procurement (Circular PRC-05)

## Standard equipment
Laptops, monitors and peripherals must be requested through the IT catalogue, not bought directly.
Direct purchases of IT equipment are reimbursed only with prior written approval from IT.

## Software
Software subscriptions require an information security review before purchase.
""",
}

CATEGORIES: dict[str, list[str]] = {
    "travel": [
        "flight to {city} for the {event}",
        "hotel stay in {city}, {n} nights",
        "taxi from airport in {city}",
        "train ticket to {city} for site visit",
        "visa fee for {city} trip",
    ],
    "training": [
        "registration fee for {course} course",
        "exam voucher for {course} certification",
        "online {course} training subscription",
        "workshop on {course} in {city}",
    ],
    "it_equipment": [
        "USB-C docking station",
        "external monitor for home office",
        "replacement laptop charger",
        "noise-cancelling headset for calls",
        "wireless keyboard and mouse",
    ],
    "office_supplies": [
        "printer paper and toner",
        "stationery for the team",
        "whiteboard markers and notebooks",
        "filing folders for archive",
        "desk organisers",
    ],
    "hospitality": [
        "lunch with {org} delegation",
        "coffee meeting with {org} officials",
        "dinner for {org} visitors",
        "refreshments for stakeholder workshop",
        "team event catering",
    ],
    "professional_services": [
        "translation of {org} agreement",
        "legal review of contract with {org}",
        "consultancy invoice for {course} assessment",
        "notary and attestation fees",
    ],
}
FILLERS = {
    "city": ["Riyadh", "London", "Singapore", "Cairo", "Muscat", "Paris"],
    "event": ["budget forum", "IMF workshop", "audit conference", "procurement summit"],
    "course": ["IFRS", "data analytics", "project management", "public procurement", "Python"],
    "org": ["ministry", "partner bank", "audit office", "municipality"],
    "n": ["2", "3", "4"],
}
# Ambiguous phrases appear in more than one category, so classification is not trivially keyword-based.
AMBIGUOUS = {
    "travel": ["meals during {city} trip"],
    "hospitality": ["meal with visiting {org} team in {city}"],
    "training": ["travel and fees for {course} course"],
}

DEPARTMENTS = [
    "revenue",
    "treasury",
    "procurement",
    "audit",
    "budget_planning",
    "it",
    "hr",
    "legal",
    "strategy",
    "facilities",
    "economic_policy",
    "investments",
]


def fill(template: str, rng: random.Random) -> str:
    return template.format(**{k: rng.choice(v) for k, v in FILLERS.items()})


def write_policies() -> None:
    (ROOT / "policies").mkdir(parents=True, exist_ok=True)
    for name, text in POLICIES.items():
        (ROOT / "policies" / f"{name}.md").write_text(text)


# Phrasings that never appear in training: the test set mixes these with familiar ones, so the
# benchmark measures generalisation to new wording, which is where LLMs should beat TF-IDF.
UNSEEN: dict[str, list[str]] = {
    "travel": [
        "airfare {city} return",
        "accommodation near {event} venue in {city}",
        "ride-hailing during {city} visit",
        "excess baggage charge, {city}",
    ],
    "training": [
        "tuition for {course} programme",
        "seat at {course} bootcamp",
        "renewal of {course} credential",
        "e-learning licence: {course}",
    ],
    "it_equipment": [
        "HDMI cable and USB adapter",
        "webcam for video meetings",
        "portable SSD for backups",
        "laptop stand and cooling pad",
    ],
    "office_supplies": [
        "sticky notes and pens",
        "binders and labels",
        "envelopes for official letters",
        "ink cartridges for the office printer",
    ],
    "hospitality": [
        "working lunch for {org} visitors",
        "tea and snacks for {org} meeting",
        "banquet for {org} guests",
        "welcome breakfast for new joiners",
    ],
    "professional_services": [
        "audit advisory fees",
        "interpreter for {org} meeting",
        "document attestation service",
        "actuarial review for {org}",
    ],
}
LABEL_NOISE = 0.04  # historical claims were labelled by people; a few are wrong


def make_claim(category: str, templates: list[str], rng: random.Random) -> str:
    text = fill(rng.choice(templates), rng)
    if rng.random() < 0.3:
        text = text.capitalize() + rng.choice([" - urgent", " (q3)", ", see attached receipt", ""])
    return text


def write_claims(rng: random.Random, n_train: int = 700, n_valid: int = 100, n_test: int = 200) -> None:
    rows = []
    cats = list(CATEGORIES)
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        for _ in range(n):
            category = rng.choice(cats)
            seen = CATEGORIES[category] + AMBIGUOUS.get(category, [])
            templates = UNSEEN[category] if split == "test" and rng.random() < 0.5 else seen
            label = category
            if split != "test" and rng.random() < LABEL_NOISE:
                label = rng.choice([c for c in cats if c != category])
            rows.append(
                {
                    "claim_id": f"C-{len(rows):05d}",
                    "description": make_claim(category, templates, rng),
                    "amount_aed": round(rng.uniform(40, 9000), 2),
                    "category": label,
                    "split": split,
                }
            )
    with (ROOT / "claims.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_spend(rng: random.Random) -> None:
    """36 months (FY2023-FY2025) of spend per department. FY2025 shifts upward to create drift."""
    rows = []
    for dept in DEPARTMENTS:
        annual_budget = rng.choice([4, 6, 9, 12, 18]) * 1_000_000
        overspender = rng.random() < 0.35
        for year in (2023, 2024, 2025):
            shift = 1.05 if year == 2025 else 1.0  # FY2025: inflation -> mild spend shift
            pace = rng.uniform(0.85, 1.0) if not overspender else rng.uniform(1.0, 1.2)
            if rng.random() < 0.25:  # departments change behaviour some years
                pace = rng.uniform(0.85, 1.2)
            for month in range(1, 13):
                seasonal = 1.25 if month in (11, 12) else (0.8 if month in (7, 8) else 1.0)
                spend = annual_budget / 12 * pace * seasonal * shift * rng.uniform(0.85, 1.15)
                commitments = annual_budget * rng.uniform(0.02, 0.12) * (1.4 if month >= 9 else 1.0)
                if year == 2025:  # new capital programmes committed up front -> distribution shift
                    commitments *= 1.8
                rows.append(
                    {
                        "department": dept,
                        "year": year,
                        "month": month,
                        "annual_budget_aed": annual_budget,
                        "spend_aed": round(spend, 2),
                        "open_commitments_aed": round(commitments, 2),
                    }
                )
    with (ROOT / "spend_monthly.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rng = random.Random(SEED)
    write_policies()
    write_claims(rng)
    write_spend(rng)
    print(f"Wrote synthetic data to {ROOT}")
