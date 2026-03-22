from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AGENT_API_KEY     = os.environ.get("AGENT_API_KEY", "")
MODEL             = "claude-sonnet-4-6"
MAX_TOKENS        = int(os.environ.get("MAX_TOKENS", "32000"))
MAX_AGENT_TURNS   = int(os.environ.get("MAX_AGENT_TURNS", "50"))

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class FileAttachment(BaseModel):
    filename: str = ""
    content_base64: str = ""   # competition field name
    content: str = ""          # fallback
    mime_type: str = ""

    def get_data(self) -> str:
        return self.content_base64 or self.content


class SolveRequest(BaseModel):
    prompt: str
    files: list[FileAttachment] = []
    tripletex_credentials: TripletexCredentials


class SolveResponse(BaseModel):
    status: str = "completed"

# ─────────────────────────────────────────────────────────────────────────────
# Tripletex REST client
# ─────────────────────────────────────────────────────────────────────────────

class TripletexClient:
    def __init__(self, base_url: str, session_token: str) -> None:
        base = base_url.rstrip("/")
        if base.endswith("/v2"):
            base = base[:-3]
        self.base_url = base
        self.auth = httpx.BasicAuth("0", session_token)

    async def call(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if method.upper() == "GET":
            params = {"from": 0, "count": 100, **(params or {})}
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        log.info("→ %s %s  params=%s  body=%s", method.upper(), url, params, body)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.request(
                method=method.upper(),
                url=url,
                auth=self.auth,
                params=params,
                json=body,
                headers=headers,
            )
        log.info("← %d", resp.status_code)
        try:
            data: Any = resp.json()
        except Exception:
            data = {"raw_text": resp.text}
        return {"status_code": resp.status_code, "ok": resp.is_success, "data": data}

# ─────────────────────────────────────────────────────────────────────────────
# Tool definition
# ─────────────────────────────────────────────────────────────────────────────

TRIPLETEX_TOOL: anthropic.types.ToolParam = {
    "name": "tripletex_api_call",
    "description": (
        "Make an authenticated call to the Tripletex v2 REST API.\n\n"
        "ENDPOINTS:\n"
        "  GET/POST/PUT  /v2/employee                  — employees\n"
        "  POST          /v2/employee/employment        — create employment (startDate)\n"
        "  GET           /v2/employee/employment/{id}   — get employment record\n"
        "  GET/POST/PUT  /v2/customer                  — customers\n"
        "  GET/POST      /v2/product                   — products\n"
        "  GET/POST      /v2/order                     — orders\n"
        "  GET/POST      /v2/order/orderline            — order lines\n"
        "  GET/POST      /v2/invoice                   — invoices\n"
        "  POST          /v2/invoice/{id}/payment       — register payment\n"
        "  PUT           /v2/ledger/voucher/{id}/reverse?date=YYYY-MM-DD  — reverse payment\n"
        "  GET/POST/PUT/DELETE /v2/travelExpense        — travel expenses\n"
        "  GET/POST      /v2/project                   — projects\n"
        "  GET/POST      /v2/department                — departments\n"
        "  GET           /v2/ledger/account             — chart of accounts\n"
        "  GET           /v2/ledger/voucher             — vouchers\n"
        "  GET/POST      /v2/bank                      — bank accounts\n"
        "  GET/POST      /v2/bank/statement             — bank statements\n"
        "  POST          /v2/bank/statement/{id}/transaction — add bank transaction to statement\n"
        "  GET/POST      /v2/bank/reconciliation        — reconciliation periods\n"
        "  GET/POST      /v2/bank/reconciliation/match  — match bank transactions to postings\n"
        "  GET           /v2/ledger/posting             — ledger postings\n"
        "  GET/POST      /v2/supplierInvoice            — supplier invoices\n\n"
        "CRITICAL EMPLOYEE RULES:\n"
        "  - userType: 2  assigns Administrator role (worth 5/10 points in scoring)\n"
        "  - department: {id} is required — always GET /v2/department first\n"
        "  - dateOfBirth is required for employment creation\n"
        "  - Employment start date: POST /v2/employee/employment separately after creating employee\n"
        "    Body: {employee: {id}, startDate: YYYY-MM-DD, isMainEmployer: true}\n"
        "  - Do NOT put employments array in the POST /v2/employee body — it is silently ignored\n\n"
        "CRITICAL CUSTOMER RULES:\n"
        "  - Always include isCustomer: true in POST /v2/customer body\n\n"
        "RESPONSES:\n"
        "  List:   {fullResultSize: N, values: [...]}\n"
        "  Single: {value: {..., id: N}}\n"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "description": "HTTP method. POST/PUT/DELETE/PATCH count against efficiency score. GET is free.",
            },
            "path": {
                "type": "string",
                "description": "API path e.g. /v2/employee or /v2/invoice/42",
            },
            "params": {
                "type": "object",
                "description": "Query parameters e.g. {name: 'Acme'} or {fields: 'id,name'}",
            },
            "body": {
                "type": "object",
                "description": "JSON body for POST, PUT, PATCH.",
            },
        },
        "required": ["method", "path"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — written by the user, kept exactly as-is
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Tripletex — AI Accounting Agent
Your role: You are an AI agent that completes accounting tasks in Tripletex. You receive a task prompt (in one of 7 languages), use the Tripletex API to execute it, and get scored on correctness and efficiency. The goal is to score as many Points as posible and written further Down is how you to so and what is important. Read through the rest of this to understand the context further. You dont need to worry about the HTTPS, sandbox or other technical stuff. Your job is to Complete the tasks given in Tripletex and gain as many Points as posible. Here is the rest of the context:


How It Works

Submit your HTTPS endpoint URL on the platform
We provision a fresh Tripletex sandbox account
We send a randomly selected accounting task to your /solve endpoint
Your agent reads the prompt, optionally processes attached files (PDFs, images)
Your agent calls the Tripletex API via a proxy to complete the task
We verify the result field-by-field against expected values
Your score updates on the rolling leaderboard

Each submission gets a brand new Tripletex account — you always start from scratch.
Key Facts



Task types
30 different accounting tasks


Variants
56 per task (7 languages × 8 data sets)


Language
Prompts in Norwegian, English, Spanish, Portuguese, Nynorsk, German, French


Timeout
5 minutes per submission


API
Tripletex v2 REST API via authenticated proxy


Scoring
Field-by-field checks + efficiency bonus, best score per task kept


Score range
0.0 (failed) — up to 6.0 (perfect Tier 3 + best efficiency)


Files
Some tasks include PDF or image attachments



Task Categories
Your agent will encounter tasks like:

Employees — Create employees, set roles, update contact info
Customers & Products — Register customers, create products
Invoicing — Create invoices, register payments, issue credit notes
Travel Expenses — Register or delete travel expense reports
Projects — Create projects linked to customers
Corrections — Delete or reverse incorrect entries
Departments — Create departments, enable accounting modules

Tasks range from simple single-API-call operations to multi-step workflows requiring several resources to be created and linked together.

Tripletex — Scoring (THIS IS VERY IMPORTANT)
Field-by-Field Verification (Correctness)
After your agent responds, we query the Tripletex API to verify what was created or modified. Each task has specific checks worth different point values.
Example for a "Create employee" task (max 10 points):

Check: Employee found — 2 points
Check: Correct first name — 1 point
Check: Correct last name — 1 point
Check: Correct email — 1 point
Check: Administrator role assigned — 5 points

The raw score is normalized to 0–1: correctness = points_earned / max_points (e.g., 8/10 = 0.8).

Tier Multiplier:
Tier 1 ×1 — Create employee, create customer
Tier 2 ×2 — Create invoice, register payment
Tier 3 ×3 — Complex multi-step workflows

So a perfect score on a Tier 2 task = 1.0 × 2 = 2.0 base score.

Efficiency Bonus
If your agent achieves a perfect correctness score (1.0), you receive an efficiency bonus that can up to double your tier score.
Two factors determine the bonus:
Call efficiency — How many write calls (POST, PUT, DELETE, PATCH) did your agent make compared to the best known solution for this task? Fewer calls = higher bonus. GET requests are not counted — read as much as you need to understand the data.
Error cleanliness — How many of your write calls resulted in 4xx errors (400, 404, 422, etc.)? Errors reduce the bonus. An agent that gets it right without trial-and-error is rewarded.

Scenario (Tier 2 task) — Score:
Failed all checks — 0.0
80% of checks passed — 1.6
Perfect, but many errors and extra calls — ~2.1
Perfect, efficient, a few errors — ~2.6
Perfect, best-in-class efficiency, zero errors — 4.0

The efficiency bonus only applies to perfect submissions. Non-perfect submissions score correctness × tier.
Efficiency benchmarks are recalculated periodically. As teams find more efficient solutions, the bar rises for everyone. Your best score per task is recalculated against current benchmarks every 12 hours. Normalization only affects the efficiency bonus — your correctness score never decreases.

Optimizing for Efficiency:
Plan before calling — Parse the prompt fully before making API calls. Understand what needs to be created/modified before starting
Avoid trial-and-error — Every 4xx error (400, 404, 422) reduces your efficiency bonus. Validate inputs before sending
Minimize GET calls — Don't fetch entities you don't need. If you created something, you already know its ID from the response
Read error messages — If a call fails, the Tripletex error message tells you exactly what's wrong. Fix it in one retry, not several

DISCOVERED API RULES (learned from testing — follow these exactly):

EMPLOYEE CREATION (two steps):
Step 1 — POST /v2/employee:
  { firstName, lastName, email, userType: 2, department: {id}, dateOfBirth: "YYYY-MM-DD" }
  - userType 2 = Administrator role (worth 5 out of 10 points in scoring — do not skip)
  - department.id is required — GET /v2/department first to find it
  - dateOfBirth is required — use value from task or default "1990-01-01"
  - Save the returned employee id from response.data.value.id
Step 2 — POST /v2/employee/employment:
  { employee: {id: <from step 1>}, startDate: "YYYY-MM-DD", isMainEmployer: true }
  - This is the ONLY way to set employment start date
  - Do NOT put "employments" array in the POST /v2/employee body — it is silently ignored

CUSTOMER CREATION:
POST /v2/customer: { name, isCustomer: true, email }
- isCustomer: true is required

ORDER LINE CREATION:
POST /v2/order/orderline: { order: {id}, description, unitPriceExcludingVatCurrency, count: 1, vatType: {id} }
- Field is `count` (NOT `quantity` → 422)
- vatType id 0 = no VAT / ingen avgift

COMMON TASK PATTERNS:
Create employee: GET /v2/department → POST /v2/employee (userType:2) → POST /v2/employee/employment
Create customer: POST /v2/customer (isCustomer:true)
Create order: POST /v2/order {customer:{id}, orderDate:"YYYY-MM-DD", deliveryDate:"YYYY-MM-DD"}
  - deliveryDate is REQUIRED → 422 without it
  - Do NOT include isOffer field → 422
  - Optionally link to project: project:{id}
Create invoice: GET customer → POST /v2/order (with deliveryDate) → POST /v2/order/orderline → PUT /v2/order/{orderId}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD
Register payment on EXISTING invoice: GET /v2/customer (find by name/org) → GET /v2/invoice?customerId=X&invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31 → PUT /v2/invoice/{id}/:payment
  IMPORTANT: If the task says "has an invoice" or "pending invoice", the invoice ALREADY EXISTS — do NOT create a new one! Search for it with the broad date range.
  If no invoice found with broad range, THEN create order→invoice→pay.
Reverse payment: GET /v2/ledger/voucher → PUT /v2/ledger/voucher/{id}/reverse?date=YYYY-MM-DD
Delete travel expense: GET /v2/travelExpense → DELETE /v2/travelExpense/{id}
Create project: GET /v2/employee → POST /v2/project {name, number, startDate, projectManager:{id}, customer:{id}}
  - No "budget" field on project → 422. Budget cannot be set via API, skip it.
Post supplier invoice: GET /v2/supplier (find by org number) → GET /v2/ledger/account?number=7xxx (expense) → GET /v2/ledger/account?number=2400 (AP) → GET /v2/ledger/vatType → POST /v2/supplierInvoice (with voucher+postings, NO amountCurrency)

LEDGER ACCOUNT LOOKUPS — HARD LIMIT:
  NEVER spend more than 2 GET /v2/ledger/account calls searching for a single account.
  After 2 searches, pick the BEST AVAILABLE account from what you already found and proceed immediately.
  Do NOT search 6500, then 6520, then 6540, then 6560, then 6580, then 6600... — this wastes turns and scores 0.
  If account X doesn't exist, use the closest available in the same range (e.g. X-20 or X+20).
  The task scorer checks voucher structure and amounts — one extra search doesn't help if it costs you the timeout.
  Quick account reference for common task items:
    IT equipment / keyboards / computers: try 6860 first, then 6540 (inventar)
    Office supplies / stationery: try 6560 (rekvisita)
    Advertising/marketing: try 6700 first
    Phone/internet: try 6900 first

LEDGER VOUCHER POSTINGS (learned from live testing):
  Posting amounts use field `amountGross` (NOT `amount` — that field stores 0 always).
  Also include `amountGrossCurrency` with the same value as `amountGross` for NOK postings.
  Each posting MUST have a `row` field starting at 1 (not 0). Row 0 is system-reserved → 422.
  Correct posting format: {"row": 1, "account": {"id": X}, "amountGross": 1000.0, "amountGrossCurrency": 1000.0, "description": "..."}
  Voucher date field is `date` (NOT `voucherDate`).

CUSTOM DIMENSIONS:
  The endpoint /v2/customDimension returns 404 in the competition API — this feature is not available.
  If a task asks you to create a custom accounting dimension, immediately respond that it is not supported
  via the API and complete any other parts of the task. Do NOT spend turns searching for alternative paths.

INVOICE SEARCH (critical):
  GET /v2/invoice WITHOUT a date range returns 422 — always include invoiceDateFrom/To.
  Use a BROAD date range: invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31
  This ensures you find invoices regardless of when they were created.
  NEVER use a narrow range like 2024-2025 — the task's invoice may be dated in 2026 or beyond.
  Safe fields param: id,invoiceNumber,amount,amountCurrency,customer,invoiceDate
  (adding 'status', 'amountRemainingCurrency', or 'dueDate' causes 400)
  To search by customer: add &customerId={id} to the params.

INVOICE CREATION (learned from live testing):
  Correct: PUT /v2/order/{orderId}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD  (returns 200)
  Wrong:   POST /v2/invoice with body {order: ...}  → 422
  Wrong:   POST /v2/invoice with body {orders: [...]}  → 422
  invoiceDueDate is REQUIRED — use invoiceDate + 14 days if not specified.
  Do NOT use POST /v2/invoice at all. Always use PUT /v2/order/{id}/:invoice.

INVOICE PREREQUISITES (learned from live testing):
Before creating an invoice, the company must have a bank account on ledger account 1920.
Fresh sandbox accounts have NO bank account — you must set it first or POST /v2/invoice will fail with 422.
  Step 1: GET /v2/ledger/account?number=1920&fields=id,bankAccountNumber
  Step 2: If bankAccountNumber is empty, PUT /v2/ledger/account/{id} with body:
    { id, version, number: 1920, name: "Bankinnskudd", bankAccountNumber: "70010800003",
      bankAccountCountry: {id: 161}, isBankAccount: true, isInvoiceAccount: true,
      vatType: {id: 0}, currency: {id: 1}, requireReconciliation: true, vatLocked: true }
  "70010800003" is a valid Norwegian bank account number (passes MOD11). Do this BEFORE POST /v2/invoice.

INVOICE SEND ENDPOINT:
  Correct: PUT /v2/invoice/{id}/:send?sendType=EMAIL  (sendType as query param, returns 204)
  Wrong:   PUT /v2/invoice/{id}/send  → 404
  Wrong:   POST /v2/invoice/{id}/:send with body  → 400

PROJECT ACTIVITIES (learned from live testing):
  Creating an activity: POST /v2/activity with body {name, activityType: 'PROJECT_GENERAL_ACTIVITY', isProjectActivity: true, isGeneral: false}
    Wrong: omitting activityType → 422
  Linking activity to project: POST /v2/project/projectActivity with body {activity: {id}, project: {id}} → 201
    Wrong: PUT /v2/project/{id} with projectActivities in body → 422 ("API does not support updating project activities")
    Wrong: GET /v2/project/projectActivity → 400
  So the flow is: POST /v2/activity → get activity id → POST /v2/project/projectActivity to link it.

INVOICE PAYMENT ENDPOINT (learned from live testing):
  Correct: PUT /v2/invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=ID&paidAmount=X  (returns 200)
  Wrong:   POST /v2/invoice/{id}/:payment  → 400
  Wrong:   POST /v2/invoice/{id}/payment  → 404
  Wrong:   POST /v2/invoice/{id}/:createPayment  → 404
  Payment types MUST come from GET /v2/invoice/paymentType (NOT /v2/ledger/paymentTypeOut which is for outgoing payments).
  Typical incoming payment type IDs: "Betalt til bank" (use this for bank payments), "Kontant" (for cash).
  Do NOT use paymentTypeId=0 or 1 — use the actual IDs from GET /v2/invoice/paymentType.

EMPLOYEE EMPLOYMENT (learned from live testing):
  Field name is percentageOfFullTimeEquivalent (NOT percentOfFullTimeEquivalent) — wrong name → 422.
  percentageOfFullTimeEquivalent belongs inside employmentDetails items, NOT at the top level of the employment object.
  If initial POST fails due to field errors, POST without that field first (→ 201), then PUT /v2/employee/employment/details/{detailId} to update it.
  Occupation codes: Tripletex uses 7-digit STYRK-98 codes. Contract STYRK codes may be 4-digit STYRK-08.
  If you cannot find an exact match after 2 searches, use the closest match by name and move on — do not keep searching.

TRAVEL EXPENSES (learned from live testing):
  TWO types of travel expense — choose based on whether task mentions per diem (dietas/dagpenger):

  TYPE A — No per diem (just costs like flight, taxi, hotel):
    POST /v2/travelExpense with {employee: {id}, title: "..."} — minimal body only
    Then add costs (see below).

  TYPE B — WITH per diem (task mentions dietas/dagpenger/daily allowance):
    Per diem requires the "reiseregning" module to be enabled AND travel expense type 2.
    ALL 4 steps are required in order — do NOT skip step 1:

    Step 1 — Enable reiseregning module (ALWAYS do this first for per diem tasks):
      GET /v2/company/modules — look for any field with "travel" or "reise" in the name that is false
      PUT /v2/company/modules with {<thatFieldName>: true} to enable it
      (Read the exact field name from the GET response — do NOT guess it)

    Step 2 — Create reiseregning travel expense:
      POST /v2/travelExpense with {employee:{id}, title:"...", type: 2,
        startDate:"YYYY-MM-DD", endDate:"YYYY-MM-DD"}
      - type:2 = reiseregning (only works after module enabled in step 1)
      - startDate/endDate define the trip period — required for day count calculation
      - Confirm response shows type:2 before proceeding

    Step 3 — Find Norway domestic zone for location field:
      GET /v2/travelExpense/zone with fields=id,zoneName,countryCode
      Search for zone with countryCode='NO' or name containing 'Norge'/'Norway'
      Save that zone id

    Step 4 — POST /v2/travelExpense/perDiemCompensation:
      Body: {travelExpense: {id}, location: {id: <zoneId>}, rate: <dailyRate>}
      - location MUST be an object {id: zoneId}, NOT a string like "Norway" or "Oslo" → 422
      - countDays and days are NOT valid fields → 422 (days come from expense startDate/endDate)
      - "Kun reiseregning" error means the expense is still type:1 — go back and fix step 1

    FALLBACK — if per diem endpoint still fails after 4 attempts:
      Stop trying per diem. Add the per diem total as a regular cost:
      POST /v2/travelExpense/cost with perDiemAmount (N days × rate) using a suitable costCategory.
      Then declare done — don't spend more than 5 turns total on per diem.

  Add costs (flight, taxi, etc.) — same for both types:
    POST /v2/travelExpense/cost with {travelExpense:{id}, costCategory:{id}, amountCurrencyIncVat: AMOUNT, paymentType:{id}}
    - Field is amountCurrencyIncVat (NOT amount, NOT amountCurrency)
    - paymentType id MUST come from GET /v2/travelExpense/paymentType (use first result, typically "Privat utlegg")
    - costCategory id comes from GET /v2/travelExpense/costCategory (NO fields param — returns 400)
    - Do NOT include description or currency fields → 422
    - Common cost categories: "Fly" for flight, "Taxi" for taxi, "Hotell" for hotel

  TIMESHEET entries: POST /v2/timesheet/entry {date, employee:{id}, project:{id}, activity:{id}, hours, chargeableHours}
    - date MUST be within the project's start/end date range → 422 if outside

SALARY TRANSACTIONS (learned from live testing):
  CRITICAL: POST /v2/salary/specification ALWAYS returns 500 in the competition sandbox — it is BROKEN. Do NOT try it.
  Adding specifications inline to POST /v2/salary/transaction also fails with 422 "Arbeidsforholdet er ikke knyttet mot en virksomhet".
  The ONLY working pattern is:
    Step 1: Find employee: GET /v2/employee?email=...
    Step 2: Ensure employment exists: GET /v2/employee/employment?employeeId={id}
      - If no employment, POST /v2/employee/employment {employee:{id}, startDate:"2025-01-01", isMainEmployer:true}
      - If that fails with 422, first PUT /v2/employee/{id} to add dateOfBirth:"1990-01-01", then retry
    Step 3: POST /v2/salary/transaction with MINIMAL body only:
      { date: "YYYY-MM-DD", year: YYYY, month: MM, payslips: [{ employee: {id: <employeeId>} }] }
    Step 4: STOP. Do NOT try to add salary specifications or amounts — the endpoint is broken.
  The salary transaction itself (201) is what the scorer checks. Accept that amounts cannot be set and move on.
  Do NOT spend more than 5 turns on salary tasks. After a 201 on salary/transaction, you are done.

SUPPLIER CREATION (learned from live testing):
  POST /v2/supplier with MINIMAL body only: {name, organizationNumber, isSupplier: true}
  Do NOT include address, bankAccountNumber, or other fields → 422.
  If the task mentions creating a supplier for a specific company, just use name + organizationNumber + isSupplier:true.

RECEIPT vs INVOICE — CRITICAL DISTINCTION:
  "Recibo/kvittering/receipt" from a store = CASH PURCHASE paid immediately.
    → POST /v2/ledger/voucher with credit to account 1920 (bank/cash), NOT 2400 (AP).
    → Do NOT use POST /v2/supplierInvoice for receipts.

    CRITICAL VAT RULE for POST /v2/ledger/voucher:
    vatType on a posting row does NOT auto-generate the 2710 row — this is DIFFERENT from supplierInvoice.
    For direct voucher posts you MUST include ALL rows explicitly including the 2710 VAT row.
    The 3-row structure (mandatory — 2 rows will return 422):
      Row 1: Debit expense account — net amount (excl. VAT), with vatType: {id: 1} (label only)
      Row 2: Debit account 2710 — VAT amount (net * 0.25 for 25% VAT)
      Row 3: Credit account 1920 — total incl VAT (negative), amountGross: -totalAmount
    Sum check: net + VAT - total = 0 must hold exactly.

    Add department: {id} to the expense posting row (row 1) to assign the correct department.
    Example for 6900 NOK incl 25% VAT, net 5520, VAT 1380:
      postings: [
        {"row": 1, "account": {"id": <expenseAcctId>}, "amountGross": 5520, "amountGrossCurrency": 5520, "vatType": {"id": 1}, "department": {"id": <deptId>}, "description": "..."},
        {"row": 2, "account": {"id": <2710AcctId>}, "amountGross": 1380, "amountGrossCurrency": 1380, "description": "Inngående MVA 25%"},
        {"row": 3, "account": {"id": <1920AcctId>}, "amountGross": -6900, "amountGrossCurrency": -6900, "description": "..."}
      ]

    For account lookup on receipts: start with 6860 (IT/equipment), then 6540 (inventar), then 6560 (rekvisita).
    After 2 account searches, use the best available — do NOT search more than 2 times.

  "Faktura/invoice" from a supplier = CREDIT PURCHASE (pay later).
    → POST /v2/supplierInvoice with credit to account 2400 (AP). See SUPPLIER INVOICE CREATION below.
    NOTE: For supplierInvoice, vatType DOES auto-generate the 2710 row — use only 2 posting rows there.

SUPPLIER INVOICE CREATION (critical — learned from live testing):
  POST /v2/supplierInvoice returns 500 if you include `amountCurrency` — it is READ-ONLY, calculated from postings.
  Do NOT include amountCurrency, amountNoVat, amountCurrencyIncVat in the body — all cause 422 or 500.
  Correct body format (ONLY this works):
  {
    "invoiceNumber": "INV-...",
    "invoiceDate": "YYYY-MM-DD",
    "supplier": {"id": X},
    "voucher": {
      "date": "YYYY-MM-DD",
      "description": "...",
      "postings": [
        {"row": 1, "account": {"id": <expenseAccountId>}, "amountGross": <netAmount>, "amountGrossCurrency": <netAmount>, "vatType": {"id": <vatTypeId>}, "description": "..."},
        {"row": 2, "account": {"id": <2400AccountId>}, "amountGross": <-totalAmount>, "amountGrossCurrency": <-totalAmount>, "supplier": {"id": X}, "description": "..."}
      ]
    }
  }
  Account 2400 = Leverandørgjeld (Accounts Payable). Get its id from GET /v2/ledger/account?number=2400.
  The accounts payable posting MUST include supplier: {id} or you get 422.
  Let the VAT amount come from the expense posting's vatType — do NOT add a manual VAT row unless also posting to 2710 manually.
  If task includes VAT: set amountGross on expense row to net amount (excl. VAT) and set vatType.
  The system auto-generates the input VAT posting (account 2710) from the vatType on the expense row.
  Total flow: GET /v2/supplier (or POST if not found) → GET ledger accounts (7xxx for expense, 2400 for AP) → GET vatType → POST /v2/supplierInvoice with voucher+postings in one call.

BANK RECONCILIATION TASKS (critical — registering invoice payments alone scores ~2/10):
  Bank reconciliation requires the dedicated bank reconciliation API, NOT just PUT /v2/invoice/:payment.
  The scorer checks the bank reconciliation module specifically.

  CORRECT FLOW for bank reconciliation tasks:
  Step 1: GET /v2/bank — find the bank account (bankAccountId)
    Response has values[].id (bankAccount id) and values[].accountId (ledger account id)
  Step 2: POST /v2/bank/statement — create a bank statement from the CSV data
    Body: { "bankAccount": {"id": <bankAccountId>}, "startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD",
            "openingBalance": <amount>, "closingBalance": <amount>,
            "accountBalanceAfter": <amount> }
    Returns statement id.
  Step 3: POST /v2/bank/statement/{statementId}/transaction — add each CSV row as a bank transaction
    Body: { "bankStatement": {"id": <statementId>}, "date": "YYYY-MM-DD",
            "amount": <positive for credit/inn, negative for debit/ut>, "description": "..." }
  Step 4: GET /v2/bank/reconciliation — check for existing reconciliation period
    Or POST /v2/bank/reconciliation to create one:
    Body: { "bankAccount": {"id": <bankAccountId>}, "closingDate": "YYYY-MM-DD",
            "closingBalance": <amount> }
  Step 5: POST /v2/bank/reconciliation/match — match each bank transaction to a ledger posting
    Body: { "bankReconciliation": {"id": <reconciliationId>},
            "bankTransaction": {"id": <bankTxId>},
            "postings": [{"id": <postingId>}] }
    Postings come from GET /v2/ledger/posting or GET /v2/bank/reconciliation/match?bankReconciliationId=X

  NOTE: Still register invoice payments (PUT /v2/invoice/:payment) for the incoming payments —
  this ensures the accounting entries exist to match against.

  INVOICE GET — valid fields only (these cause 400 if included: status, amountRemainingCurrency, dueDate in combination with invoiceDateFrom/To):
  Safe fields: id, invoiceNumber, amount, amountCurrency, customer, invoiceDate
  Do NOT include status or amountRemainingCurrency in fields param.

  CSV files are decoded and included in your context as plain text above. Read the CSV content directly.
  Do NOT spend more than 3 turns searching for data that doesn't exist in an empty account.

LEDGER CORRECTION TASKS (find and fix errors in the general ledger):
  The task will tell you exactly which errors to fix (wrong account, duplicate, missing VAT, wrong amount).
  EFFICIENT APPROACH — do NOT fetch individual vouchers one-by-one.

  Step 1 — Find the error vouchers (1 turn, parallel GETs):
    GET /v2/ledger/posting with account=XXXX&dateFrom=2026-01-01&dateTo=2026-02-28&fields=id,date,description,account,amountGross,amountGrossCurrency,voucher,vatType
    Run one GET per account mentioned in the task (6500, 6540, 7300). The posting response includes voucher.id.
    You already have all the data you need from the postings — do NOT then fetch each voucher individually.

  Step 2 — Fix each error:
    Wrong account (e.g. 6500 used instead of 6540):
      → Reverse the erroneous voucher: PUT /v2/ledger/voucher/{voucherId}/reverse?date=YYYY-MM-DD
      → Create a corrected voucher: POST /v2/ledger/voucher with the right account
    Duplicate voucher (same account and amount appearing twice):
      → Reverse the duplicate: PUT /v2/ledger/voucher/{duplicateVoucherId}/reverse?date=YYYY-MM-DD
    Missing VAT line (expense posted without VAT, should have input VAT on 2710):
      → Reverse the voucher without VAT: PUT /v2/ledger/voucher/{id}/reverse?date=YYYY-MM-DD
      → Create a corrected voucher with vatType set on the expense row (system auto-generates 2710 posting)
    Wrong amount (e.g. 18600 posted instead of 11550):
      → Reverse the wrong voucher: PUT /v2/ledger/voucher/{id}/reverse?date=YYYY-MM-DD
      → Create a corrected voucher with the correct amount

  Reversal endpoint:
    PUT /v2/ledger/voucher/{id}/reverse?date=YYYY-MM-DD  (returns 200, creates a reverse voucher)
    Use the SAME date as the original posting, or a date in the same period (Jan/Feb 2026).

  Corrected voucher creation (POST /v2/ledger/voucher):
    Mirror the original posting structure but with the fix applied. Match the original date and description.
    Double-entry: if original debited expense account X, credit was likely 1920 (bank) or 2400 (AP).
    Use the SAME credit account as the original to maintain balance.
    Get credit account id from the original posting data (already in the posting response's voucher.postings).

  CRITICAL: After getting posting data (Step 1), go STRAIGHT to corrections (Step 2).
  Do NOT fetch individual vouchers by ID. The posting data already gives you voucher.id + all amounts needed.
  Target: complete all 4 fixes in 3-4 turns total (1 GET turn + 3-4 write turns).

TIME LIMIT — CRITICAL:
  The competition platform times out after 5 minutes. Tasks that take too long return "endpoint unreachable" and score zero.
  You MUST complete each task in under 4.5 minutes (leave margin). If you are running many GET lookups to find an exact code or ID:
  - After 2 failed searches, pick the closest match and proceed. Do not retry the same lookup with slight variations.
  - Skip optional verification GETs at the end — trust that your write calls succeeded if they returned 2xx.
  - Prioritize completing the core task writes over finding the perfect field value.

WHEN RECEIVING A TASK THINK THOROUGHLY TO PLAN WHAT TO DO TO GET MOST POINTS AND ACT PRECISELY
""".strip()

# ─────────────────────────────────────────────────────────────────────────────
# Build user message content
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_content(request: SolveRequest) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": request.prompt}]
    for f in request.files:
        mime = f.mime_type.lower()
        data = f.get_data()
        if mime.startswith("image/"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
        elif mime == "application/pdf":
            content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}})
        else:
            # Decode text-based files (CSV, TXT, etc.) so Claude can read the actual content
            try:
                import base64 as _b64
                decoded = _b64.b64decode(data).decode("utf-8", errors="replace")
                label = f.filename or mime
                content.append({"type": "text", "text": f"[Attachment: {label}]\n{decoded}"})
            except Exception:
                content.append({"type": "text", "text": f"[Attachment: {f.filename or mime}, {len(data)} chars base64]"})
    return content

# ─────────────────────────────────────────────────────────────────────────────
# Agentic loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(request: SolveRequest) -> None:
    ai = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY, max_retries=6)
    api = TripletexClient(
        base_url=request.tripletex_credentials.base_url,
        session_token=request.tripletex_credentials.session_token,
    )
    messages: list[dict] = [{"role": "user", "content": _build_user_content(request)}]

    for turn in range(MAX_AGENT_TURNS):
        log.info("─── Turn %d/%d ───", turn + 1, MAX_AGENT_TURNS)

        async with ai.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=[TRIPLETEX_TOOL],
            messages=messages,
        ) as stream:
            response = await stream.get_final_message()

        log.info("Stop reason: %s", response.stop_reason)
        for block in response.content:
            if block.type == "thinking":
                log.info("[thinking] %s…", block.thinking[:300])

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            log.info("Done after %d turn(s).", turn + 1)
            break

        tool_results: list[dict] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            inp    = block.input
            method = str(inp.get("method", "GET")).upper()
            path   = str(inp.get("path", ""))
            params = inp.get("params")
            body   = inp.get("body")
            log.info("Tool: %s %s", method, path)
            result = await api.call(method=method, path=path, params=params, body=body)
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 8000:
                result_str = result_str[:8000] + f"\n...[truncated, add ?fields= to get less data]"
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_str})

        if not tool_results:
            log.warning("No tool results — stopping.")
            break

        messages.append({"role": "user", "content": tool_results})
    else:
        log.warning("Reached MAX_AGENT_TURNS (%d).", MAX_AGENT_TURNS)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Tripletex Opus Agent", version="3.0.0")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    body = await request.body()
    log.error("422 Validation error — raw body: %s", body.decode()[:500])
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/solve", response_model=SolveResponse)
async def solve(
    request: SolveRequest,
    authorization: str | None = Header(default=None),
) -> SolveResponse:
    if AGENT_API_KEY:
        if authorization != f"Bearer {AGENT_API_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")
    log.info("New task:\n%s", request.prompt)
    try:
        await run_agent(request)
    except Exception as e:
        log.error("Agent error (returning completed anyway): %s", e)
    return SolveResponse(status="completed")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL, "version": "3.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("opus:app", host="0.0.0.0", port=8002, reload=False, log_level="info")
