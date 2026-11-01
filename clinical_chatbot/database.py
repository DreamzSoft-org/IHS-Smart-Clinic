import os
import json
import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import gspread
import requests
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Anchor everything to this file's own folder (Chatbot/), not to whatever
# directory the process happened to be launched from. Without this, a
# relative GOOGLE_SERVICE_ACCOUNT_FILE value resolves against the caller's
# working directory, so it only finds the key when launched one specific
# way (e.g. "cd Chatbot && ...") and silently fails otherwise.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

def _read_project_google_config() -> dict:
    """Read the same Google Sheet configuration used by the main SMART CDSS app."""
    path = PROJECT_DIR / "google_config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload.get("google_sheet", payload) or {}
    except Exception:
        return {}

logger = logging.getLogger("cdss.database")

# Exceptions worth retrying: transient network/connection issues (e.g. a
# brief Wi-Fi reconnect or VPN blip aborting the TCP/SSL connection mid
# request — WinError 10053 on Windows). NOT retried: auth errors, missing
# sheet/tab, bad data — those are real problems, not transient ones.
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionAbortedError,
    ConnectionResetError,
)


def _call_with_retry(fn: Callable, *args, max_attempts: int = 3, base_delay: float = 0.75, **kwargs):
    """Calls fn(*args, **kwargs), retrying with exponential backoff
    (0.75s, 1.5s, ...) if it raises a transient connection error. Re-raises
    the last error if every attempt fails.

    Note: this is used for reads and for update_cell (safe to repeat), and
    for append_row too — in the rare case a request actually succeeded on
    Google's end but the response was lost before we saw it, a retry could
    append a duplicate row. That's an acceptable tradeoff here (a stray
    duplicate row is easy to spot/clean up; a hard-failed intake/appointment
    write is worse), but worth knowing about.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s: transient error on attempt %d/%d (%s), retrying in %.1fs",
                getattr(fn, "__name__", "gspread_call"), attempt, max_attempts, exc, delay,
            )
            time.sleep(delay)
    logger.error("%s: failed after %d attempts: %s", getattr(fn, "__name__", "gspread_call"), max_attempts, last_exc, exc_info=True)
    raise last_exc


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

_project_google = _read_project_google_config()
SHEET_ID = (os.environ.get("GOOGLE_SHEET_ID") or _project_google.get("spreadsheet_id") or "").strip()

_service_account_file_raw = (os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE") or "service_account.json").strip()
_service_account_path = Path(_service_account_file_raw)
if _service_account_path.is_absolute():
    _resolved_service_account = _service_account_path
else:
    # Use the same project-root credential as the main SMART CDSS app first.
    # A stale chatbot-local key can still be a syntactically valid JSON/private
    # key while Google rejects it with "invalid_grant: Invalid JWT Signature"
    # after that key has been rotated/deleted.  Keeping the chatbot on the
    # exact credential used by registration/login avoids that split-brain
    # authentication failure.  The chatbot-local file remains a fallback for
    # standalone chatbot launches where no project-root key is present.
    root_candidate = PROJECT_DIR / _service_account_path.name
    local_candidate = BASE_DIR / _service_account_path
    _resolved_service_account = root_candidate if root_candidate.is_file() else local_candidate

SERVICE_ACCOUNT_FILE = str(_resolved_service_account)

# --- Patients: merged into SMART CDSS's own "Patients" worksheet ------------
# SMART CDSS already owns a "Patients" tab (user_id, full_name, age, gender,
# symptoms, disease, notes, updated_at) in this same spreadsheet. Rather than
# keeping a second, separate "patients" tab, the chatbot's fields live as
# extra trailing columns on that SAME tab/row per patient — one patient, one
# row, no duplicate table. SMART_CDSS/app.py has a matching non-destructive
# header-upgrade step, so this is safe to point at even on first run.
PATIENTS_SHEET_TITLE = "Patients"

# The on-disk column order (must match SMART_CDSS's PATIENT_HEADERS exactly).
PATIENTS_SHEET_HEADERS = [
    "user_id", "full_name", "age", "gender", "symptoms", "disease", "notes", "updated_at",
    "allergies", "current_medications", "medical_history", "intake_completed",
    "visit_reason", "visit_category", "needs_prescription_renewal", "needs_medical_note",
    "regular_patient", "phone", "symptom_onset", "symptom_severity",
]

# The rest of this codebase (agents.py, mcp_tools.py, index.html) refers to
# "patient_id" and "name" — only these two columns are named differently on
# SMART CDSS's sheet, so alias just these two both ways.
_SHEET_TO_LOGICAL = {"user_id": "patient_id", "full_name": "name"}
_LOGICAL_TO_SHEET = {v: k for k, v in _SHEET_TO_LOGICAL.items()}
PATIENTS_LOGICAL_HEADERS = [_SHEET_TO_LOGICAL.get(h, h) for h in PATIENTS_SHEET_HEADERS]

# Tab names + header rows for everything EXCEPT patients (see above). This IS
# the schema — ensure_schema() creates any tab that's missing with these
# exact headers, so a brand-new blank spreadsheet works out of the box.
SCHEMA: dict[str, list[str]] = {
    "doctors": [
        "patient_id", "name", "specialty", "working_hours_start",
        "working_hours_end", "slot_minutes",
    ],
    "staff": ["patient_id", "name", "department"],
    "otc_medications": ["medication_name", "targets", "interacts_with", "contraindications", "usage_note"],
    "appointments": [
        "appointment_id", "patient_id", "doctor_id", "doctor_name", "specialist", "date",
        "duration_minutes", "status", "reason", "consultation_mode", "contact_phone", "created_at",
    ],
}

_client = None
_spreadsheet = None
DB_ERROR: Optional[str] = None
_lock = threading.Lock()


def _connect_google_sheets() -> bool:
    """(Re)connect to the configured Google Sheet.

    A transient network failure during FastAPI startup must not leave the
    chatbot permanently disconnected for the lifetime of the server process.
    Every DB access can call this function and recover automatically.
    """
    global _client, _spreadsheet, DB_ERROR
    try:
        if not SHEET_ID:
            raise RuntimeError("GOOGLE_SHEET_ID is not configured")
        if not Path(SERVICE_ACCOUNT_FILE).is_file():
            raise RuntimeError(f"Google service-account file was not found: {SERVICE_ACCOUNT_FILE}")
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        spreadsheet = _call_with_retry(client.open_by_key, SHEET_ID)
        _client = client
        _spreadsheet = spreadsheet
        DB_ERROR = None
        return True
    except Exception as exc:
        _client = None
        _spreadsheet = None
        DB_ERROR = str(exc)
        logger.error("Google Sheets connection failed: %s", exc, exc_info=True)
        return False


# IMPORTANT: do not contact Google during module import.
# Uvicorn must become available immediately so the embedded chatbot UI can
# render even when Google is slow, offline, or rejecting a credential.  The
# first database operation (or the background startup task) performs the
# connection lazily instead.
if not SHEET_ID:
    DB_ERROR = "GOOGLE_SHEET_ID is not configured"
elif not Path(SERVICE_ACCOUNT_FILE).is_file():
    DB_ERROR = f"Google service-account file was not found: {SERVICE_ACCOUNT_FILE}"
else:
    DB_ERROR = "Google Sheets connection has not been attempted yet."


def is_available(reconnect: bool = True) -> bool:
    if _spreadsheet is not None:
        return True
    return _connect_google_sheets() if reconnect else False


def _unavailable() -> dict:
    return {"error": f"Google Sheets unavailable: {DB_ERROR}. Check .env and sheet sharing."}


# --- SCHEMA / WORKSHEET ACCESS ------------------------------------------------

def ensure_schema():
    """Creates any missing tab (with headers) in the spreadsheet. Safe to call
    repeatedly; a tab that already exists is left untouched."""
    if not is_available():
        return _unavailable()
    worksheets = list(_call_with_retry(_spreadsheet.worksheets))
    existing = {ws.title for ws in worksheets}
    existing_cf = {ws.title.strip().casefold(): ws.title for ws in worksheets}
    for tab, headers in SCHEMA.items():
        if tab.casefold() not in existing_cf:
            ws = _call_with_retry(_spreadsheet.add_worksheet, title=tab, rows=200, cols=max(10, len(headers)))
            _call_with_retry(ws.update, "A1", [headers])
            existing.add(tab)
            existing_cf[tab.casefold()] = tab
        else:
            _ensure_standard_tab_schema(tab, headers)
    _ensure_patients_schema(existing)
    return {"ok": True}


def _ensure_standard_tab_schema(tab: str, expected_headers: list[str]):
    """Non-destructively add any missing columns to an existing standard tab.

    Older project copies can already have an ``appointments`` worksheet with
    the pre-consultation schema.  ``ensure_schema`` previously skipped every
    existing tab entirely, so new appointment rows were appended with more
    values than the sheet had headers for.  Reads then lost the trailing
    appointment fields and, depending on the existing sheet shape, could make
    booking fail before a usable response reached the frontend.

    Existing columns and data are never renamed, reordered, or deleted.
    Missing expected columns are appended to the right only.
    """
    try:
        ws = _call_with_retry(_spreadsheet.worksheet, tab)
    except gspread.exceptions.WorksheetNotFound:
        target = tab.strip().casefold()
        ws = next(
            (candidate for candidate in _call_with_retry(_spreadsheet.worksheets)
             if candidate.title.strip().casefold() == target),
            None,
        )
        if ws is None:
            raise
    current = [str(v).strip() for v in _call_with_retry(ws.row_values, 1)]
    if not current:
        _call_with_retry(ws.update, "A1", [expected_headers])
        return

    missing = [h for h in expected_headers if h not in current]
    if not missing:
        return

    needed_cols = len(current) + len(missing)
    if getattr(ws, "col_count", 0) < needed_cols:
        _call_with_retry(ws.resize, cols=needed_cols)
    for offset, header in enumerate(missing, start=len(current) + 1):
        _call_with_retry(ws.update_cell, 1, offset, header)


def _ensure_patients_schema(existing_titles: set):
    """Creates or non-destructively upgrades SMART CDSS's "Patients" tab so
    it has all the chatbot's columns too. Mirrors the same upgrade logic
    SMART_CDSS/app.py itself uses — existing rows/values are never touched,
    only missing trailing columns are appended."""
    if PATIENTS_SHEET_TITLE not in existing_titles:
        ws = _call_with_retry(
            _spreadsheet.add_worksheet, title=PATIENTS_SHEET_TITLE,
            rows=1000, cols=len(PATIENTS_SHEET_HEADERS),
        )
        _call_with_retry(ws.update, "A1", [PATIENTS_SHEET_HEADERS])
        return

    ws = _call_with_retry(_spreadsheet.worksheet, PATIENTS_SHEET_TITLE)
    first_row = _call_with_retry(ws.row_values, 1)
    if first_row == PATIENTS_SHEET_HEADERS:
        return
    if not first_row:
        _call_with_retry(ws.update, "A1", [PATIENTS_SHEET_HEADERS])
    elif first_row == PATIENTS_SHEET_HEADERS[: len(first_row)]:
        ws.resize(cols=len(PATIENTS_SHEET_HEADERS))
        for column_number, header in enumerate(
            PATIENTS_SHEET_HEADERS[len(first_row):], start=len(first_row) + 1
        ):
            _call_with_retry(ws.update_cell, 1, column_number, header)
    # else: headers diverge in an unexpected way — leave it alone rather than
    # risk corrupting SMART CDSS's sheet; SMART_CDSS/app.py's own startup
    # check will surface a clear error if this happens.


def _sheet_headers(tab: str) -> list[str]:
    return PATIENTS_SHEET_HEADERS if tab == "patients" else SCHEMA[tab]


def _logical_headers(tab: str) -> list[str]:
    return PATIENTS_LOGICAL_HEADERS if tab == "patients" else SCHEMA[tab]


def _ws(tab: str):
    if not is_available():
        raise RuntimeError(f"Google Sheets unavailable: {DB_ERROR}")
    title = PATIENTS_SHEET_TITLE if tab == "patients" else tab
    try:
        return _call_with_retry(_spreadsheet.worksheet, title)
    except gspread.exceptions.WorksheetNotFound:
        target = title.strip().casefold()
        for candidate in _call_with_retry(_spreadsheet.worksheets):
            if candidate.title.strip().casefold() == target:
                return candidate

        # Self-heal missing chatbot-owned tabs.  This also means a server that
        # started before schema initialization completed can recover later.
        if tab in SCHEMA:
            headers = SCHEMA[tab]
            ws = _call_with_retry(
                _spreadsheet.add_worksheet,
                title=tab,
                rows=500,
                cols=max(12, len(headers)),
            )
            _call_with_retry(ws.update, "A1", [headers])
            return ws
        if tab == "patients":
            ws = _call_with_retry(
                _spreadsheet.add_worksheet,
                title=PATIENTS_SHEET_TITLE,
                rows=500,
                cols=max(20, len(PATIENTS_SHEET_HEADERS)),
            )
            _call_with_retry(ws.update, "A1", [PATIENTS_SHEET_HEADERS])
            return ws
        raise


_CACHE_TTL_SECONDS = 15
_cache: dict[str, tuple[float, list[dict]]] = {}
_REGISTERED_DOCTOR_CACHE_TTL_SECONDS = 60
_registered_doctor_cache: tuple[float, list[dict]] | None = None


def _read_all(tab: str, force: bool = False) -> list[dict]:
    with _lock:
        cached = _cache.get(tab)
        if not force and cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
            return [row.copy() for row in cached[1]]

    ws = _ws(tab)
    values = _call_with_retry(ws.get_all_values)
    if not values:
        records = []
    else:
        raw_headers = [str(v).strip() for v in values[0]]
        # Preserve the first instance of a duplicate header and ignore blank
        # header cells. gspread.get_all_records raises on duplicate headers;
        # manual mapping keeps old/hand-edited Sheets usable.
        seen = set()
        header_slots = []
        for idx, header in enumerate(raw_headers):
            if not header or header in seen:
                continue
            seen.add(header)
            logical = _SHEET_TO_LOGICAL.get(header, header) if tab == "patients" else header
            header_slots.append((idx, logical))
        records = []
        for sheet_row_number, raw in enumerate(values[1:], start=2):
            if not any(str(v).strip() for v in raw):
                continue
            rec = {}
            for idx, header in header_slots:
                rec[header] = raw[idx] if idx < len(raw) else ""
            rec["_row_number"] = sheet_row_number
            records.append(rec)

    if not values:
        records = []
    for row in records:
        row.setdefault("_row_number", None)

    with _lock:
        _cache[tab] = (time.time(), records)
    return [row.copy() for row in records]


def _invalidate(tab: str):
    with _lock:
        _cache.pop(tab, None)


def _split(value) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return []
    return [v.strip() for v in text.split(",") if v.strip() and v.strip().lower() != "none"]


def _append_row(tab: str, row: dict):
    ws = _ws(tab)

    if tab == "patients":
        sheet_headers = _sheet_headers(tab)
        logical_headers = _logical_headers(tab)
        values = [
            row.get(logical_h, row.get(sheet_h, ""))
            for sheet_h, logical_h in zip(sheet_headers, logical_headers)
        ]
    else:
        # Align writes to the sheet's ACTUAL current header order.  This keeps
        # old installations working even when their appointments tab predates
        # newer columns such as consultation_mode/contact_phone/created_at.
        # ensure_schema() appends missing expected headers without touching
        # existing data, and this mapping prevents positional miswrites.
        current_headers = [str(v).strip() for v in _call_with_retry(ws.row_values, 1)]
        expected = _sheet_headers(tab)
        missing = [h for h in expected if h not in current_headers]
        if missing:
            _ensure_standard_tab_schema(tab, expected)
            current_headers = [str(v).strip() for v in _call_with_retry(ws.row_values, 1)]
        values = [row.get(header, "") for header in current_headers]

    _call_with_retry(ws.append_row, values, value_input_option="USER_ENTERED")
    _invalidate(tab)


def _update_row(tab: str, row_number: int, updates: dict):
    sheet_headers = _sheet_headers(tab)
    to_sheet = _LOGICAL_TO_SHEET if tab == "patients" else {}
    ws = _ws(tab)
    for key, value in updates.items():
        sheet_key = to_sheet.get(key, key)
        if sheet_key not in sheet_headers:
            continue
        col = sheet_headers.index(sheet_key) + 1
        _call_with_retry(ws.update_cell, row_number, col, value if not isinstance(value, list) else ", ".join(value))
    _invalidate(tab)


def _find_one(tab: str, predicate: Callable[[dict], bool]) -> Optional[dict]:
    for row in _read_all(tab):
        if predicate(row):
            return row
    return None


def _find_all(tab: str, predicate: Callable[[dict], bool] = lambda r: True) -> list[dict]:
    return [r for r in _read_all(tab) if predicate(r)]



# --- SMART CDSS REGISTERED DOCTORS --------------------------------------------

# Doctor accounts are created by the main SMART CDSS signup form in the same
# spreadsheet.  Keep the chatbot's doctor roster linked to those registrations
# instead of requiring a second manual entry in the "doctors" tab.
_REGISTRATION_SHEET_CANDIDATES = (
    "Users",
    "User Registration Responses",
    "Form Responses 1",
    "Form Responses",
)


def _norm_registration_header(value: object) -> str:
    text = str(value or "").strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_")
    aliases = {
        "userid": "user_id",
        "user_id": "user_id",
        "name": "full_name",
        "fullname": "full_name",
        "full_name": "full_name",
        "user_type": "user_type",
        "usertype": "user_type",
        "account_type": "user_type",
        "role": "user_type",
        "speciality": "specialty",
        "specialty": "specialty",
        "specialization": "specialty",
        "medical_specialty": "specialty",
        "medical_speciality": "specialty",
        "doctor_specialty": "specialty",
        "doctor_speciality": "specialty",
        "department": "specialty",
    }
    return aliases.get(text, text)


def _registered_doctor_accounts() -> list[dict]:
    """Return doctor accounts from SMART CDSS's signup/registration sheets.

    Existing rows in the chatbot's ``doctors`` tab are still used for optional
    scheduling metadata such as specialty and working hours, but registration
    is the source of truth for who is a doctor.
    """
    global _registered_doctor_cache
    if not is_available():
        return []

    with _lock:
        cached = _registered_doctor_cache
        if (
            cached
            and time.time() - cached[0] < _REGISTERED_DOCTOR_CACHE_TTL_SECONDS
        ):
            return [doctor.copy() for doctor in cached[1]]

    doctors_by_id: dict[str, dict] = {}
    read_failed = False
    try:
        worksheets = {
            ws.title: ws for ws in _call_with_retry(_spreadsheet.worksheets)
        }
    except Exception:
        return []

    # Prefer the current Users sheet, while retaining compatibility with the
    # older Google Form / User Registration Responses tabs.
    for title in _REGISTRATION_SHEET_CANDIDATES:
        ws = worksheets.get(title)
        if ws is None:
            # Be tolerant of capitalization differences.
            ws = next(
                (candidate for candidate_title, candidate in worksheets.items()
                 if candidate_title.strip().casefold() == title.casefold()),
                None,
            )
        if ws is None:
            continue
        try:
            values = _call_with_retry(ws.get_all_values)
        except Exception:
            read_failed = True
            continue
        if len(values) < 2:
            continue

        headers = [_norm_registration_header(v) for v in values[0]]
        for raw_row in values[1:]:
            padded = list(raw_row) + [""] * max(0, len(headers) - len(raw_row))
            record = {
                header: str(padded[i] or "").strip()
                for i, header in enumerate(headers)
                if header
            }
            role = record.get("user_type", "").strip().casefold()
            if role not in {"doctor", "clinician", "physician"} and "doctor" not in role:
                continue
            doctor_id = record.get("user_id", "").strip()
            if not doctor_id:
                continue
            doctors_by_id[doctor_id.casefold()] = {
                "patient_id": doctor_id,
                "name": record.get("full_name", "").strip() or doctor_id,
                "specialty": record.get("specialty", "").strip(),
            }

    doctors = list(doctors_by_id.values())
    if not read_failed:
        with _lock:
            _registered_doctor_cache = (time.time(), doctors)
    return [doctor.copy() for doctor in doctors]


def _merged_registered_doctor(doctor_id: str) -> Optional[dict]:
    wanted = str(doctor_id or "").strip().casefold()
    registered = next(
        (d for d in _registered_doctor_accounts()
         if str(d.get("patient_id", "")).strip().casefold() == wanted),
        None,
    )
    if not registered:
        return None

    # Optional scheduling settings can still live in the doctors tab. Merge
    # them onto the registered account without allowing a stale tab row to
    # change the signup identity/name.
    configured = _find_one(
        "doctors",
        lambda r: str(r.get("patient_id", "")).strip().casefold() == wanted,
    )
    merged = {
        "patient_id": registered["patient_id"],
        "name": registered.get("name", "") or registered["patient_id"],
        "specialty": registered.get("specialty", ""),
        # Blank means "no explicit working-hours restriction".  Registered
        # doctors are bookable at a patient-selected future time unless the
        # clinic has actually configured hours for them in the doctors tab.
        "working_hours_start": "",
        "working_hours_end": "",
        "slot_minutes": 30,
    }
    if configured:
        for key in ("specialty", "working_hours_start", "working_hours_end", "slot_minutes"):
            if configured.get(key) not in (None, ""):
                merged[key] = configured.get(key)
    return merged


# --- ROLE RESOLUTION -----------------------------------------------------------

def resolve_role(person_id: str) -> Optional[str]:
    """Resolve roles server-side, with SMART CDSS registration as doctor truth."""
    if _merged_registered_doctor(person_id):
        return "doctor"
    if _find_one("patients", lambda r: r.get("patient_id") == person_id):
        return "patient"
    if _find_one("doctors", lambda r: r.get("patient_id") == person_id):
        return "doctor"
    if _find_one("staff", lambda r: r.get("patient_id") == person_id):
        return "staff"
    return None


# --- PATIENTS ------------------------------------------------------------------

def get_patient(patient_id: str) -> Optional[dict]:
    row = _find_one("patients", lambda r: r.get("patient_id") == patient_id)
    if not row:
        return None
    row = row.copy()
    row["allergies"] = _split(row.get("allergies"))
    row["current_medications"] = _split(row.get("current_medications"))
    row["medical_history"] = _split(row.get("medical_history"))
    row.pop("_row_number", None)
    return row


def get_patient_list() -> list[dict]:
    return [{"patient_id": r["patient_id"], "name": r.get("name", "")} for r in _read_all("patients")]


def update_patient(patient_id: str, updates: dict) -> bool:
    row = _find_one("patients", lambda r: r.get("patient_id") == patient_id)
    if not row:
        return False
    _update_row("patients", row["_row_number"], updates)
    return True


def create_patient(patient_id: str, name: str, age="", gender="") -> dict:
    row = {
        "patient_id": patient_id, "name": name, "age": age, "gender": gender,
        "allergies": "", "current_medications": "", "medical_history": "",
        "intake_completed": "false", "visit_reason": "", "visit_category": "",
        "needs_prescription_renewal": "", "needs_medical_note": "",
        "regular_patient": "", "phone": "",
    }
    _append_row("patients", row)
    return row


# --- DOCTORS / STAFF -------------------------------------------------------

def get_doctor(doctor_id: str) -> Optional[dict]:
    """Resolve a doctor by canonical user_id, with name compatibility for old rows/UI."""
    wanted = str(doctor_id or "").strip().casefold()
    registered = _merged_registered_doctor(doctor_id)
    if registered:
        return registered

    # Compatibility: some earlier appointment/UI builds persisted or passed a
    # doctor's display name instead of the immutable registration user_id.
    for account in _registered_doctor_accounts():
        if str(account.get("name", "")).strip().casefold() == wanted:
            return _merged_registered_doctor(account["patient_id"]) or account

    return _find_one(
        "doctors",
        lambda r: (
            str(r.get("patient_id", "")).strip().casefold() == wanted
            or str(r.get("name", "")).strip().casefold() == wanted
        ),
    )


def get_doctor_list() -> list[dict]:
    # Show every doctor who signed up in SMART CDSS. Existing doctors-tab
    # metadata is merged in by _merged_registered_doctor().
    registered = _registered_doctor_accounts()
    if registered:
        result = []
        for account in registered:
            merged = _merged_registered_doctor(account["patient_id"]) or account
            result.append({
                "patient_id": merged["patient_id"],
                "name": merged.get("name", ""),
                "specialty": merged.get("specialty", ""),
            })
        result.sort(key=lambda d: (str(d.get("name", "")).casefold(), str(d.get("patient_id", "")).casefold()))
        return result

    # Backward-compatible fallback for installations that have not yet
    # migrated their signup sheet.
    return [{"patient_id": r["patient_id"], "name": r.get("name", ""), "specialty": r.get("specialty", "")}
            for r in _read_all("doctors")]


def get_staff_list() -> list[dict]:
    return [{"patient_id": r["patient_id"], "name": r.get("name", "")} for r in _read_all("staff")]


def create_doctor(doctor_id: str, name: str, specialty: str = "",
                   working_hours_start: str = "", working_hours_end: str = "",
                   slot_minutes: int = 30) -> dict:
    """Registers a new doctor row. Used the first time a real doctor account
    (created in SMART_CDSS's own signup form) opens the chatbot — mirrors
    create_patient()'s lazy-create-on-first-visit pattern above, so doctors
    no longer have to be pre-seeded via seed_sheets.py to be recognized."""
    row = {
        "patient_id": doctor_id, "name": name, "specialty": specialty,
        "working_hours_start": working_hours_start or "",
        "working_hours_end": working_hours_end or "",
        "slot_minutes": slot_minutes or 30,
    }
    _append_row("doctors", row)
    return row


def find_doctors_by_specialty(specialty: str) -> list[dict]:
    requested = str(specialty or "").strip().casefold()
    doctors = get_doctor_list()

    # The appointment widget now presents registered doctor names. Match an
    # exact doctor name (or ID) first; specialty matching remains supported for
    # symptom-flow recommendations and older clients.
    named = [
        d for d in doctors
        if str(d.get("name", "")).strip().casefold() == requested
        or str(d.get("patient_id", "")).strip().casefold() == requested
    ]
    if named:
        return [get_doctor(d["patient_id"]) or d for d in named]

    return [
        get_doctor(d["patient_id"]) or d
        for d in doctors
        if str(d.get("specialty", "")).strip().casefold() == requested
    ]


# --- OTC MEDICATIONS ---------------------------------------------------------

def get_otc_medications() -> list[dict]:
    rows = _read_all("otc_medications")
    out = []
    for r in rows:
        r = r.copy()
        r["targets"] = _split(r.get("targets"))
        r["interacts_with"] = _split(r.get("interacts_with"))
        r.pop("_row_number", None)
        out.append(r)
    return out


# --- APPOINTMENTS ------------------------------------------------------------

def _same_identity(left: object, right: object) -> bool:
    """Compare persisted SMART CDSS IDs safely across Sheets/form formatting."""
    return str(left or "").strip().casefold() == str(right or "").strip().casefold()


def get_appointments(patient_id: Optional[str] = None, doctor_id: Optional[str] = None,
                      active_only: bool = True, force_refresh: bool = False) -> list[dict]:
    def pred(r):
        if patient_id and not _same_identity(r.get("patient_id"), patient_id):
            return False
        if doctor_id and not _same_identity(r.get("doctor_id"), doctor_id):
            return False
        if active_only and str(r.get("status", "")).strip().casefold() in ("cancelled", "completed"):
            return False
        return True
    rows = [r for r in _read_all("appointments", force=force_refresh) if pred(r)]
    for r in rows:
        r.pop("_row_number", None)
    return rows


def get_doctor_schedule_appointments(doctor_id: str, active_only: bool = True) -> list[dict]:
    """Return appointments belonging to a doctor, including legacy rows.

    Current bookings use appointments.doctor_id = the doctor's immutable
    registration user_id. Older builds sometimes stored a doctor name in
    doctor_id, doctor_name, or specialist. We resolve the logged-in doctor
    once and accept those aliases so old and new bookings both appear.
    """
    canonical = get_doctor(doctor_id)
    canonical_id = str((canonical or {}).get("patient_id") or doctor_id or "").strip()
    canonical_name = str((canonical or {}).get("name") or "").strip()

    wanted = {
        value.casefold()
        for value in (canonical_id, canonical_name, str(doctor_id or "").strip())
        if value
    }

    rows = _read_all("appointments", force=True)
    matched = []
    for raw in rows:
        status = str(raw.get("status", "")).strip().casefold()
        if active_only and status in ("cancelled", "completed"):
            continue

        aliases = {
            str(raw.get("doctor_id", "")).strip().casefold(),
            str(raw.get("doctor_name", "")).strip().casefold(),
        }

        # Legacy compatibility only: old chatbot builds sometimes persisted
        # the selected doctor's name in specialist.
        specialist_value = str(raw.get("specialist", "")).strip().casefold()
        if specialist_value:
            aliases.add(specialist_value)

        if wanted.intersection(a for a in aliases if a):
            row = raw.copy()
            row.pop("_row_number", None)
            matched.append(row)

    return matched


def _next_appointment_id() -> str:
    existing = _read_all("appointments")
    return f"APT{len(existing) + 1:05d}-{int(time.time()) % 100000}"


def create_appointment(patient_id: str, doctor_id: str, specialist: str, date: str,
                        duration_minutes: int = 30, reason: str = "", consultation_mode: str = "in-clinic",
                        contact_phone: str = "", doctor_name: str = "") -> dict:
    # Resolve the selected doctor to the immutable registration user_id before
    # persisting. This is the exact ID used again when that doctor logs in.
    resolved_doctor = get_doctor(doctor_id)
    canonical_doctor_id = str(
        (resolved_doctor or {}).get("patient_id") or doctor_id or ""
    ).strip()
    canonical_doctor_name = str(
        (resolved_doctor or {}).get("name") or doctor_name or ""
    ).strip()

    appt = {
        "appointment_id": _next_appointment_id(),
        "patient_id": str(patient_id or "").strip(),
        "doctor_id": canonical_doctor_id,
        "doctor_name": canonical_doctor_name,
        "specialist": specialist,
        "date": date,
        "duration_minutes": duration_minutes,
        "status": "scheduled",
        "reason": reason,
        "consultation_mode": consultation_mode,
        "contact_phone": contact_phone or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_row("appointments", appt)

    # Read-after-write validation. A successful patient confirmation must mean
    # that the appointment is actually visible in the selected doctor's
    # schedule, not merely that append_row returned without an exception.
    persisted = next(
        (
            row for row in get_doctor_schedule_appointments(canonical_doctor_id, active_only=False)
            if str(row.get("appointment_id", "")).strip() == appt["appointment_id"]
        ),
        None,
    )
    if not persisted:
        raise RuntimeError(
            "Appointment was written but could not be resolved back to the selected doctor's schedule."
        )
    return persisted


def update_appointment(appointment_id: str, updates: dict) -> bool:
    row = _find_one("appointments", lambda r: r.get("appointment_id") == appointment_id)
    if not row:
        return False
    _update_row("appointments", row["_row_number"], updates)
    return True


if __name__ == "__main__":
    print(ensure_schema())