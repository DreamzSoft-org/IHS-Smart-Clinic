# What changed, and how to get it running

## 1. Data layer: MongoDB → Google Sheets
`database.py` is fully rewritten around `gspread`. Setup steps are also
written at the top of that file:

1. In Google Cloud Console, enable the **Sheets API** and **Drive API** for a project.
2. Create a **Service Account**, then a JSON key for it — download the file.
3. Create a blank Google Sheet. Copy its ID out of the URL:
   `https://docs.google.com/spreadsheets/d/<THIS_PART>/edit`
4. **Share** that Sheet with the service account's email (ends in
   `...iam.gserviceaccount.com`) as **Editor**. This is the step people
   usually forget — without it you'll get a 403 on first run.
5. Add to `.env`:
   ```
   GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/to/service_account.json
   GOOGLE_SHEET_ID=<the id from step 3>
   ```
6. Run `python seed_sheets.py` once to create all five tabs (patients,
   doctors, staff, otc_medications, appointments) and load your CSVs into
   them. See the docstring at the top of `seed_sheets.py` for the exact CSV
   shape it expects — `specialty` / `working_hours_start` /
   `working_hours_end` are required on doctor rows for the appointment
   matching logic to work.
7. Your colleague can have this same spreadsheet open/shared normally —
   the app only needs the service account to have Editor access; it
   doesn't care who else is looking at it.

`pip install gspread google-auth` (already in `requirements.txt`).

## 2. New agent architecture (still ReAct — `create_react_agent` from
LangGraph, same as before)
`agents.py` adds:
- **`run_patient_symptom_flow()`** — the orchestrator. Always runs the
  symptom checker first, then *in plain Python* (not left to the LLM)
  decides the next step:
  - clarifying question pending → return it, wait for the patient's answer
  - emergency / high severity / `recommend_appointment` → hands off to the
    **appointment agent**
  - otherwise → automatically calls the **medication agent** with the
    same complaint and merges both replies into one response
- The symptom checker is now based on direct patient complaint and clinical
  reasoning, not a separate symptom knowledge base.
- A new **Lab Report agent** (`run_lab_report_agent`) — not a ReAct tool
  loop, since it's a single multimodal read rather than a multi-step
  task. It extracts text from text-layer PDFs directly (PyMuPDF), or
  rasterizes scanned PDFs/images and sends them to a vision-capable model
  (Gemini by default — set `GEMINI_API_KEY`). Explains results
  in plain language, flags anything worth discussing with a doctor.
- Intake is intentionally **not** an LLM agent — `INTAKE_QUESTIONS` +
  `submit_intake_answers()` is a small scripted Q&A that writes straight
  onto the patient's sheet row. Cheaper and more predictable than an
  agent for five fixed questions; the symptom checker still asks its own
  free-form clarifying questions afterward.

`pip install pymupdf` for the lab report PDF handling (in
`requirements.txt`).

## 3. Smarter appointment matching (`mcp_tools.py`)
- **`find_available_doctor(specialty, date, duration_minutes)`** — new
  tool. Filters doctors by specialty, then by working hours, then checks
  their existing appointments for a time overlap. `manage_appointment`
  calls this internally on `action="create"`, so booking now auto-assigns
  a specific, actually-free doctor of the right specialty instead of just
  recording a bare "specialist" string.
- **`get_doctor_schedule(doctor_id, requester_patient_id)`** — new tool
  enforcing: a doctor sees only their own schedule, staff sees any
  doctor's, everyone else is denied. Patients still can't see anyone
  else's appointments (unchanged from before — `manage_appointment`
  lookup still scopes to the caller's own `patient_id` unless they're
  staff/doctor).

## 4. Frontend (`index.html`)
Full rewrite: light theme (soft mint/teal palette instead of the old dark
glass UI) and a conversational chatbot instead of separate tab forms.
- **Identify screen** → **one-time intake modal** (only shown if
  `intake_completed` is false) → **chat**.
- Chat auto-routes: send a symptom message, and the assistant's reply
  already reflects whichever agent(s) the orchestrator invoked — you'll
  see a tag like "Symptom Checker → Medication" or
  "Symptom Checker → Appointment". If it routes to Appointment, your next
  chat message is treated as the date/time you want, and it books
  directly.
- A file-attach button next to the input sends an image or PDF straight
  to `/agent/lab-report` and explains it inline.
- A **Dashboard** tab (top right) holds appointment lookup and the
  doctor-schedule viewer, using "Acting As" the same way the old UI did.

The integrated Streamlit app renders this same `index.html` through
`clinical_chatbot_streamlit.py`. The chatbot's FastAPI app is called in-process
through a request bridge, so the main launcher starts the full UI without
requiring a separate chatbot server or startup step.

## 5. Real datasets, sourced not invented
- **`cdss_otc_medications.csv`** — 26 real, genuinely non-prescription
  active ingredients (paracetamol, ibuprofen, loratadine, clotrimazole,
  benzoyl peroxide, etc.), cross-checked against FDA-recognized OTC
  active ingredient categories. No dosage numbers — those belong on the
  actual product label. Run `python seed_sheets.py` again to reload both
  into your sheet.

## 6. Medication → still offers an appointment
`run_patient_symptom_flow()`'s medication branch (low/medium severity) now
always appends a follow-up: "Would you also like me to book an appointment
with a [specialist], just in case?" The frontend shows this as Yes/No
buttons under the medication card. Saying yes drops the patient straight
into the same date → consultation-mode → booked flow the high-severity
branch already used — so booking always ends the same way regardless of
which path got you there. Falls back to "General Physician" if the
knowledge base didn't name a specialist (typical for a low-severity case) —
**you'll need at least one doctor row with `specialty=General Physician`**
in your sheet for that fallback to actually find someone.

## 7. Consultation mode + expanded intake
Booking now asks Video / Phone / In-Clinic before confirming (stored as
`consultation_mode` on the appointment row). Intake picked up two new
question types (`select`, `yesno`) and three new questions inspired by a
real clinic's booking form: primary reason + category for the visit,
whether a prescription needs renewing, and whether a medical note is
needed — stored as `visit_reason` / `visit_category` /
`needs_prescription_renewal` / `needs_medical_note` on the patient row.

## Still worth doing yourself
- The Sheets read path is cached for ~4s per tab to stay under Sheets API
  rate limits; if you add more simultaneous users, consider batching
  reads further or moving to a proper DB once the sheet-based prototype
  has served its purpose.
- `seed_sheets.py` expects your existing CSVs plus the two new columns
136:  noted above — you'll need to add specialty/hours to doctor rows before running it.
