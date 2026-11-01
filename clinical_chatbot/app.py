import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator
from typing import Literal, Optional

import database as db
import sessions
from agents import (
    run_patient_symptom_flow, run_medication_agent, run_appointment_agent,
    run_appointment_direct, run_lab_report_agent, run_patient_medication_then_appointment,
    PROFILE_QUESTIONS, VISIT_QUESTIONS, submit_intake_answers,
    prewarm_agent_resources,
)
from mcp_tools import clear_appointments, get_patient_list, get_doctor_list, get_staff_list, get_doctor_schedule

# Same reasoning as in database.py: anchor to this file's folder so the
# frontend loads no matter what directory the app was launched from.
BASE_DIR = Path(__file__).resolve().parent

CHATBOT_BUILD_ID = "fast-start-v9"
app = FastAPI(title="CDSS Agent Orchestrator API")

@app.middleware("http")
async def disable_browser_cache(request, call_next):
    """Never let an old chatbot HTML/API response survive a project update."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-CDSS-Build"] = CHATBOT_BUILD_ID
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


_startup_tasks: set[asyncio.Task] = set()


def _track_startup_task(coro) -> None:
    task = asyncio.create_task(coro)
    _startup_tasks.add(task)
    task.add_done_callback(_startup_tasks.discard)


async def _initialize_schema_background() -> None:
    """Run Google Sheets schema verification off the event loop.

    The old startup hook performed several synchronous network calls before
    FastAPI could answer /api/health, which made the first chatbot open feel
    frozen and could block unrelated requests during slow Sheets responses.
    """
    log = logging.getLogger("cdss.app")
    if not await asyncio.to_thread(db.is_available):
        log.warning("Google Sheets is unavailable at startup: %s", db.DB_ERROR)
        return
    try:
        result = await asyncio.to_thread(db.ensure_schema)
        log.info("Database schema check: %s", result)
    except Exception:
        log.exception("Database schema initialization failed; interface will still start.")


async def _prewarm_agents_background() -> None:
    try:
        await prewarm_agent_resources()
        logging.getLogger("cdss.app").info("Clinical agent resources prewarmed.")
    except Exception:
        # Prewarming is an optimization only. The normal request path retries
        # initialization and retains the existing provider fallback behavior.
        logging.getLogger("cdss.app").exception(
            "Clinical agent prewarm failed; resources will initialize on first use."
        )


@app.on_event("startup")
async def initialize_data_layer():
    """Start quickly, then prepare slower reusable resources in background."""
    _track_startup_task(_initialize_schema_background())
    _track_startup_task(_prewarm_agents_background())


logger = logging.getLogger("cdss")


def _safe_error(user_message: str, exc: Exception) -> HTTPException:
    """Logs the full exception server-side and returns an HTTPException whose
    detail is a short, patient-safe message — never the raw provider/agent
    error text (which can be long, technical, and expose internal config)."""
    logger.error("%s", exc, exc_info=True)
    if getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return HTTPException(
            status_code=503,
            detail=(
                "The clinic's Google Sheets request limit was temporarily "
                "reached. Please wait a minute and try again."
            ),
        )
    return HTTPException(status_code=503, detail=user_message)




@app.get("/api/health")
async def health():
    """Liveness + actual data-layer readiness check used by the host."""
    sheets_ok = db.is_available(reconnect=False)
    return {
        "ok": bool(sheets_ok),
        "build": CHATBOT_BUILD_ID,
        "google_sheets": bool(sheets_ok),
        "sheet_error": None if sheets_ok else (db.DB_ERROR or "Google Sheets is unavailable"),
    }

# --- PATIENT IDENTIFICATION / INTAKE -----------------------------------------

class IdentifyRequest(BaseModel):
    patient_id: str
    name: Optional[str] = None  # used only if this patient_id doesn't exist yet
    session_id: Optional[str] = None  # pass a previously-issued session_id to resume it


@app.post("/api/identify")
async def identify_patient(request: IdentifyRequest):
    """Looks up a patient by ID, creating a bare record if this is a brand
    new patient_id. Frontend uses the response's `needs_intake` flag to
    decide whether to run the intake questionnaire before opening the chat.

    Also resumes the caller's session if session_id is still valid (not
    expired), or starts a fresh one otherwise — the returned session_id is
    what the frontend should send back on every /agent/chat and /api/intake
    call so conversation history and this-visit answers aren't lost between
    messages, the same way a ChatGPT/Claude conversation id works."""
    try:
        patient = await asyncio.to_thread(db.get_patient, request.patient_id)
        if not patient:
            await asyncio.to_thread(db.create_patient, request.patient_id, request.name or request.patient_id)
            patient = await asyncio.to_thread(db.get_patient, request.patient_id)
        needs_intake = str(patient.get("intake_completed", "false")).strip().lower() != "true"

        session_id = request.session_id
        existing_session = sessions.get_session(session_id) if session_id else None
        if (
            not existing_session
            or str(existing_session.get("patient_id", "")) != str(request.patient_id)
            or not sessions.touch(session_id)
        ):
            session_id = sessions.create_session(request.patient_id)

        return {"patient": patient, "needs_intake": needs_intake, "session_id": session_id}
    except Exception as e:
        raise _safe_error("Sorry, we couldn't reach patient records right now. Please try again in a moment.", e)


class IdentifyDoctorRequest(BaseModel):
    doctor_id: str
    name: Optional[str] = None  # used only if this doctor_id doesn't exist yet
    specialty: Optional[str] = None  # used only if this doctor_id doesn't exist yet
    working_hours_start: Optional[str] = None
    working_hours_end: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/api/identify-doctor")
async def identify_doctor(request: IdentifyDoctorRequest):
    """Same idea as /api/identify, but for the doctor side: looks a doctor
    up by ID in the 'doctors' sheet, creating a row (using the specialty /
    working hours SMART_CDSS collected at signup) the first time this
    doctor_id is seen. If that ID is already registered as a patient or
    staff member, this is refused rather than silently mixing roles."""
    try:
        existing_role = await asyncio.to_thread(db.resolve_role, request.doctor_id)
        if existing_role not in (None, "doctor"):
            raise HTTPException(
                status_code=409,
                detail=f"This ID is already registered as a {existing_role}, not a doctor.",
            )
        doctor = await asyncio.to_thread(db.get_doctor, request.doctor_id)
        if not doctor:
            doctor = await asyncio.to_thread(
                db.create_doctor,
                request.doctor_id, request.name or request.doctor_id,
                specialty=request.specialty or "",
                working_hours_start=request.working_hours_start or "",
                working_hours_end=request.working_hours_end or "",
            )

        session_id = request.session_id
        if not session_id or not sessions.touch(session_id):
            session_id = sessions.create_session(request.doctor_id)

        return {"doctor": doctor, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise _safe_error("Sorry, we couldn't reach doctor records right now. Please try again in a moment.", e)


@app.get("/api/intake-questions")
async def get_intake_questions(stage: Literal["profile", "visit"] = "profile", session_id: Optional[str] = None):
    """Filters out any question already answered earlier THIS session (even
    if the patient re-opens the intake modal, e.g. after a page refresh)
    so the same question is never asked twice in one conversation — this is
    what makes intake feel less repetitive, on top of intake_completed
    already skipping the one-time profile questions on later visits."""
    questions = PROFILE_QUESTIONS if stage == "profile" else VISIT_QUESTIONS
    cached = sessions.get_cached_answers(session_id) if session_id else {}
    if cached:
        questions = [q for q in questions if q["id"] not in cached]
    return {"questions": questions}


class IntakeSubmitRequest(BaseModel):
    patient_id: str
    answers: dict  # {question_id: answer}
    stage: Literal["profile", "visit"] = "profile"
    session_id: Optional[str] = None


@app.post("/api/intake")
async def submit_intake(request: IntakeSubmitRequest):
    try:
        result = await asyncio.to_thread(
            submit_intake_answers, request.patient_id, request.answers, request.stage
        )
    except Exception as e:
        raise _safe_error("Sorry, we couldn't save that right now. Please try again in a moment.", e)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    if request.session_id:
        for question_id, answer in request.answers.items():
            sessions.cache_answer(request.session_id, question_id, answer)
    return result


# --- UNIFIED CHAT (symptom checker -> auto-routes to medication/appointment) --

class ChatRequest(BaseModel):
    patient_id: str
    message: str
    clarifying_answer: Optional[str] = None
    clarify_round: int = 0
    session_id: Optional[str] = None
    # Set only when this call is resolving a route="intent_clarify" menu tap
    # ("symptom" / "appointment_management" / "other") — not a new message
    # the patient typed. See run_patient_symptom_flow's forced_intent param.
    forced_intent: Optional[str] = None


_chat_inflight: dict[tuple, asyncio.Task] = {}
_chat_inflight_lock = asyncio.Lock()


async def _process_chat_once(request: ChatRequest, session_id: str) -> dict:
    """Process one logical chat turn.

    The browser already disables duplicate submits while a request is active;
    this server-side single-flight guard also protects against double-clicks,
    browser retries, or two identical requests arriving concurrently.
    """
    if not request.forced_intent:
        sessions.append_turn(
            session_id, "user", request.clarifying_answer or request.message
        )

    history = sessions.get_history(session_id)
    res = await run_patient_symptom_flow(
        request.patient_id, request.message,
        request.clarifying_answer, request.clarify_round,
        session_history=history,
        forced_intent=request.forced_intent,
    )
    display_text = res.get("display_text")
    if display_text:
        sessions.append_turn(session_id, "assistant", display_text)
    return {"agent": "orchestrator", "output": res, "session_id": session_id}


@app.post("/agent/chat")
async def handle_chat(request: ChatRequest):
    session_id = request.session_id
    if not session_id or not sessions.touch(session_id):
        session_id = sessions.create_session(request.patient_id)

    request_key = (
        session_id,
        request.patient_id,
        request.message,
        request.clarifying_answer,
        request.clarify_round,
        request.forced_intent,
    )

    task = None
    try:
        async with _chat_inflight_lock:
            task = _chat_inflight.get(request_key)
            if task is None:
                task = asyncio.create_task(_process_chat_once(request, session_id))
                _chat_inflight[request_key] = task

        return await asyncio.shield(task)
    except Exception as e:
        raise _safe_error(
            "Sorry, I'm having trouble reaching the AI service right now. "
            "Please try again in a moment.",
            e,
        )
    finally:
        if task is not None:
            async with _chat_inflight_lock:
                current = _chat_inflight.get(request_key)
                if current is task and task.done():
                    _chat_inflight.pop(request_key, None)


# --- LAB REPORT UPLOAD --------------------------------------------------------

@app.post("/agent/lab-report")
async def handle_lab_report(patient_id: str = Form(...), file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        res = await run_lab_report_agent(patient_id, file_bytes, file.filename or "", file.content_type or "")
        return {"agent": "Lab Report", "output": res}
    except Exception as e:
        raise _safe_error("Sorry, I couldn't process that lab report right now. Please try again in a moment.", e)


# --- DIRECT AGENT ACCESS (used by the staff/doctor dashboard, and for
# resuming an appointment booking once the patient has given a date) --------

class MedicationRequest(BaseModel):
    patient_id: str
    complaint: str


@app.post("/agent/medication")
async def handle_medication(request: MedicationRequest):
    try:
        res = await run_medication_agent(request.patient_id, request.complaint)
        return {"agent": "Medication", "output": res}
    except Exception as e:
        raise _safe_error("Sorry, I'm having trouble reaching the AI service right now. Please try again in a moment.", e)


class MedicationThenAppointmentRequest(BaseModel):
    patient_id: str
    complaint: str
    specialist: str = "General Physician"
    session_id: Optional[str] = None


@app.post("/agent/medication-then-appointment")
async def handle_medication_then_appointment(request: MedicationThenAppointmentRequest):
    """Called when the patient explicitly says yes to medication recommendations
    after seeing their assessment. Runs the medication agent and returns a
    self_care result with the appointment offer."""
    session_id = request.session_id
    if not session_id or not sessions.touch(session_id):
        session_id = sessions.create_session(request.patient_id)
    try:
        res = await run_patient_medication_then_appointment(
            request.patient_id, request.complaint, request.specialist
        )
        return {"agent": "orchestrator", "output": res, "session_id": session_id}
    except Exception as e:
        raise _safe_error("Sorry, I'm having trouble reaching the AI service right now. Please try again in a moment.", e)


class AppointmentRequest(BaseModel):
    patient_id: str
    action: Literal["create", "lookup", "cancel", "update"]
    user_input: str
    date: str | None = None
    specialist: str | None = None
    doctor_id: str | None = None
    appointment_id: str | None = None
    reason: str | None = None
    consultation_mode: str | None = None
    contact_phone: str | None = None
    updates: dict | None = None
    requester_patient_id: str | None = None

    @model_validator(mode="after")
    def check_action_fields(self):
        action = self.action
        if action == "create" and (not self.date or (not self.doctor_id and not self.specialist)):
            raise ValueError("Create requires date and either doctor_id or specialist.")
        if action == "create" and self.consultation_mode in ("video", "phone") and not self.contact_phone:
            raise ValueError("Video/phone consultations require a contact_phone.")
        if action == "cancel" and not self.appointment_id:
            raise ValueError("Cancel requires appointment_id.")
        if action == "update":
            if not self.appointment_id:
                raise ValueError("Update requires appointment_id.")
            if not self.updates:
                raise ValueError("Update requires an updates object, e.g. {\"status\": \"completed\"}.")
            # For a mode change to video/phone via update, contact_phone goes
            # in the updates dict rather than as a top-level field.
        if not self.requester_patient_id:
            self.requester_patient_id = self.patient_id
        return self


@app.post("/agent/appointment")
async def handle_appointment(request: AppointmentRequest):
    try:
        # For create, update, and cancel: all required data (date, specialist,
        # mode, phone, appointment_id, updates) is already fully resolved by
        # the frontend widgets before this call is made. Routing these through
        # the LLM agent just to call manage_appointment is unnecessary and is
        # the source of timeout / LLM-unavailable errors. Call it directly.
        #
        # lookup is the only action that may need LLM reasoning (e.g. the
        # patient typed "show me my Tuesday appointment" and we need to
        # interpret that), so that one still goes through the agent.
        if request.action in ("create", "update", "cancel"):
            res = await run_appointment_direct(
                patient_id=request.patient_id,
                action=request.action,
                date=request.date,
                specialist=request.specialist,
                doctor_id=request.doctor_id,
                appointment_id=request.appointment_id,
                updates=request.updates,
                reason=request.reason,
                consultation_mode=request.consultation_mode,
                requester_patient_id=request.requester_patient_id,
                contact_phone=request.contact_phone,
            )
        else:
            res = await run_appointment_agent(
                patient_id=request.patient_id, user_input=request.user_input, action=request.action,
                date=request.date, specialist=request.specialist, doctor_id=request.doctor_id,
                appointment_id=request.appointment_id, requester_patient_id=request.requester_patient_id,
                updates=request.updates, reason=request.reason,
                consultation_mode=request.consultation_mode, contact_phone=request.contact_phone,
            )
        return {"agent": "Appointment", "output": res}
    except Exception as e:
        raise _safe_error("Sorry, I'm having trouble with appointments right now. Please try again in a moment.", e)


class ClearAppointmentsRequest(BaseModel):
    patient_id: str


@app.post("/agent/appointments/clear")
async def handle_clear_appointments(request: ClearAppointmentsRequest):
    try:
        res = clear_appointments(request.patient_id)
        return {"agent": "Appointment", "output": res}
    except Exception as e:
        raise _safe_error("Sorry, I'm having trouble with appointments right now. Please try again in a moment.", e)


@app.get("/api/doctor-schedule")
async def doctor_schedule(doctor_id: str, requester_patient_id: str):
    try:
        res = get_doctor_schedule(doctor_id, requester_patient_id)
    except Exception as e:
        raise _safe_error("Sorry, we couldn't load the schedule right now. Please try again in a moment.", e)
    if not res.get("authorized", True):
        raise HTTPException(status_code=403, detail=res.get("message"))
    res["appointment_count"] = len(res.get("appointments") or [])
    return res


@app.get("/api/appointments")
async def list_appointments(patient_id: str, active_only: bool = True):
    try:
        appts = await asyncio.to_thread(
            db.get_appointments, patient_id=patient_id, active_only=active_only
        )
        return {"appointments": appts}
    except Exception as e:
        raise _safe_error("Sorry, we couldn't load appointments right now. Please try again in a moment.", e)


@app.get("/api/patients")
async def list_patients():
    try:
        return {"patients": get_patient_list()}
    except Exception as e:
        raise _safe_error("Sorry, we couldn't load patient records right now. Please try again in a moment.", e)


@app.get("/api/patient")
async def get_patient_record(patient_id: str):
    try:
        patient = await asyncio.to_thread(db.get_patient, patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return {"patient": patient}
    except HTTPException:
        raise
    except Exception as e:
        raise _safe_error("Sorry, we couldn't load that patient's record right now. Please try again in a moment.", e)


@app.get("/api/doctors")
async def list_doctors():
    try:
        return {"doctors": get_doctor_list()}
    except Exception as e:
        raise _safe_error("Sorry, we couldn't load doctor records right now. Please try again in a moment.", e)


@app.get("/api/staff")
async def list_staff():
    try:
        return {"staff": get_staff_list()}
    except Exception as e:
        raise _safe_error("Sorry, we couldn't load staff records right now. Please try again in a moment.", e)


@app.get("/")
async def serve_frontend():
    return FileResponse(
        str(BASE_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-CDSS-Build": CHATBOT_BUILD_ID,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)