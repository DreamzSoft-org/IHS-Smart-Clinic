import asyncio
import base64
import json
import logging
import os
import random
import re
import sys
from datetime import datetime, timedelta
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from rapidfuzz import fuzz

from dateutil import parser as dateutil_parser

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

import database as db

# Only imported directly (not via MCP) because these are deliberately NOT
# left to the agent's discretion:
#   - check_emergency: safety-critical, must run before the agent even starts
#   - check_medication_conflicts: deterministic backstop even if the
#     medication agent already called it itself as a tool
from mcp_tools import check_emergency, check_medication_conflicts, manage_appointment

AGENTS_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger("cdss.agents")

# Load chatbot-local settings first, then fill missing values from the shared
# project-root .env. Existing process environment variables keep precedence.
load_dotenv(os.path.join(AGENTS_BASE_DIR, ".env"))
for _provider_setting in (
    "GROQ_API_KEY", "GROQ_MODEL",
    "CEREBRAS_API_KEY", "CEREBRAS_MODEL",
    "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_VISION_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL",
):
    if not os.environ.get(_provider_setting):
        os.environ.pop(_provider_setting, None)
load_dotenv(os.path.join(AGENTS_BASE_DIR, "..", ".env"))

# Generic, patient-facing message used whenever every configured LLM provider
# fails. The real exception chain (provider names, status codes, rate-limit
# headers, etc.) is logged server-side via logger.error(..., exc_info=True)
# instead of being shown to the patient — it's internal/operational detail,
# not something a patient can act on, and it can expose config info.
LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, I'm having trouble reaching the AI service right now. "
    "Please try again in a moment, or contact the clinic if this keeps happening."
)

# --- LLM PROVIDER SETUP -----------------------------------------------------
# Groq is primary. Fallback chain: Cerebras → Gemini → OpenRouter → Ollama.
# Cerebras sits right after Groq: same Llama models, very high free-tier
# rate limits, OpenAI-compatible API — no extra package required.
# Gemini is next but must use a non-thinking model; thinking models (2.5+)
# require round-tripping thought_signature fields that LangChain's OpenAI
# adapter doesn't preserve, causing 400 errors on multi-turn tool calls.
# OpenRouter and Ollama are last resorts.
LLM_PROVIDER = "groq"

LLM_FALLBACK_ORDER = [
    "groq",
    "cerebras",
    "gemini",
    "openrouter",
    "ollama",
]


# Kept modest so free/on-demand provider tiers (Groq OTPM, OpenRouter balance
# remnants) accept every request instead of returning hard rejections.
_LLM_MAX_OUTPUT_TOKENS = 900


def _normalize_ollama_base_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url[: -len("/v1")]
    return url


def _build_llm(provider: str):

    if provider == "ollama":
        raw_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL") or "qwen2.5:1.5b",
            base_url=_normalize_ollama_base_url(raw_url),
            temperature=0,
            validate_model_on_init=False,
        )
        
    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return None
        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=key,
            temperature=0,
            # Free/on-demand Groq tiers enforce an output-tokens-per-minute
            # (OTPM) cap (e.g. 1000) per request; keeping max_tokens safely
            # under that limit avoids hard 429 rejections.
            max_tokens=_LLM_MAX_OUTPUT_TOKENS,
        )

    if provider == "cerebras":
        key = os.environ.get("CEREBRAS_API_KEY")
        if not key:
            return None
        return ChatOpenAI(
            model=os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b"),
            api_key=key,
            base_url="https://api.cerebras.ai/v1",
            temperature=0,
            max_tokens=_LLM_MAX_OUTPUT_TOKENS,
        )

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        return ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            # Tight budget keeps requests inside the balance leftover on
            # free-tier keys ("can only afford ~950 tokens" 402 errors).
            max_tokens=_LLM_MAX_OUTPUT_TOKENS,
        )


    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return None
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            google_api_key=key,
            temperature=0,
            max_output_tokens=_LLM_MAX_OUTPUT_TOKENS,
        )
    return None


_llm_chain_cache: Optional[list] = None


def _get_llm_chain():
    global _llm_chain_cache
    if _llm_chain_cache is None:
        ordered_providers = [LLM_PROVIDER] + [p for p in LLM_FALLBACK_ORDER if p != LLM_PROVIDER]
        chain = []
        for provider in ordered_providers:
            instance = _build_llm(provider)
            if instance is not None:
                chain.append((provider, instance))
        if not chain:
            raise RuntimeError(
                "No LLM providers are configured. Set GROQ_API_KEY, "
                "OPENROUTER_API_KEY, or ensure Ollama is running (OLLAMA_BASE_URL)."
            )
        _llm_chain_cache = chain
    return _llm_chain_cache


# --- TOOLING -----------------------------------------------------------------
# The ReAct agents below use named tools (get_patient_record, manage_appointment,
# ...). Two transports can provide them:
#   1. MCP stdio subprocess (default) - used by the standalone FastAPI chatbot.
#   2. In-process direct wrappers - used when the host cannot spawn a reliable
#      stdio MCP server (e.g. Streamlit Community Cloud), enabled with the
#      CDSS_DIRECT_TOOLS environment variable. The underlying implementations
#      are the very same mcp_tools functions, so agent behavior is unchanged.
CDSS_DIRECT_TOOLS = os.environ.get("CDSS_DIRECT_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}

_MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_tools.py")

_mcp_client = MultiServerMCPClient({
    "cdss": {"command": sys.executable, "args": [_MCP_SERVER_PATH], "transport": "stdio"}
})

_all_mcp_tools_cache: Optional[list] = None
_all_mcp_tools_task: Optional[asyncio.Task] = None
_symptom_agents: dict = {}
_medication_agents: dict = {}
_appointment_agents: dict = {}


def _build_direct_tools() -> list:
    """Build the named tools the ReAct agents expect as in-process callables
    instead of MCP stdio stubs. Every wrapper delegates to the exact function
    the MCP server exports, keeping authorization and safety behavior intact."""
    from langchain_core.tools import tool
    from mcp_tools import (
        get_patient_record as _mcp_get_patient_record,
        check_medication_conflicts as _mcp_check_medication_conflicts,
        update_patient_profile as _mcp_update_patient_profile,
        manage_appointment as _mcp_manage_appointment,
        find_available_doctor as _mcp_find_available_doctor,
        get_doctor_schedule as _mcp_get_doctor_schedule,
    )

    @tool
    def get_patient_record(patient_id: str, requester_patient_id: str, fields: Optional[list] = None) -> dict:
        """Retrieves a patient's allergies, active medications, history, and today's visit context. requester_patient_id identifies who is asking; authorization is verified inside."""
        return _mcp_get_patient_record(patient_id, requester_patient_id, fields)

    @tool
    def check_medication_conflicts(patient_id: str, medication_names: list) -> dict:
        """Check candidate medication names against a patient's recorded allergies and current medications."""
        return _mcp_check_medication_conflicts(patient_id, medication_names)

    @tool
    def update_patient_profile(patient_id: str, requester_patient_id: str, field: str, add_values: list) -> dict:
        """Append new allergies or current medications to a patient's profile. field must be 'allergies' or 'current_medications'."""
        return _mcp_update_patient_profile(patient_id, requester_patient_id, field, add_values)

    @tool
    def manage_appointment(action: str, patient_id: str, requester_patient_id: str, date: Optional[str] = None, specialist: Optional[str] = None, doctor_id: Optional[str] = None, appointment_id: Optional[str] = None, updates: Optional[dict] = None, reason: Optional[str] = None, consultation_mode: Optional[str] = None, contact_phone: Optional[str] = None) -> dict:
        """Create, look up, update, or cancel an appointment. Role validation and schedule-conflict matching happen inside; requester_patient_id never grants elevation by itself."""
        return _mcp_manage_appointment(action=action, patient_id=patient_id, requester_patient_id=requester_patient_id, date=date, specialist=specialist, doctor_id=doctor_id, appointment_id=appointment_id, updates=updates, reason=reason, consultation_mode=consultation_mode, contact_phone=contact_phone)

    @tool
    def find_available_doctor(specialty: str, date: str, duration_minutes: int = 30) -> dict:
        """Find an available doctor matching a specialty for the given date."""
        return _mcp_find_available_doctor(specialty, date, duration_minutes)

    @tool
    def get_doctor_schedule(doctor_id: str, requester_patient_id: str) -> dict:
        """Get a doctor's schedule for upcoming appointments."""
        return _mcp_get_doctor_schedule(doctor_id, requester_patient_id)

    return [
        get_patient_record,
        check_medication_conflicts,
        update_patient_profile,
        manage_appointment,
        find_available_doctor,
        get_doctor_schedule,
    ]


async def _get_all_mcp_tools():
    """Load tools once, even when startup prewarming and a user request
    arrive at the same time."""
    global _all_mcp_tools_cache, _all_mcp_tools_task
    if _all_mcp_tools_cache is not None:
        return _all_mcp_tools_cache

    if CDSS_DIRECT_TOOLS:
        _all_mcp_tools_cache = _build_direct_tools()
        return _all_mcp_tools_cache

    if _all_mcp_tools_task is None:
        _all_mcp_tools_task = asyncio.create_task(_mcp_client.get_tools())

    try:
        _all_mcp_tools_cache = await _all_mcp_tools_task
        return _all_mcp_tools_cache
    except Exception:
        _all_mcp_tools_task = None
        raise


async def prewarm_agent_resources() -> None:
    """Initialize reusable provider clients and MCP tools in the background.

    This removes one-time orchestration startup work from the first patient
    message without changing providers, prompts, or agent behavior.
    """
    _get_llm_chain()
    await _get_all_mcp_tools()


def _select_tools(all_tools, names: set):
    selected = [t for t in all_tools if t.name in names]
    found_names = {t.name for t in selected}
    missing = names - found_names
    if missing:
        raise RuntimeError(
            f"Expected MCP tools not found on server: {sorted(missing)}. "
            f"Available: {sorted(t.name for t in all_tools)}"
        )
    return selected


# --- INTAKE QUESTIONNAIRE ----------------------------------------------------
# PROFILE_QUESTIONS: asked once ever, when a patient first opens the chatbot
# (intake_completed is false/blank on their sheet row). Stable facts about
# the person, not this particular visit.
PROFILE_QUESTIONS = [
    {"id": "age", "field": "age", "question": "How old are you?", "type": "number"},
    {"id": "gender", "field": "gender", "question": "What is your gender?", "type": "text"},
    {"id": "current_medications", "field": "current_medications", "question": "Please list all of your current medications, or say 'none'.", "type": "list"},
    {"id": "medical_conditions_yn", "field": None, "question": "Do you have any medical conditions?", "type": "yesno"},
    {"id": "medical_history", "field": "medical_history", "question": "Please list your medical conditions.", "type": "list",
     "depends_on": {"id": "medical_conditions_yn", "equals": "Yes"}},
    {"id": "medication_allergies_yn", "field": None, "question": "Do you have any known allergies to medications?", "type": "yesno"},
    {"id": "allergies", "field": "allergies", "question": "Please list your medication allergies.", "type": "list",
     "depends_on": {"id": "medication_allergies_yn", "equals": "Yes"}},
    {"id": "regular_patient", "field": "regular_patient", "question": "Are you a regular patient of our medical group?", "type": "yesno"},
]

# VISIT_QUESTIONS: asked at the start of EVERY session, regardless of
# intake_completed — these describe today's visit, not the person. They give
# the symptom checker immediate clinical context (onset, severity, character)
# so it can skip generic clarifying questions and get straight to assessment.
VISIT_QUESTIONS = [
    {
        "id": "visit_reason",
        "field": "visit_reason",
        "question": "What's the primary reason for today's visit? Describe your main symptom or concern.",
        "type": "text",
    },
    {
        "id": "symptom_onset",
        "field": "symptom_onset",
        "question": "When did it start, and did it come on suddenly or gradually?",
        "type": "select",
        "options": [
            "Started today — suddenly",
            "Started today — gradually",
            "Started 2–3 days ago",
            "Started 4–7 days ago",
            "Going on for more than a week",
            "Recurring / comes and goes",
        ],
    },
    {
        "id": "symptom_severity",
        "field": "symptom_severity",
        "question": "How would you rate the severity right now?",
        "type": "select",
        "options": [
            "Mild — noticeable but not stopping me from daily activities",
            "Moderate — affecting my daily activities",
            "Severe — I can't do normal activities",
            "Very severe — I need help immediately",
        ],
    },
    {
        "id": "needs_prescription_renewal",
        "field": "needs_prescription_renewal",
        "question": "Do you need any prescriptions renewed today?",
        "type": "yesno",
    },
    {
        "id": "needs_medical_note",
        "field": "needs_medical_note",
        "question": "Do you need a medical note for work or school today?",
        "type": "yesno",
    },
]

# Kept for backwards compatibility with any caller still importing the old
# combined list.
INTAKE_QUESTIONS = PROFILE_QUESTIONS + VISIT_QUESTIONS


def _questions_for_stage(stage: str):
    return PROFILE_QUESTIONS if stage == "profile" else VISIT_QUESTIONS


def submit_intake_answers(patient_id: str, answers: dict, stage: str = "profile") -> dict:
    """Deterministically maps intake answers onto the patient's sheet row.
    stage="profile" also flips intake_completed so the one-time questions
    never show again; stage="visit" never touches that flag, so visit
    questions keep showing on every session."""
    updates = {}
    for q in _questions_for_stage(stage):
        if q.get("field") is None:
            continue  # gate-only question (e.g. the yes/no before a list) — nothing to persist
        raw = answers.get(q["id"])
        if raw is None:
            continue
        if q["type"] == "list":
            if isinstance(raw, list):
                updates[q["field"]] = ", ".join(str(x).strip() for x in raw if str(x).strip())
            else:
                cleaned = "" if str(raw).strip().lower() in ("none", "n/a", "") else str(raw).strip()
                updates[q["field"]] = cleaned
        else:
            # select / yesno / text / number all store as the plain answer string
            updates[q["field"]] = raw
    if stage == "profile":
        updates["intake_completed"] = "true"

    ok = db.update_patient(patient_id, updates)
    if not ok:
        return {"error": "Patient not found; cannot save intake answers."}
    return {"ok": True, "saved": updates}


# --- SYSTEM PROMPTS -----------------------------------------------------------

SYMPTOM_CHECKER_SYSTEM_PROMPT = """You are a Symptom Checker agent in a research-prototype CDSS. You behave like an experienced nurse triage line, NOT a diagnosis bot — you are NOT a licensed clinician and must never present output as, or imply, a diagnosis.

You have a tool to look up a patient's record (allergies, current medications, history, and today's visit context). Use it when the patient's medical history, allergies, or visit context could change your assessment.

Note: get_patient_record requires both patient_id and requester_patient_id. In this context the patient is looking up their own record, so pass the same value for both.

CONTEXT AWARENESS — the user message you receive will begin with a "Today's check-in context" block. That block contains fields the patient already answered during check-in minutes ago (visit_reason, symptom_onset, symptom_severity, needs_prescription_renewal, needs_medical_note). These are KNOWN FACTS — treat them exactly as if the patient just told you in conversation. NEVER ask the patient to repeat or re-confirm anything already present in that context block, including onset, duration, severity, or how long it has been going on.

STRICT RULE — never diagnose THIS patient: do not guess, mention, rank, or hint at what specific disease or condition this patient individually has, anywhere in your output. That said, general, widely-published patient-education content about the symptom itself — the kind found on a Mayo Clinic or NHS patient information page — is expected and encouraged, not a diagnosis.

CRITICAL — never withhold general guidance behind a question: even on the very first, briefest message, give the patient real, useful, evidence-based guidance right away — self_care_advice and warning_signs should describe what's generally true and generally safe for that symptom. A clarifying question is something you ask IN ADDITION to that guidance, never a replacement for it.

CRITICAL — always terminate in a usable state: every response must include general guidance (self_care_advice/warning_signs) for a recognizable symptom, optionally topped off with exactly one clarifying question if more detail would meaningfully change the severity or appointment recommendation.

Your job, in order:
1. Read the complaint AND the check-in context block carefully. The context block gives you onset, severity, and duration the patient already filled in — use them directly and do NOT ask about them again. Only ask a clarifying question if there is still a genuinely missing piece (e.g. associated symptoms, what makes it worse, whether it has happened before) that would materially change your severity verdict or appointment recommendation. If onset AND severity are already known from the context, you almost certainly have enough to commit to a full assessment — do so.
2. Once you have enough to go on — assess severity as exactly one of:
   - "mild": self-limiting, safely manageable at home.
   - "urgent": warrants a clinical evaluation soon (days), but is not an emergency.
   - "emergency": needs immediate emergency care right now. Use this rarely.
3. Write assessment_summary as a short, warm, plain-language explanation directed at the patient. ALWAYS open by naming the specific symptom(s) they reported (e.g. "For the nausea you mentioned..."), then explain what's generally going on symptom-wise. Keep the focus on that symptom — only weave in other context (like a prescription renewal) if it's directly relevant to explaining the symptom itself, not as the main subject of the summary.
4. Give self_care_advice: 2-4 concrete, evidence-based, non-drug steps.
5. Set otc_relevant true whenever an OTC medication could reasonably help this specific complaint.
6. Give warning_signs: 2-4 specific, concrete red-flag symptoms.
7. Set appointment_recommended true when the complaint genuinely warrants clinical evaluation.
8. If the conversation has already reached the maximum number of clarifying rounds, do NOT ask another question — commit immediately to your best-effort complete assessment.

Respond with ONLY a valid JSON object (no other text, no markdown fences) matching this schema:
{
  "severity": "mild" | "urgent" | "emergency" | null,
  "assessment_summary": "string or null",
  "self_care_advice": ["string"],
  "otc_relevant": true | false,
  "warning_signs": ["string"],
  "appointment_recommended": true | false,
  "recommended_specialist": "string or null",
  "clarifying_question": "string or null"
}"""

MEDICATION_SYSTEM_PROMPT = """You are a Self-Care & Medication Agent in a research-prototype CDSS. Given a patient's complaint, you reason freely from your own general clinical knowledge — there is no fixed database of conditions or medications behind you, so use your judgment the way a well-informed clinical reference (e.g. Mayo Clinic, NHS) would, and be explicit that this is general guidance, not a diagnosis.

You have tools to: look up a patient's record (allergies, current medications, visit context), and check candidate medication names you come up with against that patient's record for conflicts. Decide for yourself which to call and when.

Note: get_patient_record requires both patient_id and requester_patient_id. In this context the patient is looking up their own record, so pass the same value for both. Their record also includes needs_prescription_renewal, captured earlier today when they checked in — if it's set to yes, treat that as a signal they're likely asking about a maintenance medication they're already on, not a new OTC need; a prescription renewal itself is not something you can act on (you only handle OTC/self-care), so route it toward the appointment flow rather than proposing an OTC substitute.

New allergy/medication disclosed mid-conversation: if the patient's message mentions an allergy or a medication they're taking that is NOT already in the record you looked up, call update_patient_profile once to add it (field="allergies" or field="current_medications", add_values=[the new item(s)] — this only appends, it never erases anything already on file. Do this in addition to your normal guidance, not instead of it, and do NOT ask permission first — just add it, and include a short confirmation as the first item in "self_care" (e.g. "Noted — I've added penicillin to your allergy record."). This is the ONLY case where you call update_patient_profile; never use it to record anything else.

Your job, for every complaint that isn't an emergency:
1. Give concrete, practical self-care steps first (rest, hydration, positioning, home remedies, what to monitor) — this is useful regardless of whether OTC medication is appropriate.
2. Decide whether an OTC medication is appropriate for this complaint. If yes, name specific common OTC options (e.g. "Paracetamol (Acetaminophen)", "Ibuprofen") and note any standard contraindications for each (e.g. NSAID + stomach ulcers/kidney disease/allergy) directly in your reasoning, in plain language, the way a pharmacist would caution a patient. If OTC medication genuinely isn't appropriate or safe here, say so plainly rather than forcing a suggestion.
3. Call check_medication_conflicts on every named OTC option against this specific patient's record before finalizing it — this checks their actual recorded allergies and current medications, on top of the general contraindications you already reasoned about.
4. List the red-flag warning signs that mean this patient should seek urgent/emergency care instead of continuing self-care, specific to this complaint (not generic boilerplate).
5. Set "escalate": true only when the complaint itself warrants clinical review beyond self-care (persistent fever, allergic reaction, pregnancy-related concern, chest symptoms, heavy/prolonged bleeding, recurrent episodes, or anything ambiguous/off-topic) — never just because you're unsure of an exact drug name.

Respond with ONLY a valid JSON object (no other text, no markdown fences) matching this schema:
{
  "self_care": ["string"],
  "otc_appropriate": true | false,
  "otc_suggestions": [{"option": "string", "note": "string - plain-language contraindication/caution note"}],
  "red_flags": ["string"],
  "interaction_warnings": ["string"],
  "reasoning": "string",
  "escalate": true | false,
  "escalation_reason": "string or null"
}

Rules:
- NEVER recommend prescription drugs, controlled substances, or specific dosages — refer to label instructions instead.
- NEVER include an item in "otc_suggestions" without having called check_medication_conflicts on it first.
- Always populate "red_flags" with at least one concrete warning sign when the complaint is a physical symptom, even in a low-severity case — self-care advice without knowing when to stop self-treating is incomplete.

Note: every suggestion you make is independently re-checked in code after you respond, regardless of whether you checked it yourself. That backstop exists in case you miss something — it is not a reason to skip the check.
"""

APPOINTMENT_SYSTEM_PROMPT = """You are an Appointment Agent in a research-prototype CDSS.

**DOCTOR SELECTION RULE**: Appointments may be booked with ANY doctor account registered in SMART CDSS.
When a doctor_id is supplied by the appointment picker, pass that exact doctor_id to manage_appointment unchanged. Do not replace the selected doctor with a General Physician or a different specialty. If no doctor_id is supplied, specialty matching may be used.

You have tools to look up a patient's record, find an available doctor matching a specialty at a given date/time, view a doctor's own schedule, and manage appointments (create/lookup/update/cancel). Authorization is verified inside these tools themselves, using the requester's real recorded role — you cannot grant yourself or anyone else elevated access by asserting a role, and you should not try.

The Date given below (for action="create") has already been resolved to exact ISO8601 by deterministic code before you ever saw it — pass it to manage_appointment byte-for-byte as given. Never reformat it, reinterpret it, or invent/guess a date yourself under any circumstance.

Note: if you call get_patient_record, pass the requester patient ID given below as requester_patient_id — do not substitute the patient_id whose appointment this concerns unless they are the same person. Their record also includes needs_medical_note and needs_prescription_renewal, captured earlier today when they checked in — if a reason wasn't explicitly given for this appointment, these can help you write a sensible default reason (e.g. "prescription renewal" or "medical note request").

Your job:
1. The requested action is given to you explicitly below as "Requested action" — always use that value; do not re-infer it from the user's wording.
2. For action="create", call manage_appointment with the doctor_id when one is given, plus specialist, date, reason, and consultation_mode — doctor_id must be passed unchanged so the exact registered doctor selected by the user is booked. If doctor_id is absent, specialty matching is used. It checks the schedule for conflicts, so you do not need to call find_available_doctor yourself first unless you want to preview options. If consultation_mode is "video" or "phone", also pass the given Contact phone through as the tool's contact_phone argument — the patient can't be reached for that consult without it.
3. For action="lookup", you MUST call manage_appointment with action="lookup", patient_id, and requester_patient_id set from the values given below — do not skip the tool call and do not answer from memory or from anything earlier in the conversation. Pass no date/specialist filters unless the user explicitly asked to narrow by one. The tool returns every matching appointment; do not drop, summarize away, or invent any of them.
4. For action="cancel", call manage_appointment with action="cancel", the given appointment_id (or patient_id/requester_patient_id if no appointment_id was given, so the tool can resolve it), patient_id, and requester_patient_id.
5. For action="update", pass the given Updates object as-is as the `updates` argument — only status/date/specialist changes belong there, and the tool itself will reject the call if the requester isn't staff/doctor.
6. Every action above requires an actual manage_appointment tool call — never fabricate a result JSON yourself, even if you believe you already know the answer.
7. If a tool reports the request is unauthorized, invalid, or that there's no availability, report that back as-is — do not retry with different claims, and surface any suggested alternatives verbatim.

After calling the appointment-management tool, respond with ONLY the exact JSON object it returned. Do not add commentary, do not reformat it, no markdown fences.
"""

LAB_REPORT_SYSTEM_PROMPT = """You are a Lab Report Explainer in a research-prototype CDSS, helping a patient who finds their lab report hard to understand.

You will be given the report's content (as text and/or as an image). Explain it in plain, everyday language:
1. Go test-by-test: what it measures, the patient's value, whether it's within the typical reference range, and what "high"/"low"/"normal" practically tends to mean for that test.
2. Use a calm, reassuring, non-alarmist tone. Do not diagnose a condition from the results.
3. Clearly flag anything meaningfully outside range as worth discussing with their doctor — but say so factually, not urgently, unless a value is dangerously abnormal.
4. If the image/text is unreadable or incomplete, say exactly which part you couldn't read rather than guessing values.
5. End with one short reminder that this explanation doesn't replace a clinician's interpretation.

Respond with ONLY a valid JSON object (no other text, no markdown fences) matching this schema:
{
  "summary": "string - one or two sentence plain-language overview",
  "tests": [{"name": "string", "value": "string", "reference_range": "string or null", "flag": "normal" | "high" | "low" | "unclear", "explanation": "string"}],
  "flagged_for_doctor": ["string"],
  "unreadable_sections": ["string"]
}"""


async def _get_symptom_agent(provider: str, llm_instance):
    if provider not in _symptom_agents:
        tools = _select_tools(await _get_all_mcp_tools(), {"get_patient_record"})
        _symptom_agents[provider] = create_react_agent(llm_instance, tools=tools, prompt=SYMPTOM_CHECKER_SYSTEM_PROMPT)
    return _symptom_agents[provider]


async def _get_medication_agent(provider: str, llm_instance):
    if provider not in _medication_agents:
        tools = _select_tools(
            await _get_all_mcp_tools(),
            {"get_patient_record", "check_medication_conflicts", "update_patient_profile"},
        )
        _medication_agents[provider] = create_react_agent(llm_instance, tools=tools, prompt=MEDICATION_SYSTEM_PROMPT)
    return _medication_agents[provider]


async def _get_appointment_agent(provider: str, llm_instance):
    if provider not in _appointment_agents:
        tools = _select_tools(
            await _get_all_mcp_tools(),
            {"get_patient_record", "manage_appointment", "find_available_doctor", "get_doctor_schedule"},
        )
        _appointment_agents[provider] = create_react_agent(llm_instance, tools=tools, prompt=APPOINTMENT_SYSTEM_PROMPT)
    return _appointment_agents[provider]


AGENT_TIMEOUT_SECONDS = 180

# Hard ceiling on clarifying-question rounds. The system prompt asks the
# model to stop on its own once this is reached, but that's a request, not
# a guarantee — this is the actual enforcement. Without it, a model that
# never feels "confident enough" (small local models especially) can loop
# clarifying questions indefinitely instead of ever reaching a diagnosis or
# an appointment.
MAX_CLARIFY_ROUNDS = 2

# Same rationale as MAX_CLARIFY_ROUNDS above, mirrored at the other end of the
# conversation: the system prompt already tells the model to ask exactly one
# clarifying question when the complaint is too brief to safely assess
# (step 1), but a small local model does not reliably honor that — it will
# sometimes skip straight to a full assessment off a bare symptom label like
# "nose bleeding" or "headache", with no onset/duration/severity/context, and
# paper over the missing information with a vague, safety-padded severity
# instead of actually asking. That produces exactly the failure this
# threshold exists to catch: a severity verdict the model had no real basis
# for. This is a blunt word-count heuristic, not real NLP — it only needs to
# catch the common case of a short symptom label with no detail attached,
# not correctly judge every possible input.
MIN_WORDS_FOR_ASSESSMENT = 4

GENERIC_BREVITY_CLARIFYING_QUESTION = (
    "Can you tell me a bit more about that — how long has it been going on, "
    "how severe or frequent is it, and is there anything else notable (like a "
    "recent injury, medication you're taking, or other symptoms alongside it)?"
)


def _too_brief_for_assessment(user_input: str) -> bool:
    return len(user_input.strip().split()) < MIN_WORDS_FOR_ASSESSMENT


# How many times to retry a single provider when the model returns an
# empty response (no text, no tool call). Ollama/small local models do
# this occasionally under load — a second attempt almost always succeeds.
_EMPTY_OUTPUT_MAX_RETRIES = 3

# The exact error text LangGraph raises when a model turn produces nothing.
_EMPTY_OUTPUT_MARKER = "model output must contain either output text or tool calls"


async def _run_with_fallback(agent_getter, user_message: str):
    failures: list[str] = []
    for provider, llm_instance in _get_llm_chain():
        last_exc: Optional[Exception] = None
        for attempt in range(1, _EMPTY_OUTPUT_MAX_RETRIES + 1):
            try:
                agent = await agent_getter(provider, llm_instance)
                result = await asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": [{"role": "user", "content": user_message}]},
                        config={"recursion_limit": 15},
                    ),
                    timeout=AGENT_TIMEOUT_SECONDS,
                )
                if not isinstance(result, dict):
                    failures.append(f"{provider}: returned result of type {type(result).__name__}")
                    last_exc = None
                    break  # not a retryable issue — move to next provider
                messages = result.get("messages")
                if not isinstance(messages, list):
                    failures.append(f"{provider}: response missing 'messages' list")
                    last_exc = None
                    break
                if not messages:
                    failures.append(f"{provider}: response contained no messages")
                    last_exc = None
                    break
                final_message = messages[-1]
                final_text = getattr(final_message, "content", None)
                if final_text is None:
                    final_text = str(final_message)
                parsed = _extract_json_object(final_text)
                if parsed is None:
                    # This provider responded but didn't return valid JSON —
                    # small local models do this more often than hosted ones.
                    # Don't surface a parse error to the patient when another
                    # configured provider might just work — try the next one.
                    failures.append(f"{provider}: returned non-JSON output")
                    last_exc = None
                    break
                return parsed, provider
            except asyncio.TimeoutError:
                failures.append(f"{provider}: timed out after {AGENT_TIMEOUT_SECONDS}s")
                last_exc = None
                break  # timeout is not a retryable empty-output issue
            except Exception as exc:
                exc_str = str(exc)
                if _EMPTY_OUTPUT_MARKER in exc_str:
                    # Ollama returned an empty turn — wait briefly and retry
                    last_exc = exc
                    logger.warning(
                        "%s: empty model output on attempt %d/%d — retrying",
                        provider, attempt, _EMPTY_OUTPUT_MAX_RETRIES,
                    )
                    await asyncio.sleep(0.5 * attempt)  # brief back-off: 0.5s, 1s, 1.5s
                    continue
                # Any other exception: log and move to next provider
                failures.append(f"{provider}: {exc}")
                last_exc = None
                break
        else:
            # All retries exhausted for this provider on empty-output errors
            if last_exc is not None:
                failures.append(
                    f"{provider}: model returned empty output on all {_EMPTY_OUTPUT_MAX_RETRIES} attempts"
                )
    raise RuntimeError(
        "All configured LLM providers failed:\n" + "\n".join(f"- {f}" for f in failures)
        if failures else "No LLM providers available"
    )


# --- HELPER FUNCTIONS ---

def _extract_json_object(content: str) -> Optional[dict]:
    """
    Best-effort extraction of a JSON object from raw model output. Hosted
    frontier models reliably return exactly what's asked for; a local
    7B-class model (Qwen included) is noticeably more prone to wrapping the
    JSON in a <think>...</think> reasoning block, a leading sentence, or a
    code fence despite explicit instructions not to. Returns None (not a
    dict) on failure so the caller can decide to retry another provider
    rather than surface a raw parse error.
    """
    text = content.strip()

    # Strip a <think>...</think> reasoning block some local models emit
    # before the actual answer, even when told to output only JSON.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    # Strip a markdown code fence if present, wherever it appears (a model
    # can still lead with a short sentence before ```json despite instructions).
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: extract the outermost {...} or [...] and try again,
    # in case the model added prose around an otherwise-valid JSON object.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


def parse_llm_response(response):
    content = response.content if hasattr(response, 'content') else str(response)
    parsed = _extract_json_object(content)
    if parsed is None:
        return {"error": "Failed to parse JSON response from the model", "raw_response": content}
    return parsed


def render_symptom_output(parsed: dict) -> str:
    if parsed.get("error"):
        return f"Model error: {parsed['error']}"
    if parsed.get("emergency"):
        return parsed.get("message", "Emergency flagged.")

    lines = []
    summary = parsed.get("assessment_summary")
    if summary:
        lines.append(summary)

    self_care = parsed.get("self_care_advice") or []
    if self_care:
        lines.append("")
        lines.append("What you can do:")
        for step in self_care:
            lines.append(f"• {step}")

    warning_signs = parsed.get("warning_signs") or []
    if warning_signs:
        lines.append("")
        lines.append("Seek care sooner if you notice:")
        for sign in warning_signs:
            lines.append(f"• {sign}")

    if not lines:
        lines.append("Thanks for sharing that.")

    # Note: the clarifying question (if any) is deliberately NOT appended
    # here. The only caller of run_symptom_checker is the orchestrator
    # (run_patient_symptom_flow below), which also merges in OTC/self-care
    # guidance from the medication agent — the question needs to land after
    # THAT, not right after this text, so it stays out here and gets
    # appended once, last, by the orchestrator.
    return "\n".join(lines)


def render_medication_output(parsed: dict) -> str:
    if parsed.get("error"):
        return f"Model error: {parsed['error']}"

    self_care = parsed.get("self_care") or []
    suggestions = parsed.get("otc_suggestions") or []
    red_flags = parsed.get("red_flags") or []
    warnings = parsed.get("interaction_warnings") or []
    escalate = parsed.get("escalate", False)
    lines = []

    if self_care:
        lines.append("Self-care:")
        for step in self_care:
            lines.append(f"• {step}")

    if escalate:
        lines.append("")
        lines.append("No OTC suggestions — this warrants clinical review rather than self-care alone.")
        if parsed.get("escalation_reason"):
            lines.append(parsed["escalation_reason"])
    elif suggestions:
        lines.append("")
        lines.append("Over-the-counter medication (if appropriate):")
        for suggestion in suggestions:
            opt = suggestion.get("option", "Unknown")
            note = suggestion.get("note", "")
            lines.append(f"• {opt}" + (f" — {note}" if note else ""))
    elif not parsed.get("otc_appropriate", True):
        lines.append("")
        lines.append("OTC medication isn't recommended for this — self-care above is the main approach for now.")

    if warnings:
        lines.append("")
        lines.append("Interaction warnings:")
        for warning in warnings:
            lines.append(f"• {warning}")

    if red_flags:
        lines.append("")
        lines.append("Seek urgent medical care immediately if:")
        for flag in red_flags:
            lines.append(f"• {flag}")

    if not lines:
        lines.append("No guidance could be generated.")
        if parsed.get("reasoning"):
            lines.append(parsed["reasoning"])

    return "\n".join(lines)


def render_appointment_output(parsed: dict) -> str:
    if parsed.get("error"):
        return f"Error: {parsed['error']}"
    if not parsed.get("authorized", True):
        return f"Unauthorized: {parsed.get('message', 'Request denied')}"

    status = parsed.get("status")
    if status == "created":
        details = parsed.get("details", {})
        return (
            f"Appointment created for {details.get('date')} with {details.get('assigned_doctor') or details.get('specialist')} "
            f"({details.get('specialist')}) — ID: {details.get('appointment_id')}"
        )
    if status == "already_exists":
        details = parsed.get("details", {})
        return f"Appointment already exists for {details.get('date')} with {details.get('specialist')} (ID: {details.get('appointment_id')})"
    if status == "no_availability":
        lines = [parsed.get("message", "No availability.")]
        slots = parsed.get("suggested_slots") or []
        if slots:
            lines.append("Next available slots:")
            for slot in slots:
                lines.append(f"• {slot.get('date')} — {slot.get('doctor_name')}")
        return "\n".join(lines)
    if status == "lookup_completed":
        appointments = parsed.get("appointments", [])
        if not appointments:
            return "No appointments found."
        lines = ["Appointments:"]
        for appt in appointments:
            lines.append(f"• {appt.get('date')} — {appt.get('specialist')} ({appt.get('status')}) [ID: {appt.get('appointment_id')}]")
        return "\n".join(lines)
    if status == "updated":
        return "Appointment updated successfully."
    if status == "cancelled":
        return "Appointment cancelled successfully."
    if status == "not_found":
        return "Appointment not found."

    return parsed.get("message", "Appointment action completed.")


def build_response_ui(agent_name: str, parsed: dict) -> dict:
    if parsed.get("display_text"):
        return parsed
    rendered = ""
    if agent_name == "symptom":
        rendered = render_symptom_output(parsed)
    elif agent_name == "medication":
        rendered = render_medication_output(parsed)
    elif agent_name == "appointment":
        rendered = render_appointment_output(parsed)
    parsed["display_text"] = rendered
    return parsed


def _visit_context_line(patient_id: str) -> str:
    """Deterministically folds today's check-in answers (visit_reason,
    needs_prescription_renewal, needs_medical_note, ...) into the prompt,
    instead of relying on the model to remember to call get_patient_record
    on its own. A small/local model in a ReAct loop is inconsistent about
    invoking optional tools — without this, a patient's own answers from
    minutes earlier can appear to be "forgotten" simply because the model
    never looked them up. This is a summary for context; get_patient_record
    is still available as a tool for anything else in the record (allergies,
    medications, history)."""
    try:
        patient = db.get_patient(patient_id)
    except Exception:
        logger.warning("Could not fetch patient record for visit context (patient_id=%s)", patient_id, exc_info=True)
        return ""
    if not patient:
        return ""

    parts = []
    if patient.get("visit_reason"):
        parts.append(f"stated reason for today's visit: \"{patient['visit_reason']}\"")
    if patient.get("symptom_onset"):
        parts.append(f"symptom onset: {patient['symptom_onset']}")
    if patient.get("symptom_severity"):
        parts.append(f"current severity: {patient['symptom_severity']}")
    if str(patient.get("needs_prescription_renewal", "")).strip().lower() == "yes":
        parts.append("patient indicated at check-in they need a PRESCRIPTION RENEWAL today")
    if str(patient.get("needs_medical_note", "")).strip().lower() == "yes":
        parts.append("patient indicated at check-in they need a MEDICAL NOTE for work/school today")
    if not parts:
        return ""

    return (
        "Today's check-in context (already provided by the patient minutes ago — "
        "treat as known, do NOT ask the patient to repeat or re-confirm any of these): "
        + "; ".join(parts) + ". "
        "Because onset and severity are already known, skip any clarifying question about "
        "duration, how long it has been going on, or how severe it is — you have that information.\n"
    )


# --- AGENT RUNNERS ---

async def run_symptom_checker(patient_id: str, user_input: str, clarifying_answer: Optional[str] = None, clarify_round: int = 0):
    combined_text = user_input if not clarifying_answer else f"{user_input}\n{clarifying_answer}"
    emergency_check = check_emergency(combined_text)
    if emergency_check.get("emergency"):
        emergency_check["display_text"] = emergency_check.get("message")
        return emergency_check

    user_message = f"{_visit_context_line(patient_id)}Patient ID: {patient_id}\nRequester patient ID (self): {patient_id}\nPatient input: {user_input}"
    if clarifying_answer:
        user_message += f"\nClarifying answer: {clarifying_answer}"
    user_message += (
        f"\nClarifying rounds so far this conversation: {clarify_round} "
        f"(maximum allowed: {MAX_CLARIFY_ROUNDS} — you MUST NOT ask another clarifying question once this maximum is reached)."
    )

    try:
        parsed, provider_used = await _run_with_fallback(_get_symptom_agent, user_message)
        parsed["_llm_provider_used"] = provider_used
    except Exception as exc:
        logger.error("Symptom checker agent failed on all configured LLM providers: %s", exc, exc_info=True)
        error_payload = {
            "error": "symptom_checker_llm_unavailable",
            "display_text": LLM_UNAVAILABLE_MESSAGE,
            "severity": None, "clarifying_question": None, "clarifying_required": False,
        }
        return build_response_ui("symptom", error_payload)

    parsed["clarifying_required"] = bool(parsed.get("clarifying_question"))

    # Deterministic backstop: if the patient already answered onset AND
    # severity at check-in, any clarifying question asking about duration,
    # how long it has been going on, or severity is redundant. Suppress it
    # and commit to a full assessment with what we have. We check this
    # before the brevity heuristic so a complete intake never forces a
    # needless extra round.
    if parsed.get("clarifying_required") and clarify_round == 0 and clarifying_answer is None:
        try:
            patient = db.get_patient(patient_id)
        except Exception:
            patient = None
        onset_known = bool(patient and patient.get("symptom_onset", "").strip())
        severity_known = bool(patient and patient.get("symptom_severity", "").strip())
        if onset_known and severity_known:
            # We already have onset + severity — the clarifying question is
            # asking about something the patient already told us. Suppress it
            # and force a full resolution on what we have.
            parsed["clarifying_required"] = False
            parsed["clarifying_question"] = None
            parsed["_suppressed_redundant_clarify"] = True
            # If the model left severity null (hedging), nudge it to urgent
            # rather than sending the patient back for more detail.
            if not parsed.get("severity"):
                parsed["severity"] = "urgent"
                parsed["appointment_recommended"] = True
                parsed["recommended_specialist"] = parsed.get("recommended_specialist") or "General Physician"

    # Deterministic backstop for "don't commit to a severity off a bare
    # symptom label" — only on the true first turn (no prior clarifying
    # answer yet), and only if the complaint is too brief to have given the
    # model enough to safely assess. This does NOT touch whatever general
    # self_care_advice/warning_signs the model already produced — that
    # guidance is meant to be given regardless of brevity now, so it's kept
    # as-is. It only (a) nulls out a premature severity/appointment
    # recommendation the model may have guessed at anyway, and (b) makes
    # sure a clarifying question is actually asked, using a generic one if
    # the model didn't supply its own. Doesn't touch later rounds — once the
    # patient has given a real answer, the combined text is judged on its
    # own merits by the model, not this heuristic.
    if (
        clarify_round == 0
        and clarifying_answer is None
        and _too_brief_for_assessment(user_input)
    ):
        parsed["severity"] = None
        parsed["appointment_recommended"] = False
        if not parsed.get("clarifying_question"):
            parsed["clarifying_question"] = GENERIC_BREVITY_CLARIFYING_QUESTION
        parsed["clarifying_required"] = True
        parsed["_forced_clarification"] = True

    # Deterministic enforcement of the round cap — the prompt above asks the
    # model to stop on its own, but a model (especially a small local one)
    # can still ignore that. If it's still asking after the cap, override it
    # here rather than let the conversation loop forever.
    if parsed["clarifying_required"] and clarify_round >= MAX_CLARIFY_ROUNDS:
        parsed["clarifying_required"] = False
        parsed["clarifying_question"] = None
        parsed["_forced_resolution"] = True
        if not parsed.get("assessment_summary"):
            parsed["severity"] = parsed.get("severity") or "urgent"
            parsed["assessment_summary"] = (
                "I still don't have quite enough detail to be fully confident, so to be safe "
                "I'd recommend getting this checked by a clinician."
            )
            parsed.setdefault("self_care_advice", [])
            parsed.setdefault("warning_signs", [])
            parsed["otc_relevant"] = parsed.get("otc_relevant", False)
            parsed["appointment_recommended"] = True
            parsed["recommended_specialist"] = parsed.get("recommended_specialist") or "General Physician"

    return build_response_ui("symptom", parsed)


async def run_medication_agent(patient_id: str, complaint: str):
    emergency_check = check_emergency(complaint)
    if emergency_check.get("emergency"):
        medication_emergency = {
            "escalate": True,
            "escalation_reason": "The complaint indicates a potential emergency or self-harm risk. OTC medication guidance is not appropriate.",
            "self_care": [], "otc_suggestions": [], "red_flags": [], "interaction_warnings": [],
            "reasoning": emergency_check.get("message"), "emergency": True,
        }
        return build_response_ui("medication", medication_emergency)

    user_message = f"{_visit_context_line(patient_id)}Patient ID: {patient_id}\nRequester patient ID (self): {patient_id}\nComplaint: {complaint}"

    try:
        parsed, provider_used = await _run_with_fallback(_get_medication_agent, user_message)
        parsed["_llm_provider_used"] = provider_used
    except Exception as exc:
        logger.error("Medication agent failed on all configured LLM providers: %s", exc, exc_info=True)
        error_payload = {
            "error": "medication_agent_llm_unavailable",
            "display_text": LLM_UNAVAILABLE_MESSAGE,
            "self_care": [], "otc_suggestions": [], "red_flags": [], "interaction_warnings": [],
            "escalate": False, "escalation_reason": None, "backend_error": True,
        }
        return build_response_ui("medication", error_payload)

    if parsed.get("escalate"):
        parsed["otc_suggestions"] = []
        if not parsed.get("escalation_reason"):
            parsed["escalation_reason"] = "Higher-risk complaint or clinical review required. OTC medications are not recommended at this time."
    else:
        # Deterministic safety backstop: re-check every freely-named OTC
        # suggestion against this patient's actual recorded allergies/
        # medications, regardless of whether the model already called the
        # tool itself. This is the only "database" involved anywhere in this
        # agent — the patient's own record, not a medication content dataset.
        suggestions = parsed.get("otc_suggestions") or []
        suggested_names = [s.get("option", "") for s in suggestions if s.get("option")]
        if suggested_names:
            conflict_result = check_medication_conflicts(patient_id, suggested_names)
            parsed["safety_net_conflict_check"] = conflict_result
            if not conflict_result.get("safe_to_suggest", True):
                flagged = {name for name, issues in conflict_result.get("conflicts", {}).items() if issues}
                parsed["otc_suggestions"] = [s for s in suggestions if s.get("option") not in flagged]
                parsed["escalate"] = True
                parsed["escalation_reason"] = (
                    "An automated safety check found a conflict between a suggested medication and the "
                    "patient's recorded allergies/medications. That suggestion was withheld pending clinician review."
                )
        if not parsed.get("otc_suggestions") and not parsed.get("escalate") and not parsed.get("self_care"):
            parsed["reasoning"] = parsed.get("reasoning") or (
                "I could not generate clear guidance for this complaint."
            )

    return build_response_ui("medication", parsed)


def normalize_appointment_date(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Deterministically resolves a date/time string to ISO8601 in Python —
    never left to the LLM to guess, since it has no reliable notion of
    "today" and will invent inconsistent years/dates for relative phrases
    like "tomorrow at 6pm".

    Returns (iso_string, error). If raw is already ISO8601 it's returned
    as-is. Otherwise, dateutil parses it against today's actual date,
    biased toward the future (so "Tuesday" resolves to the NEXT Tuesday,
    not last week). Returns (None, error_message) if it can't be parsed
    into anything sensible.
    """
    if not raw or not raw.strip():
        return None, "No date provided."

    now = datetime.now()

    try:
        parsed = datetime.fromisoformat(raw)
        # A fully-specified date/time (e.g. from the picker, or an already-ISO
        # string) is taken literally — if it's in the past, that's a real
        # mistake to reject, not something to silently nudge forward.
        if parsed < now:
            return None, f"{parsed.strftime('%b %d, %Y at %I:%M %p')} has already passed. Please pick a date/time in the future."
        return raw, None
    except ValueError:
        pass

    try:
        parsed = dateutil_parser.parse(raw, default=now, fuzzy=True)
    except (ValueError, OverflowError):
        return None, f"Could not understand '{raw}' as a date/time. Please provide a specific date and time."

    # dateutil's default already handles "next Tuesday" style resolution
    # correctly, but a bare time like "6pm" with no date reference
    # defaults to TODAY at 6pm even if that's already past — nudge those
    # forward a day rather than booking into the past.
    if parsed < now:
        parsed += timedelta(days=1)

    return parsed.isoformat(), None


async def run_appointment_agent(
    patient_id: str, user_input: str, action: str,
    date: Optional[str] = None, specialist: Optional[str] = None, doctor_id: Optional[str] = None,
    appointment_id: Optional[str] = None, requester_patient_id: Optional[str] = None,
    updates: Optional[dict] = None, reason: Optional[str] = None,
    consultation_mode: Optional[str] = None, contact_phone: Optional[str] = None,
):
    # Keep the doctor/specialist selected by the patient. The appointment
    # picker can send a registered doctor name, and the database layer knows
    # how to resolve either a doctor name/ID or a specialty.
    specialist = specialist or "General Physician"
    requester_patient_id = requester_patient_id or patient_id

    if action in ("create",) and date:
        normalized_date, date_error = normalize_appointment_date(date)
        if date_error:
            error_payload = {"authorized": False, "message": date_error}
            return build_response_ui("appointment", error_payload)
        date = normalized_date

    user_message = (
        f"{_visit_context_line(patient_id)}"
        f"Patient ID (whose appointment this concerns): {patient_id}\n"
        f"Requester patient ID (who is making this request — use this for authorization): {requester_patient_id}\n"
        f"Requested action: {action}\nUser input: {user_input}\nDate: {date}\nSpecialist: {specialist}\nDoctor ID: {doctor_id}\n"
        f"Appointment ID: {appointment_id}\nReason: {reason}\n"
        f"Consultation mode (video/phone/in-clinic, only relevant for action='create'): {consultation_mode}\n"
        f"Contact phone (only relevant for action='create' when consultation_mode is video/phone — pass this "
        f"straight through to manage_appointment as contact_phone so the patient can actually be reached): {contact_phone}\n"
        f"Updates (only relevant for action='update'): {updates}"
    )
    try:
        parsed, provider_used = await _run_with_fallback(_get_appointment_agent, user_message)
        parsed["_llm_provider_used"] = provider_used
    except Exception as exc:
        logger.error("Appointment agent failed on all configured LLM providers: %s", exc, exc_info=True)
        error_payload = {
            "error": "appointment_agent_llm_unavailable",
            "display_text": LLM_UNAVAILABLE_MESSAGE,
        }
        return build_response_ui("appointment", error_payload)

    # If appointment was successfully created or already exists, set a terminal route
    status = parsed.get("status")
    if status in ("created", "already_exists"):
        parsed["route"] = "appointment_booked"
        parsed["clarifying_required"] = False
        # Clear visit context to prevent old symptoms reappearing
        _clear_visit_context(patient_id)
    else:
        # For other statuses, keep the default route if any
        parsed.setdefault("route", "appointment")

    if action == "create" and status in ("created", "already_exists"):
        # Booking succeeded — clear the note/renewal flags as they are now handled
        try:
            db.update_patient(patient_id, {"needs_medical_note": "", "needs_prescription_renewal": ""})
        except Exception:
            logger.warning("Could not clear note/renewal flags after booking (patient_id=%s)", patient_id, exc_info=True)

    return build_response_ui("appointment", parsed)


# --- ORCHESTRATOR ------------------------------------------------------------
# This is the piece that ties the three agents into one conversational flow:
# the symptom checker always runs first; its own output then deterministically
# decides (in Python, not left to an LLM to remember) whether the next step is
# a clarifying question back to the patient, an automatic OTC suggestion from
# the medication agent, or a routed handoff to the appointment agent.

_DETERMINER = r"(?:my|the|our|this|that)\s+"

_APPOINTMENT_MANAGEMENT_PATTERN = re.compile(
    rf"\bappointment\w*\b|\bre-?schedul\w*\b|\bcancel\s+({_DETERMINER})?(visit|booking)\b"
    rf"|\bchange\s+({_DETERMINER})?(appointment|time|date|mode|slot)\b"
    rf"|\bmove\s+({_DETERMINER})?(appointment|booking)\b"
    rf"|\bupdate\s+({_DETERMINER})?(appointment|booking)\b",
    re.IGNORECASE,
)

# Phrases that indicate the user wants to ADD a new symptom on top of
# previously-mentioned ones, not start a fresh unrelated complaint.
_ADDITIVE_SYMPTOM_PATTERN = re.compile(
    r"\b(also|additionally|along with|on top of|plus|and also|i also|"
    r"previous symptoms?|prior symptoms?|mentioned symptoms?|same symptoms?|"
    r"those symptoms?|existing symptoms?|earlier symptoms?|as well)\b",
    re.IGNORECASE,
)


def _looks_like_appointment_management(text: str) -> bool:
    """Deterministic keyword fallback — ONLY used by _resolve_intent() when
    every LLM provider is unreachable (the LLM intent classifier is now the
    primary detection mechanism; see _classify_message_intent /
    _resolve_intent). Kept broad on purpose: a genuine symptom description
    essentially never contains the word "appointment", so matching on that
    word alone is more robust than trying to enumerate every phrasing.
    Also used by the narrow standalone-"cancel" check below."""
    return bool(_APPOINTMENT_MANAGEMENT_PATTERN.search(text))


def _looks_like_additive_symptom(text: str) -> bool:
    """Returns True when the patient is clearly adding a symptom to the ones
    already discussed rather than describing a completely new, unrelated
    complaint. Used to prevent the topic-switch detector from treating
    cumulative symptom reports as fresh conversations."""
    return bool(_ADDITIVE_SYMPTOM_PATTERN.search(text))


# Detects questions like "what are my symptoms?", "what did I tell you?",
# "can you summarise what I said?" — these are questions ABOUT the ongoing
# conversation, not new symptoms. Sending them to the symptom checker
# causes the LLM to hallucinate symptoms it never heard because it has no
# session history in its context window.
_META_QUESTION_PATTERN = re.compile(
    r"\b(what\s+(are|were|is)\s+(my|the)\s+(symptoms?|issues?|complaints?|problems?|conditions?))"
    r"|\b(what\s+did\s+i\s+(say|tell|mention|report))"
    r"|\b(summaris[ez]|summarize|recap|repeat|remind\s+me)"
    r"|\b(what\s+have\s+i\s+(said|told|mentioned|reported))"
    r"|\b(list\s+(my|the)\s+symptoms?)"
    r"|\b(what\s+symptoms?\s+(do\s+i\s+have|have\s+i\s+mentioned|did\s+i\s+mention))"
    r"|\b(what\s+(kind\s+of\s+)?(problem|issue|condition|disease|illness)\s+(do\s+i\s+have|is\s+it))"
    r"|\b(what(?:'s|\s+is)\s+wrong\s+with\s+me)"
    r"|\b(based\s+on\s+(my\s+)?(symptoms?|what\s+i\s+(said|told)))",
    re.IGNORECASE,
)


# --- Fuzzy / typo-tolerant NLU backstop -------------------------------------
# The regex patterns above (and _SYMPTOM_KEYWORDS below) catch exact
# phrasing; real patients mistype ("waht are my symtoms", "feverr since
# yesterday"). rapidfuzz gives us a lightweight, no-model-download way to
# catch near-matches on top of the regexes, so the bot reads more like it's
# actually parsing what was typed rather than pattern-matching rigidly.
# It's intentionally a backstop: only checked on short inputs, and only
# when the regex already missed, so it can't swallow a real symptom
# description that happens to share a few letters with a canonical phrase.
_META_QUESTION_CANON = [
    "what are my symptoms", "what did i tell you", "what did i say",
    "summarize what i said", "recap what i told you", "remind me what i told you",
    "list my symptoms", "what symptoms do i have", "what problem do i have",
    "based on my symptoms what is it", "what does my report suggest",
    "what does my lab report show", "what disease do i have", "what illness do i have",
]


def _fuzzy_matches(
    text: str, canon_phrases: list[str], threshold: int = 82, partial: bool = True
) -> bool:
    """True if `text` is a close typo/paraphrase match to one of the
    canonical phrases. Only applied to short inputs (<=8 words) — beyond
    that, a message is almost always a real symptom description rather
    than a mistyped meta-question or greeting."""
    stripped = text.strip().lower()
    if not stripped or len(stripped.split()) > 8:
        return False
    scorer = fuzz.partial_ratio if partial else fuzz.ratio
    return any(scorer(stripped, phrase) >= threshold for phrase in canon_phrases)


# Order-independent backstop for the phrase-level checks above. Real
# patients don't always phrase things in the canonical word order —
# "what diesese symptoms they are?" has a typo ("diesese") AND unusual
# word order ("symptoms they are" rather than "are the symptoms"), so it
# matches neither _META_QUESTION_PATTERN nor _META_QUESTION_CANON well
# enough (whole-phrase similarity drops once word order shifts). This
# checks each word independently instead of the phrase as a whole, so
# typos and reordering don't matter — only whether a diagnostic word
# ("symptom", "disease", "diagnosis"...) shows up somewhere in a short,
# clearly-interrogative message.
_META_QUESTION_KEYWORDS = ["symptom", "symptoms", "disease", "diagnosis", "condition", "illness"]
_QUESTION_START_PATTERN = re.compile(r"^(what|which|why|how|do|is|are)\b", re.IGNORECASE)


def _looks_like_diagnostic_query(text: str) -> bool:
    """True for short, clearly-interrogative questions about the patient's
    own symptoms or diagnosis, including misspellings and unusual word order.

    General educational questions such as "What are common symptoms of
    diabetes?" must not be mistaken for requests to recall this conversation.
    """
    stripped = text.strip()
    if not stripped or len(stripped.split()) > 10:
        return False
    is_question = "?" in stripped or bool(_QUESTION_START_PATTERN.match(stripped))
    if not is_question:
        return False
    words = [w for w in re.findall(r"[a-zA-Z]+", stripped.lower()) if len(w) > 3]
    has_diagnostic_word = any(fuzz.ratio(w, kw) >= 82 for w in words for kw in _META_QUESTION_KEYWORDS)
    if not has_diagnostic_word:
        return False
    asks_about_patient = bool(re.search(
        r"\b(my|do\s+i\s+have|what\s+do\s+i\s+have|i\s+have|i\s+feel|"
        r"i\s+am|i'm|these|those)\b",
        stripped,
        re.IGNORECASE,
    ))
    if not asks_about_patient:
        return False
    return not _symptom_bearing(stripped)


def _looks_like_meta_question(text: str) -> bool:
    """Returns True for questions asking the bot to recall or summarise what
    the patient already described (or a lab report they already uploaded),
    or asking what problem/disease they have based on that. These must
    NEVER reach the symptom checker — it has no session history and will
    hallucinate symptoms."""
    if _META_QUESTION_PATTERN.search(text):
        return True
    if _fuzzy_matches(text, _META_QUESTION_CANON, partial=False):
        return True
    return _looks_like_diagnostic_query(text)


# Detects clear non-medical chitchat — greetings, thanks, plain
# acknowledgements, compliments, weather remarks. Split into categories
# (rather than one catch-all) so the reply can sound like a person actually
# responding to what was said — "Hey there!" to a greeting, "You're
# welcome!" to thanks — instead of the same canned redirect every time.
# Kept intentionally narrow: only high-confidence chitchat, never anything
# that could also be a health complaint ("I feel great" is fine; "I feel
# feverish" is not caught here).
_CHITCHAT_GREETING_PATTERN = re.compile(
    r"^(hi+|hey+|hello+|good\s+(morning|afternoon|evening|night)|howdy)[!.\s]*$", re.IGNORECASE,
)
_CHITCHAT_THANKS_PATTERN = re.compile(r"^(thank(s|\s+you)?|ty|thx|cheers)[!.\s]*$", re.IGNORECASE)
_CHITCHAT_ACK_PATTERN = re.compile(r"^(ok(ay)?|sure|got\s+it|alright)[!.\s]*$", re.IGNORECASE)
_CHITCHAT_COMPLIMENT_PATTERN = re.compile(
    r"\b(what\s+a\s+(lovely|beautiful|nice|great|wonderful|amazing)\s+(day|morning|evening|weather))"
    r"|\b(lovely\s+day|beautiful\s+day|nice\s+day|great\s+day|wonderful\s+day)"
    r"|\b(you(?:'re|\s+are)\s+(great|amazing|helpful|awesome|brilliant|wonderful))"
    r"|^(great|awesome|perfect)[!.\s]*$",
    re.IGNORECASE,
)
_CHITCHAT_GREETING_CANON = ["hi", "hello", "hey there", "good morning", "good evening"]
_CHITCHAT_THANKS_CANON = ["thanks", "thank you", "thx", "appreciate it"]
_CHITCHAT_ACK_CANON = ["ok", "okay", "sure", "got it", "alright"]


def _chitchat_category(text: str) -> Optional[str]:
    """Returns 'greeting' / 'thanks' / 'ack' / 'compliment' if `text` is
    non-medical small talk, else None."""
    stripped = text.strip()
    if len(stripped.split()) > 12:
        return None
    if _CHITCHAT_GREETING_PATTERN.search(stripped) or _fuzzy_matches(
        stripped, _CHITCHAT_GREETING_CANON, threshold=88, partial=False
    ):
        return "greeting"
    if _CHITCHAT_THANKS_PATTERN.search(stripped) or _fuzzy_matches(
        stripped, _CHITCHAT_THANKS_CANON, threshold=80, partial=False
    ):
        return "thanks"
    if _CHITCHAT_ACK_PATTERN.search(stripped) or _fuzzy_matches(
        stripped, _CHITCHAT_ACK_CANON, threshold=88, partial=False
    ):
        return "ack"
    if _CHITCHAT_COMPLIMENT_PATTERN.search(stripped):
        return "compliment"
    return None


def _looks_like_chitchat(text: str) -> bool:
    """Returns True for clearly non-medical small-talk. These get a brief
    friendly redirect instead of a hallucinated symptom assessment."""
    return _chitchat_category(text) is not None


# A few natural-sounding variants per chitchat category so replies don't
# feel like the same canned line every time — randomly picked, not the
# same string on every turn.
_CHITCHAT_REPLIES: dict[str, list[str]] = {
    "greeting": [
        "Hey there! I'm here to help with any symptoms or health concerns — what's going on?",
        "Hi! Tell me about any symptoms you're dealing with, or upload a lab report if you have one.",
        "Hello! What can I help you with today — symptoms, medications, or a lab report?",
    ],
    "thanks": [
        "You're welcome! Let me know if anything else comes up.",
        "Anytime — happy to help. Anything else on your mind?",
        "Glad I could help. I'm here if you need anything else.",
    ],
    "ack": [
        "Got it. Let me know if there's anything else you'd like to go over.",
        "Sounds good — I'm here whenever you need me.",
        "Understood. Feel free to bring up anything else, health-related or otherwise.",
    ],
    "compliment": [
        "That's kind of you to say! I'm here to help with health-related questions — "
        "feel free to describe any symptoms or concerns you have.",
        "Thank you! Whenever you're ready, tell me about any symptoms or concerns and I'll take a look.",
    ],
}


def _chitchat_reply(category: str) -> str:
    return random.choice(_CHITCHAT_REPLIES.get(category, _CHITCHAT_REPLIES["compliment"]))


# Symptom-bearing keywords — used by _meta_question_response to extract
# only the symptom-relevant user turns from session history rather than
# dumping every message verbatim.
_SYMPTOM_BEARING_PATTERN = re.compile(
    r"\b(pain|ache|fever|cough|sore|nausea|vomit|dizzy|rash|bleed|swollen|swelling|"
    r"headache|migraine|fatigue|tired|weak|breath|sneez|runny|stiff|cramp|burn|itch|"
    r"stomach|chest|back|throat|ear|eye|joint|muscle|skin|blurr|faint|shiver|chills|"
    r"diarr|constip|infect|allerg|numb|tingle|discharge|wound|cut|bruise|sprain)\b",
    re.IGNORECASE,
)
_SYMPTOM_KEYWORDS = [
    "pain", "ache", "fever", "cough", "sore", "nausea", "vomit", "dizzy", "rash",
    "bleed", "swollen", "swelling", "headache", "migraine", "fatigue", "tired",
    "weak", "breath", "sneeze", "runny", "stiff", "cramp", "burn", "itch",
    "stomach", "chest", "throat", "joint", "muscle", "blurry", "faint",
    "shiver", "chills", "diarrhea", "constipation", "infection", "allergy",
    "numb", "tingle", "discharge", "wound", "bruise", "sprain",
]


def _symptom_bearing(text: str) -> bool:
    """True if `text` contains symptom language. Tries the exact regex
    first (fast, no false positives); falls back to per-word fuzzy
    matching against _SYMPTOM_KEYWORDS so common typos ("stomache",
    "feverr", "heaache") still register instead of silently being dropped
    from a symptom summary."""
    if _SYMPTOM_BEARING_PATTERN.search(text):
        return True
    words = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 3]
    return any(fuzz.ratio(word, kw) >= 84 for word in words for kw in _SYMPTOM_KEYWORDS)


def _appointment_is_upcoming(
    appointment: dict, now: Optional[datetime] = None
) -> bool:
    """Return whether an appointment has a parseable date/time in the future."""
    raw_date = str(appointment.get("date") or "").strip()
    if not raw_date:
        return False

    try:
        appointment_dt = datetime.fromisoformat(raw_date)
    except ValueError:
        try:
            appointment_dt = dateutil_parser.parse(raw_date)
        except (ValueError, TypeError, OverflowError):
            return False

    appointment_dt = appointment_dt.astimezone()
    current_dt = (now or datetime.now()).astimezone()
    return appointment_dt > current_dt


def _appointment_management_redirect(patient_id: str) -> dict:
    """Deterministic short-circuit for messages about an EXISTING
    appointment (reschedule/cancel/change time/change mode) — these are not a
    new symptom and must never reach the symptom checker. Returns structured
    data so the frontend can render real reschedule/cancel/mode-change buttons
    per appointment rather than requiring the patient to manually type IDs.
    Only future appointments are shown — past ones are already done and
    cannot be rescheduled."""
    try:
        all_active = db.get_appointments(patient_id=patient_id, active_only=True)
    except Exception:
        logger.warning("Could not fetch appointments for management redirect (patient_id=%s)", patient_id, exc_info=True)
        all_active = []

    appointments = [a for a in all_active if _appointment_is_upcoming(a)]

    if appointments:
        text = (
            "That sounds like it's about an existing appointment. Here's what you currently have booked — "
            "use the buttons below to reschedule, change the consultation mode, or cancel."
        )
        return {
            "route": "appointment_management",
            "display_text": text,
            "appointments": appointments,
        }
    else:
        text = (
            "That sounds like it's about an appointment, but I don't see any upcoming appointments on file for "
            "you. Would you like to book a new one instead?"
        )
        return {
            "route": "appointment_management",
            "display_text": text,
            "appointments": [],
        }


def _clear_visit_context(patient_id: str) -> None:
    """Once a visit_reason has actually been addressed (assessed to a
    terminal severity, or an emergency flagged), clear it from the patient
    record rather than leaving it there for the rest of the session.
    _visit_context_line re-reads this field fresh on every single call, so
    without this, an old resolved complaint ("nose bleeding") keeps getting
    re-injected as "already known, treat as such" into completely unrelated
    later messages ("I have a stomach ache", "can we make the appointment
    in-clinic instead?") for the rest of the day — which is exactly the bug
    this fixes. A clarifying round in progress is NOT a resolution, so this
    must only be called once the symptom checker reaches an actual
    endpoint (emergency, or a final self_care result), never mid-clarify."""
    try:
        db.update_patient(patient_id, {"visit_reason": "", "visit_category": "", "symptom_onset": "", "symptom_severity": ""})
    except Exception:
        logger.warning("Could not clear visit context after resolution (patient_id=%s)", patient_id, exc_info=True)


# Deterministic backstop — same rationale/pattern as
# _looks_like_appointment_management above: a small/local model asked to
# interpret a "clarifying answer" will sometimes take the reply at face
# value even when it's actually an unrelated new complaint (e.g. answering
# "how long has your nosebleed been going on?" gets "i have a stomach
# ache"), silently folding the new symptom into the old one instead of
# recognizing the topic changed. This is not a general intent classifier —
# it only catches the common, high-confidence case: the reply names a
# DIFFERENT recognizable symptom than anything already in the prior
# context. A short direct answer (duration, severity, yes/no, a number)
# never trips this, since it won't contain a symptom keyword at all.
_SYMPTOM_KEYWORDS = [
    "headache", "head ache", "migraine", "stomach ache", "stomachache", "stomach pain",
    "abdominal pain", "belly ache", "fever", "cough", "sore throat", "rash", "chest pain",
    "back pain", "backache", "nosebleed", "nose bleed", "nose bleeding", "dizziness", "dizzy",
    "vomiting", "vomit", "nausea", "nauseous", "diarrhea", "diarrhoea", "constipation",
    "earache", "ear pain", "toothache", "tooth ache", "sprain", "burn", "cut", "wound",
    "swelling", "swollen", "bruise", "itching", "itchy", "allergic reaction",
    "shortness of breath", "breathless", "cramp", "cramps", "fatigue", "joint pain",
    "knee pain", "eye pain", "blurred vision",
]


_TOPIC_SWITCH_SYSTEM_PROMPT = """You check whether a patient's reply to a clinician's clarifying question is actually about a DIFFERENT, new problem instead of answering the question asked.

Respond with ONLY a JSON object: {"new_complaint": true or false}

true  -> the reply names a different symptom/problem than the one under discussion (e.g. asked how long the nosebleed has lasted, patient answers "actually my stomach hurts now")
false -> the reply answers the question about the SAME complaint: a duration, severity, yes/no, a related detail, or another symptom that's plausibly part of the same episode

Output nothing but the JSON object."""


async def _classify_new_complaint(prior_context: str, candidate_answer: str) -> Optional[bool]:
    """LLM-based check for whether candidate_answer is really a new, unrelated
    complaint rather than an answer about the one already being discussed.
    Generalizes to any symptom, unlike a fixed keyword list. Returns None
    (never guesses) if every configured provider fails, so the caller can
    fall back to the cheaper keyword heuristic below instead of silently
    mis-classifying."""
    text = (candidate_answer or "").strip()
    if len(text.split()) < 2:
        return False  # too short to plausibly be describing a new complaint at all
    user_message = f"Current complaint / clarifying context: {prior_context}\n\nPatient's reply: {text}"
    for provider, llm_instance in _get_llm_chain():
        try:
            response = await asyncio.wait_for(
                llm_instance.ainvoke([
                    {"role": "system", "content": _TOPIC_SWITCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]),
                timeout=15,
            )
            parsed = parse_llm_response(response)
            if isinstance(parsed, dict) and "new_complaint" in parsed:
                return bool(parsed["new_complaint"])
            logger.warning("Topic-switch check on %s returned unusable output: %r", provider, parsed)
        except Exception as exc:
            logger.warning("Topic-switch check failed on %s: %s", provider, exc)
        continue
    return None


def _looks_like_new_complaint_fallback(prior_context: str, candidate_answer: str) -> bool:
    """Deterministic keyword fallback — ONLY used if every LLM provider is
    unreachable for _classify_new_complaint. Necessarily limited to the
    symptoms in _SYMPTOM_KEYWORDS; this is a safety net for an outage, not
    the primary detection mechanism (that's the LLM check above)."""
    text = (candidate_answer or "").strip().lower()
    if len(text.split()) < 2:
        return False
    prior = (prior_context or "").lower()
    found = [kw for kw in _SYMPTOM_KEYWORDS if kw in text]
    if not found:
        return False
    # Already part of the prior context -> elaboration on the same
    # complaint, not a new one.
    return not any(kw in prior for kw in found)


async def _looks_like_new_complaint(prior_context: str, candidate_answer: str) -> bool:
    result = await _classify_new_complaint(prior_context, candidate_answer)
    if result is not None:
        return result
    logger.warning("All LLM providers unavailable for topic-switch check; falling back to keyword heuristic.")
    return _looks_like_new_complaint_fallback(prior_context, candidate_answer)


_INTENT_CLASSIFIER_SYSTEM_PROMPT = """You classify a patient's chat message into exactly one category, based on what they are asking for right now.

Categories:
- "symptom": clearly describes a new or ongoing health complaint/symptom they want assessed (e.g. "I have a headache", "my stomach hurts", "not feeling well since yesterday")
- "appointment_management": about an appointment they already have or want to change — rescheduling, cancelling, changing the date/time/mode/slot, or asking what's booked (e.g. "can we change the slot?", "I need to cancel", "move my appointment to tomorrow")
- "health_question": asks for general health information, wellness or prevention guidance, an explanation of a condition/term, or general medication information, without describing a personal symptom that needs assessment (e.g. "What is blood pressure?", "What causes diabetes?", "How does insulin work?", "Can ibuprofen affect blood pressure?")
- "unclear": default to this whenever you are not confident — very short messages, garbled/typo'd text, texting-style abbreviations, or anything that doesn't clearly fit another category

Bias strongly toward "unclear" for short or garbled input rather than guessing "symptom". A message only earns "symptom" when it names an actual bodily complaint (pain, ache, fever, a body part, "not feeling well", etc.) — the mere absence of appointment-sounding words is NOT evidence of a symptom.
When the patient reports their own symptom and asks what to do, choose "symptom", not "health_question". Hypothetical or educational questions about a symptom or condition are "health_question".

Examples:
"chng the clot" -> unclear (garbled — reads like a typo'd "change the slot", not a body complaint; don't assume "clot" means the medical term here)
"asdfgh" -> unclear
"apt tmrw?" -> appointment_management
"my chest hurts" -> symptom
"I have a headache, what should I do?" -> symptom
"What causes headaches?" -> health_question
"What is blood pressure?" -> health_question
"cancel" -> appointment_management

Respond with ONLY a JSON object: {"intent": "symptom" | "appointment_management" | "health_question" | "unclear"}
Output nothing but the JSON object."""


_GENERAL_HEALTH_QUESTION_START = re.compile(
    r"^(what|how|why|when|which|can|could|does|do|is|are|should|"
    r"tell\s+me|explain|symptoms?\s+of|causes?\s+of|side\s+effects?\s+of|"
    r"treatment\s+for)\b",
    re.IGNORECASE,
)
_GENERAL_HEALTH_TERMS = re.compile(
    r"\b(symptoms?|blood\s+pressure|high\s+bp|hypertension|heart\s+rate|"
    r"blood\s+sugar|diabetes|insulin|cholesterol|medicine|medication|drug|"
    r"side\s+effects?|vaccine|vaccination|infection|flu|cold|fever|headache|"
    r"pain|nausea|sleep|diet|exercise|hydration|pregnan\w*|asthma|allerg\w*|"
    r"antibiotics?|treatment|health|wellness|disease|illness|symptom)\b",
    re.IGNORECASE,
)
_PERSONAL_SYMPTOM_STATEMENT = re.compile(
    r"\b(i\s+have|i've\s+got|i\s+feel|i'm\s+feeling|i\s+am\s+feeling|"
    r"i'm\s+having|i\s+am\s+having|am\s+i\s+having|my\s+"
    r"(?:head|chest|stomach|back|throat|ear|eye|joint|muscle|skin|"
    r"body|symptom|pain|ache|fever|cough|nausea|rash|dizziness|breath))\b",
    re.IGNORECASE,
)


def _looks_like_general_health_question(text: str) -> bool:
    """Recognize straightforward educational questions without relying on an LLM."""
    stripped = (text or "").strip()
    if not stripped or not _GENERAL_HEALTH_QUESTION_START.match(stripped):
        return False
    if not _GENERAL_HEALTH_TERMS.search(stripped):
        return False
    if _PERSONAL_SYMPTOM_STATEMENT.search(stripped):
        return False
    return True


async def _classify_message_intent(text: str) -> Optional[str]:
    """Classify a free-text message into symptom, appointment, health question,
    or unclear.

    Replaces regex-only routing for the primary intent decision: a keyword
    pattern has to enumerate every possible phrasing ("change the slot" vs
    "change my slot" vs "move my booking" vs "the doctor's timing doesn't
    work for me"...) and will always miss some. A classifier generalizes
    across phrasing instead of needing a new pattern for every variant.

    Returns None (never guesses) if every configured provider fails, so the
    caller can fall back to something safer than silently assuming intent.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "unclear"
    for provider, llm_instance in _get_llm_chain():
        try:
            response = await asyncio.wait_for(
                llm_instance.ainvoke([
                    {"role": "system", "content": _INTENT_CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": stripped},
                ]),
                timeout=15,
            )
            parsed = parse_llm_response(response)
            intent = parsed.get("intent") if isinstance(parsed, dict) else None
            if intent in ("symptom", "appointment_management", "health_question", "unclear"):
                return intent
            logger.warning("Intent classifier on %s returned unusable output: %r", provider, parsed)
        except Exception as exc:
            logger.warning("Intent classification failed on %s: %s", provider, exc)
        continue
    return None


async def _resolve_intent(text: str) -> str:
    """Return a supported top-level chat intent.

    Tries the LLM classifier first. If every provider is unreachable, falls
    back to the narrow appointment-keyword regex and returns 'unclear' rather
    than silently assuming an unrecognized message is a symptom.
    """
    if _looks_like_appointment_management(text):
        return "appointment_management"
    if _looks_like_general_health_question(text):
        return "health_question"

    intent = await _classify_message_intent(text)
    if intent is not None:
        return intent
    logger.warning("All LLM providers unavailable for intent classification; falling back to keyword heuristic.")
    return "unclear"


def _intent_clarify_menu() -> dict:
    """Shown only when _resolve_intent can't confidently tell whether a
    fresh message is a symptom or about an existing appointment. The
    frontend should render intent_options as tappable buttons rather than
    asking the patient to retype anything. The "other" option routes the
    original question to the general health assistant."""
    return {
        "route": "intent_clarify",
        "display_text": "Just to make sure I help with the right thing — is this about:",
        "intent_options": [
            {"id": "symptom", "label": "A symptom or health concern"},
            {"id": "appointment_management", "label": "An existing appointment"},
            {"id": "other", "label": "A general health question"},
        ],
    }


GENERAL_HEALTH_SYSTEM_PROMPT = """You are a general health information assistant for a clinic. Answer the patient's health-related question directly, clearly, and in plain language. Provide general education and prevention information, not a diagnosis or a substitute for professional care. Include one brief, relevant safety precaution in every answer; avoid generic boilerplate when a specific precaution is more useful.

Do not claim to know the patient's medical history, medications, allergies, or test results unless they explicitly shared them in this conversation. Do not diagnose an individual, prescribe medication, or tell someone to start, stop, or change a prescribed treatment. For questions about whether a medicine is safe for that person, give cautious general information, explain that safety depends on their conditions and other medicines, and recommend checking with their clinician or pharmacist. Do not invent sources, statistics, or citations.

If the question describes symptoms happening to the patient, do not guess a cause; encourage them to describe the symptoms for the clinic's symptom-assessment flow. If they describe possible immediate danger (for example severe chest pain, trouble breathing, stroke signs, overdose, or risk of self-harm), advise contacting local emergency services or going to an emergency department now.

Keep the answer relevant to the question. If it is ambiguous, ask one brief follow-up question. If it is not health-related, politely say you can help with health information, symptom assessment, lab reports, and appointments."""


async def run_general_health_agent(question: str) -> dict:
    """Answer general health questions without invoking symptom or appointment tools."""
    messages = [
        {"role": "system", "content": GENERAL_HEALTH_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]

    failures = []
    try:
        providers = _get_llm_chain()
    except Exception as exc:
        logger.error("General health assistant could not initialize an LLM provider: %s", exc, exc_info=True)
        return {"route": "error", "display_text": LLM_UNAVAILABLE_MESSAGE}

    for provider, llm_instance in providers:
        try:
            response = await asyncio.wait_for(
                llm_instance.ainvoke(messages),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "\n".join(
                    part if isinstance(part, str) else str(part.get("text") or "")
                    for part in content
                    if isinstance(part, (str, dict))
                )
            answer = str(content or "").strip()
            if answer:
                return {"route": "general_health", "display_text": answer}
            failures.append(f"{provider}: returned an empty answer")
        except Exception as exc:
            failures.append(f"{provider}: {exc}")
            logger.warning("General health answer failed on %s: %s", provider, exc)

    logger.error("All configured providers failed for general health question: %s", "; ".join(failures))
    return {"route": "error", "display_text": LLM_UNAVAILABLE_MESSAGE}


def _meta_question_response(session_history: list[dict]) -> dict:
    """Answers 'what are my symptoms / what kind of problem do I have based on
    my symptoms' — and now also 'what does my lab report suggest' — from the
    actual session chat history, without calling any LLM agent.

    Pulls from two sources in the same history:
      - user turns that describe symptoms (as before)
      - assistant turns tagged source="lab_report" (see app.py's
        handle_lab_report / agents._lab_report_history_summary), i.e. a
        lab report the patient uploaded earlier in THIS session.
    Both get folded into one natural-sounding answer instead of only ever
    reflecting typed symptoms, which is what caused report-related
    meta-questions to be answered from stale symptom chat alone.
    """
    if not session_history:
        return {
            "route": "meta",
            "display_text": (
                "I don't have any previous messages recorded in this session. "
                "Please describe your symptoms, or upload a lab report, and I'll help assess it."
            ),
        }

    # Extract user turns that contain symptom-bearing language. We skip:
    #   - assistant turns (we only want what the patient told us)
    #   - turns that are themselves meta-questions or chitchat
    #   - very short non-symptom turns (yes/no answers, numbers, durations)
    #   - lab-report upload markers (source="lab_report"), handled separately below
    symptom_turns = []
    for t in session_history:
        if t.get("role") != "user" or t.get("source") == "lab_report":
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue
        if _looks_like_meta_question(content) or _looks_like_chitchat(content):
            continue
        if _looks_like_general_health_question(content):
            continue
        if _symptom_bearing(content):
            symptom_turns.append(content)

    # Extract lab-report findings stored earlier this session.
    lab_findings = [
        (t.get("content") or "").strip()
        for t in session_history
        if t.get("role") == "assistant" and t.get("source") == "lab_report" and (t.get("content") or "").strip()
    ]

    if not symptom_turns and not lab_findings:
        return {
            "route": "meta",
            "display_text": (
                "I don't see any symptom descriptions or lab reports in our conversation yet. "
                "Go ahead and tell me what's going on, or upload a report, and I'll take a look."
            ),
        }

    sections = []
    if symptom_turns:
        if len(symptom_turns) == 1:
            sections.append(f"From what you've told me, you mentioned: \"{symptom_turns[0]}\"")
        else:
            items = "\n".join(f"• {t}" for t in symptom_turns)
            sections.append(f"From what you've told me:\n{items}")
    if lab_findings:
        items = "\n".join(f"• {f}" for f in lab_findings) if len(lab_findings) > 1 else lab_findings[0]
        label = "From the lab report you uploaded" if len(lab_findings) == 1 else "From the lab reports you uploaded"
        sections.append(f"{label}:\n{items}" if len(lab_findings) > 1 else f"{label}: {items}")

    closing = (
        "Would you like me to run a full assessment combining these, or is there anything else to add?"
        if symptom_turns and lab_findings else
        "Would you like me to run a full assessment on that, or is there anything else to add?"
    )
    summary = "\n\n".join(sections) + "\n\n" + closing

    return {"route": "meta", "display_text": summary}


async def run_appointment_direct(
    patient_id: str, action: str,
    date: Optional[str] = None, specialist: Optional[str] = None, doctor_id: Optional[str] = None,
    appointment_id: Optional[str] = None, updates: Optional[dict] = None,
    reason: Optional[str] = None, consultation_mode: Optional[str] = None,
    requester_patient_id: Optional[str] = None, contact_phone: Optional[str] = None,
) -> dict:
    """
    Calls manage_appointment directly from Python — zero LLM involved.
    Used for create/update/cancel where every parameter is already known
    (date picker, mode selector, phone collector in the frontend all
    resolve the values before calling the API). The LLM appointment agent
    is only needed for free-text lookup/cancel where the patient hasn't
    provided an appointment_id explicitly.
    """
    # Keep the exact doctor/specialist selected in the appointment widget.
    # The database resolver accepts a registered doctor name/ID first and
    # falls back to specialty matching, so replacing the selection here would
    # make the booking target a different doctor than the patient chose.
    specialist = specialist or "General Physician"
    requester_patient_id = requester_patient_id or patient_id

    if action == "create" and date:
        normalized_date, date_error = normalize_appointment_date(date)
        if date_error:
            result = {"authorized": False, "message": date_error}
            return build_response_ui("appointment", result)
        date = normalized_date

    result = manage_appointment(
        action=action,
        patient_id=patient_id,
        requester_patient_id=requester_patient_id,
        date=date,
        specialist=specialist,
        doctor_id=doctor_id,
        appointment_id=appointment_id,
        updates=updates,
        reason=reason,
        consultation_mode=consultation_mode,
        contact_phone=contact_phone,
    )

    # Set terminal route if booking succeeded
    status = result.get("status")
    if status in ("created", "already_exists"):
        result["route"] = "appointment_booked"
        result["clarifying_required"] = False
        _clear_visit_context(patient_id)
        try:
            db.update_patient(patient_id, {"needs_medical_note": "", "needs_prescription_renewal": ""})
        except Exception:
            logger.warning("Could not clear note/renewal flags after booking (patient_id=%s)", patient_id, exc_info=True)
    else:
        result.setdefault("route", "appointment")

    return build_response_ui("appointment", result)


async def run_patient_symptom_flow(patient_id: str, user_input: str, clarifying_answer: Optional[str] = None, clarify_round: int = 0, session_history: Optional[list] = None, forced_intent: Optional[str] = None) -> dict:
    """
    Symptoms -> assessment -> (optionally) ask user if they want medication
    recommendations -> medication guidance -> offer appointment.

    Flow:
    1. Show assessment (severity, self-care, warning signs) immediately.
    2. Ask "Would you like medication recommendations?" — only call the
       medication agent if the user says yes.
    3. After medication (or if skipped), always offer an appointment.

    Appointment booking is NEVER auto-started here — even high-severity cases
    only get a stronger recommendation and wait for the patient to click
    "Book Appointment". EMERGENCY skips all of this.

    forced_intent: set when the patient just tapped an option on the
    _intent_clarify_menu() card. Bypasses classification so the original
    question follows the selected symptom, appointment, or general-health path.
    """
    if forced_intent == "appointment_management":
        return _appointment_management_redirect(patient_id)
    if forced_intent == "other":
        return await run_general_health_agent(user_input)
    # forced_intent == "symptom" (or None) falls through to the normal flow below.

    # --- Meta-question intercept: "what are my symptoms / what did I tell you"
    # must NEVER reach the symptom checker — it has no session history and
    # will hallucinate symptoms it never heard. Answer from the history instead.
    if clarify_round == 0 and clarifying_answer is None and _looks_like_meta_question(user_input):
        return _meta_question_response(session_history or [])

    # --- Chitchat intercept: greetings, thanks, acknowledgements, compliments,
    # non-medical small-talk ("what a lovely day") must not reach the symptom
    # checker — it will try to assess them as symptoms and hallucinate random
    # conditions. Reply is picked per-category and randomized so it reads
    # like a person responding, not the same canned line every time.
    if clarify_round == 0 and clarifying_answer is None:
        chitchat_category = _chitchat_category(user_input)
        if chitchat_category:
            return {"route": "meta", "display_text": _chitchat_reply(chitchat_category)}

    # --- NEW: Standalone "cancel" check ---
    # If the user says just "cancel" (with optional punctuation/spaces) and has
    # upcoming appointments, redirect to appointment management.
    if (
        clarify_round == 0
        and clarifying_answer is None
        and re.match(r'^\s*cancel\s*$', user_input, re.IGNORECASE)
    ):
        try:
            upcoming = db.get_appointments(patient_id=patient_id, active_only=True)
            future = [a for a in upcoming if _appointment_is_upcoming(a)]
            if future:
                return _appointment_management_redirect(patient_id)
            else:
                return {
                    "route": "meta",
                    "display_text": (
                        "I don't see any upcoming appointments to cancel. "
                        "If you have a health concern, please describe your symptoms and I'll help."
                    ),
                }
        except Exception:
            # If DB fails, continue to symptom checker
            pass

    # --- Intent classification (fresh message): symptom, appointment,
    # general health question, or unclear. A genuine "unclear" gets a
    # tappable menu rather than being silently assumed to be a symptom.
    if clarify_round == 0 and clarifying_answer is None and forced_intent is None:
        intent = await _resolve_intent(user_input)
        if intent == "appointment_management":
            return _appointment_management_redirect(patient_id)
        if intent == "health_question":
            return await run_general_health_agent(user_input)
        if intent == "unclear":
            return _intent_clarify_menu()
        # intent == "symptom" -> continue to the checks below as before.

    # --- Clarifying answer that pivots to appointment management ---
    # If the user is in a clarifying round and their reply is actually about
    # an existing appointment, abort symptom flow and redirect to the
    # appointment UI. A non-appointment classification here (including
    # "unclear") is NOT treated as ambiguous — short factual answers like
    # "3 days" or "yes" are expected and should just continue the flow
    # below; only a fresh top-level message needs the clarify-menu fallback.
    if clarify_round > 0 and clarifying_answer:
        pivot_intent = await _resolve_intent(clarifying_answer)
        if pivot_intent == "appointment_management":
            _clear_visit_context(patient_id)
            return _appointment_management_redirect(patient_id)
        if pivot_intent == "health_question":
            return await run_general_health_agent(clarifying_answer)

    # --- Additive-symptom check: if the patient is clearly adding to prior
    # symptoms ("along with previous symptoms, I also feel fever"), treat it
    # as a clarifying answer on the accumulated complaint rather than a new
    # topic. Do this BEFORE the new-complaint topic-switch check so we never
    # erroneously reset a growing symptom list.
    if clarify_round > 0 and clarifying_answer and _looks_like_additive_symptom(clarifying_answer):
        # Merge the new symptom into the existing context — NOT a topic switch.
        combined_complaint = f"{user_input}\n{clarifying_answer}"
        result = await run_symptom_checker(patient_id, user_input, clarifying_answer, clarify_round)
        result.setdefault("route", None)
        if result.get("emergency"):
            result["route"] = "emergency"
            _clear_visit_context(patient_id)
            return result
        if result.get("error"):
            result["route"] = "error"
            return result
        result["_additive_symptom"] = True
        # Additive symptom resolved — same terminal path as the main flow:
        # ask the user if they want medication, then offer appointment.
        if result.get("clarifying_required"):
            result["route"] = "clarify"
            question = result.get("clarifying_question")
            if question:
                result["display_text"] = (result.get("display_text", "") + "\n\n" + question).strip()
            return result
        severity = (result.get("severity") or "urgent").lower()
        appointment_recommended = bool(result.get("appointment_recommended"))
        # Force GP
        specialist = "General Physician"
        result["recommended_specialist"] = specialist
        result["route"] = "ask_medication"
        result["offer_appointment"] = True
        result["offer_appointment_specialist"] = specialist
        _clear_visit_context(patient_id)
        return result

    # --- Topic-switch check: a "clarifying answer" that actually names an
    # unrelated new symptom is a fresh complaint, not a reply to the pending
    # question — gracefully end/suspend the old chain and start a new one.
    if clarify_round > 0 and clarifying_answer and await _looks_like_new_complaint(user_input, clarifying_answer):
        _clear_visit_context(patient_id)
        fresh_result = await run_patient_symptom_flow(patient_id, clarifying_answer, None, 0, session_history)
        fresh_result["_topic_switched"] = True
        return fresh_result

    combined_complaint = user_input if not clarifying_answer else f"{user_input}\n{clarifying_answer}"

    # (Fresh-message appointment-management/unclear routing is already
    # handled above via _resolve_intent — no duplicate check needed here.)

    # Cheap, deterministic, non-LLM keyword scan — checked here BEFORE
    # spending any LLM calls, so a genuine emergency short-circuits fast.
    if check_emergency(combined_complaint).get("emergency"):
        result = await run_symptom_checker(patient_id, user_input, clarifying_answer, clarify_round)
        result["route"] = "emergency"
        _clear_visit_context(patient_id)
        return result

    # Run symptom checker first. Medication is now gated behind a user
    # confirmation step — we don't run it automatically anymore.
    result = await run_symptom_checker(patient_id, user_input, clarifying_answer, clarify_round)
    result.setdefault("route", None)

    if result.get("emergency"):
        result["route"] = "emergency"
        return result
    if result.get("error"):
        result["route"] = "error"
        return result

    # Still in clarification phase — show assessment + ask the clarifying
    # question. Medication is NOT run yet; it will be offered after resolution.
    if result.get("clarifying_required"):
        result["route"] = "clarify"
        question = result.get("clarifying_question")
        if question:
            result["display_text"] = (result.get("display_text", "") + "\n\n" + question).strip()
        return result

    # Terminal resolution — assessment is complete. Now ask the user if they
    # want medication recommendations before offering an appointment.
    severity = (result.get("severity") or "urgent").lower()
    appointment_recommended = bool(result.get("appointment_recommended"))
    # Force GP
    specialist = "General Physician"
    result["recommended_specialist"] = specialist

    # Terminal resolution — assessment is complete. Always ask the user if
    # they want medication recommendations. We no longer skip this based on
    # the LLM's otc_relevant flag — that flag was causing the bot to silently
    # bypass the medication offer for complaints like headaches where OTC
    # options (paracetamol, ibuprofen) are clearly appropriate but a small
    # model incorrectly marks otc_relevant=false. The patient can always
    # say "No thanks" if they don't want suggestions.
    result["route"] = "ask_medication"
    result["offer_appointment_specialist"] = specialist
    result["offer_appointment"] = True

    # Deterministic backstop: if the patient flagged a prescription renewal
    # or medical note need at check-in, surface it explicitly now so it
    # never gets silently dropped. A note/renewal can't be resolved by
    # self-care, so always ensure the appointment is offered.
    try:
        _patient_for_offer = db.get_patient(patient_id)
    except Exception:
        _patient_for_offer = None
    needs_note = bool(_patient_for_offer) and str(_patient_for_offer.get("needs_medical_note", "")).strip().lower() == "yes"
    needs_renewal = bool(_patient_for_offer) and str(_patient_for_offer.get("needs_prescription_renewal", "")).strip().lower() == "yes"
    if needs_note or needs_renewal:
        what = (
            "a prescription renewal and a medical note" if needs_note and needs_renewal
            else ("a medical note" if needs_note else "a prescription renewal")
        )
        result["display_text"] = (
            result.get("display_text", "") +
            f"\n\nAlso, since you mentioned at check-in that you need {what}, "
            f"I'll make sure that's included when we book your appointment."
        ).strip()

    _clear_visit_context(patient_id)
    return result


async def run_patient_medication_then_appointment(patient_id: str, complaint: str, specialist: str) -> dict:
    """
    Called after the user explicitly says yes to medication recommendations.
    Runs the medication agent on the complaint, then returns a result
    with route='self_care' so the frontend can show medication + appointment offer.
    """
    medication_result = await run_medication_agent(patient_id, complaint)
    return {
        "route": "self_care",
        "self_care_result": medication_result,
        "offer_appointment": True,
        "offer_appointment_specialist": specialist,
        "display_text": medication_result.get("display_text", ""),
    }


# --- LAB REPORT AGENT ---------------------------------------------------------
# Uses a vision-capable provider directly (Gemini's OpenAI-compatible endpoint
# supports image parts) rather than the ReAct tool-calling agents above, since
# this is a single-shot multimodal read, not a multi-step tool workflow.

def _build_vision_llm():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return ChatOpenAI(
            model=os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash"),
            temperature=0, api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return None


def _pdf_to_content_parts(file_bytes: bytes) -> list:
    """Extracts text if the PDF has a real text layer; otherwise rasterizes
    up to the first 3 pages to images for the vision model to read."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    full_text = "\n".join(text_parts).strip()

    if len(full_text) > 40:  # has a usable text layer
        return [{"type": "text", "text": f"Lab report text content:\n\n{full_text}"}]

    # Scanned/no text layer -> render pages as images instead
    parts = []
    for page in doc[:3]:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return parts


def _image_to_content_parts(file_bytes: bytes, mime_type: str) -> list:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return [{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}]


async def run_lab_report_agent(patient_id: str, file_bytes: bytes, filename: str, mime_type: str) -> dict:
    llm = _build_vision_llm()
    if llm is None:
        error_payload = {
            "error": "No vision-capable LLM provider configured. Set GEMINI_API_KEY in .env.",
        }
        error_payload["display_text"] = error_payload["error"]
        return error_payload

    try:
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            content_parts = _pdf_to_content_parts(file_bytes)
        else:
            content_parts = _image_to_content_parts(file_bytes, mime_type or "image/png")
    except Exception as exc:
        logger.error("Could not read the uploaded lab report file: %s", exc, exc_info=True)
        error_payload = {"error": "lab_report_unreadable"}
        error_payload["display_text"] = "Sorry, I couldn't read that file. Please make sure it's a valid PDF or image and try again."
        return error_payload

    content_parts.append({"type": "text", "text": "Explain this lab report per your instructions."})
    message = HumanMessage(content=content_parts)

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": LAB_REPORT_SYSTEM_PROMPT},
            message,
        ])
        parsed = parse_llm_response(response)
    except Exception as exc:
        logger.error("Lab report agent failed: %s", exc, exc_info=True)
        parsed = {"error": "lab_report_llm_unavailable"}

    if parsed.get("error"):
        parsed["display_text"] = parsed["error"]
        return parsed

    lines = [parsed.get("summary", "")]
    for t in parsed.get("tests", []):
        flag = t.get("flag", "unclear")
        marker = {"normal": "✓", "high": "▲ high", "low": "▼ low", "unclear": "?"}.get(flag, "?")
        lines.append(f"\n• {t.get('name')}: {t.get('value')} [{marker}] (reference: {t.get('reference_range', 'n/a')})")
        lines.append(f"  {t.get('explanation', '')}")
    if parsed.get("flagged_for_doctor"):
        lines.append("\nWorth discussing with your doctor:")
        for item in parsed["flagged_for_doctor"]:
            lines.append(f"• {item}")
    if parsed.get("unreadable_sections"):
        lines.append("\nCouldn't read clearly:")
        for item in parsed["unreadable_sections"]:
            lines.append(f"• {item}")

    parsed["display_text"] = "\n".join(lines)
    return parsed


def _lab_report_history_summary(res: dict) -> str:
    """Compact, prose version of a lab-report result for storing in the
    session history (see sessions.append_turn(..., source="lab_report")).

    Deliberately shorter than `display_text` (which is bullet-formatted for
    the chat UI) — this is what _meta_question_response and the symptom
    checker's own context will read back later, so it reads like something
    a person would say about their results, not a rendered card.
    """
    parts = []
    summary = (res.get("summary") or "").strip()
    if summary:
        parts.append(summary)

    abnormal = [
        f"{t.get('name')} was {t.get('flag')} ({t.get('value')}, reference {t.get('reference_range', 'n/a')})"
        for t in res.get("tests", [])
        if t.get("flag") in ("high", "low")
    ]
    if abnormal:
        parts.append("Abnormal results: " + "; ".join(abnormal) + ".")

    if res.get("flagged_for_doctor"):
        parts.append("Flagged to discuss with a doctor: " + "; ".join(res["flagged_for_doctor"]) + ".")

    return " ".join(parts) if parts else "Lab report reviewed; no abnormal results were flagged."