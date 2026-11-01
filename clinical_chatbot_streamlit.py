"""Streamlit host for the Clinical Chatbot.

The UI is the chatbot's own ``clinical_chatbot/index.html``, rendered unchanged
inside Streamlit, so the hosted build looks exactly like the .bat build.

How it works (no UI is re-implemented in Streamlit widgets):

* ``index.html`` is loaded in an iframe through a small two-way Streamlit
  component that is generated at runtime from the chatbot folder, so editing
  ``index.html`` is all that is needed to change the UI.
* The page calls ``fetch('/api/...')`` and ``fetch('/agent/...')`` as usual. A
  tiny shim hands each call to Streamlit, and Python runs it against the
  chatbot's own FastAPI ``app`` in-process (no server, no extra port), so
  ``clinical_chatbot/app.py`` stays the single source of truth for behaviour.
* The signed-in user is taken from ``st.session_state.current_user`` and the
  bridge only lets a patient touch their own record.
"""
import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import streamlit as st
import streamlit.components.v1 as components

_BRIDGE_DIR = Path(__file__).resolve().parent
_CHATBOT_DIR = _BRIDGE_DIR / "clinical_chatbot"

_LLM_ENV_KEYS = [
    "GROQ_API_KEY", "GROQ_MODEL",
    "CEREBRAS_API_KEY", "CEREBRAS_MODEL",
    "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_VISION_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL",
]

_cached = None
_INIT_LOCK = threading.Lock()


# --- environment / credentials ------------------------------------------------

def _ensure_cloud_env() -> None:
    try:
        cfg = st.secrets.get("google_config", {})
    except Exception:
        cfg = {}
    gs = cfg.get("google_sheet", {}) if isinstance(cfg, dict) else {}
    if isinstance(gs, dict) and gs.get("spreadsheet_id"):
        os.environ.setdefault("GOOGLE_SHEET_ID", str(gs["spreadsheet_id"]))
    for key in _LLM_ENV_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


def _resolve_service_account_file() -> str:
    existing = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "")
    if existing and os.path.isfile(existing):
        return existing
    for path in (_BRIDGE_DIR / "service_account.json", _CHATBOT_DIR / "service_account.json"):
        if path.is_file():
            os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = str(path)
            return str(path)
    try:
        sa = st.secrets.get("service_account")
    except Exception:
        sa = None
    if sa:
        fd, path = tempfile.mkstemp(suffix="-service-account.json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(sa), handle)
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = path
        return path
    return ""


# --- loading the chatbot backend ----------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_chatbot_ready() -> dict:
    """Import the chatbot backend once per process.

    The chatbot's ``agents.py`` / ``app.py`` and the Full Diagnostics page's
    ``agents`` package both want the bare module name ``agents``. The chatbot
    copies are registered under ``clinical_chatbot.*`` and ``agents`` is only
    pointed at ours while ``app.py`` runs its ``from agents import ...`` line,
    then restored, so the other page is unaffected.
    """
    global _cached
    with _INIT_LOCK:
        if _cached is not None:
            return _cached
        os.environ["CDSS_DIRECT_TOOLS"] = "1"
        _ensure_cloud_env()
        _resolve_service_account_file()
        chatbot_dir = str(_CHATBOT_DIR)
        if chatbot_dir not in sys.path:
            sys.path.insert(0, chatbot_dir)
        import database as cb_database
        import mcp_tools as cb_tools
        cb_agents = _load_module("clinical_chatbot.agents", _CHATBOT_DIR / "agents.py")
        previous = sys.modules.get("agents")
        sys.modules["agents"] = cb_agents
        try:
            cb_app = _load_module("clinical_chatbot.app", _CHATBOT_DIR / "app.py")
        finally:
            if previous is None:
                sys.modules.pop("agents", None)
            else:
                sys.modules["agents"] = previous
        _cached = {
            "database": cb_database,
            "mcp_tools": cb_tools,
            "agents": cb_agents,
            "app": cb_app,
        }
        # Same warm-up the FastAPI startup hook does (Sheets schema + agents),
        # in the background so the page paints immediately.
        startup = getattr(cb_app, "initialize_data_layer", None)
        if startup is not None:
            asyncio.run_coroutine_threadsafe(startup(), _loop())
        return _cached


# --- one background event loop shared by every Streamlit session ---------------

_LOOP = None
_LOOP_LOCK = threading.Lock()


def _loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True, name="cdss-loop").start()
            _LOOP = loop
        return _LOOP


def _run_async(coro, timeout=None):
    return asyncio.run_coroutine_threadsafe(coro, _loop()).result(timeout)


_CLIENT = None


async def _asgi_client(cb: dict):
    """httpx client wired straight into the chatbot's FastAPI app."""
    global _CLIENT
    if _CLIENT is None:
        import httpx
        _CLIENT = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=cb["app"].app),
            base_url="http://cdss.local",
            timeout=None,
        )
    return _CLIENT


# --- request bridge: browser fetch() -> FastAPI app ----------------------------

_ALLOWED_PATHS = {
    "/api/health", "/api/identify", "/api/identify-doctor", "/api/intake-questions",
    "/api/intake", "/api/patient", "/api/patients", "/api/doctors", "/api/appointments",
    "/api/doctor-schedule", "/agent/chat", "/agent/appointment",
    "/agent/medication-then-appointment", "/agent/lab-report",
}
_DOCTOR_ONLY_PATHS = {"/api/patients"}  # full patient list; identify-doctor handled separately
_IDENTITY_FIELDS = ("patient_id", "requester_patient_id")


def _json_response(status: int, payload: dict) -> dict:
    return {"status": status, "text": json.dumps(payload), "contentType": "application/json"}


def _authorize(path: str, sources: list, ctx: dict):
    """Return an error message if this session may not make the call."""
    if path not in _ALLOWED_PATHS:
        return "That endpoint is not available."
    if path == "/api/identify-doctor":
        if not ctx["is_doctor"]:
            return "Doctors only."
        if any(str(s.get("doctor_id", ctx["pid"])) != ctx["pid"] for s in sources):
            return "You can only sign in as yourself."
        return None
    if ctx["is_doctor"]:
        return None
    if path in _DOCTOR_ONLY_PATHS:
        return "Doctors only."
    for source in sources:
        for field in _IDENTITY_FIELDS:
            if field in source and str(source[field]) != ctx["pid"]:
                return "You can only access your own record."
    return None


async def _dispatch(cb: dict, req: dict, ctx: dict) -> dict:
    method = str(req.get("method") or "GET").upper()
    parts = urlsplit(str(req.get("url") or ""))
    path = parts.path
    query = dict(parse_qsl(parts.query))
    body = req.get("body") or None
    headers = {k: v for k, v in (req.get("headers") or {}).items() if k.lower() == "content-type"}

    sources = [query]
    kwargs = {}
    if body and body.get("kind") == "form":
        fields = {k: v for k, v in body.get("fields", [])}
        sources.append(fields)
        kwargs["data"] = fields
        kwargs["files"] = [
            (name, (filename, base64.b64decode(b64), ctype or "application/octet-stream"))
            for name, filename, ctype, b64 in body.get("files", [])
        ]
        headers.pop("content-type", None)
        headers.pop("Content-Type", None)
    elif body and body.get("kind") == "text":
        text = body.get("text") or ""
        kwargs["content"] = text.encode("utf-8")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                sources.append(parsed)
        except ValueError:
            pass

    denied = _authorize(path, sources, ctx)
    if denied:
        return _json_response(403, {"detail": denied})

    try:
        client = await _asgi_client(cb)
        response = await client.request(
            method, path, params=query or None, headers=headers or None, **kwargs
        )
        return {
            "status": response.status_code,
            "text": response.text,
            "contentType": response.headers.get("content-type", "application/json"),
        }
    except Exception as exc:  # the app already maps its own errors to 4xx/5xx
        st.session_state.cb_last_error = f"{type(exc).__name__}: {exc}"
        return _json_response(500, {"detail": "Sorry, something went wrong. Please try again."})


def _handle_pending_request(cb: dict, key: str, ctx: dict) -> None:
    """Run the newest request the browser sent, once, before the component is drawn."""
    state = st.session_state.get(key)
    req = state.get("req") if isinstance(state, dict) else None
    if not req or req.get("id") == st.session_state.cb_last_req_id:
        return
    st.session_state.cb_last_req_id = req["id"]
    result = _run_async(_dispatch(cb, req, ctx))
    if int(result.get("status", 200)) >= 500:
        st.session_state.cb_last_error = st.session_state.get("cb_last_error") or result.get("text", "")
    st.session_state.cb_last_resp = {"id": req["id"], **result}


# --- the component: index.html in an iframe + fetch shim -----------------------

_SHIM_JS = r"""
<script>
(function () {
  var realFetch = window.fetch.bind(window);
  function b64(buf) {
    var bytes = new Uint8Array(buf), chunk = 0x8000, out = '';
    for (var i = 0; i < bytes.length; i += chunk) {
      out += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(out);
  }
  async function serialize(body) {
    if (body == null) return null;
    if (typeof FormData !== 'undefined' && body instanceof FormData) {
      var fields = [], files = [];
      for (var pair of body.entries()) {
        if (typeof pair[1] === 'string') { fields.push([pair[0], pair[1]]); }
        else { files.push([pair[0], pair[1].name, pair[1].type, b64(await pair[1].arrayBuffer())]); }
      }
      return { kind: 'form', fields: fields, files: files };
    }
    return { kind: 'text', text: String(body) };
  }
  window.fetch = async function (input, init) {
    init = init || {};
    var url = typeof input === 'string' ? input : (input && input.url) || String(input);
    if (!/^\/(api|agent)\//.test(url)) return realFetch(input, init);
    var signal = init.signal;
    if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
    var headers = {};
    if (init.headers) {
      new Headers(init.headers).forEach(function (v, k) { headers[k] = v; });
    }
    var req = { method: init.method || 'GET', url: url, headers: headers, body: await serialize(init.body) };
    var call = window.parent.__cdssBridge(req);
    var aborted = new Promise(function (_, reject) {
      if (signal) signal.addEventListener('abort', function () {
        reject(new DOMException('Aborted', 'AbortError'));
      });
    });
    var res = await Promise.race([call, aborted]);
    return new Response(res.text, { status: res.status, headers: { 'Content-Type': res.contentType } });
  };
})();
</script>
"""

_WRAPPER_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
  iframe { border: 0; width: 100%; display: block; }
</style></head>
<body>
<iframe id="app" title="Clinical Chatbot" allow="clipboard-write"></iframe>
<script>
(function () {
  var nonce = Math.random().toString(36).slice(2), seq = 0;
  var queue = [], inflight = null, pending = {}, mounted = false;
  var REQUEST_TIMEOUT_MS = 180000;

  function post(type, data) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data || {}), '*');
  }
  function resize() {
    var h = 700;
    try { h = Math.max(640, window.parent.innerHeight - 110); } catch (e) {}
    document.getElementById('app').style.height = h + 'px';
    post('streamlit:setFrameHeight', { height: h });
  }
  function pump() {
    if (inflight || !queue.length) return;
    inflight = queue.shift();
    post('streamlit:setComponentValue', { value: { req: inflight.req }, dataType: 'json' });
    inflight.timer = setTimeout(function () { finish(inflight.req.id, null, 'The server took too long to answer.'); }, REQUEST_TIMEOUT_MS);
  }
  function finish(id, response, error) {
    var entry = pending[id];
    if (!entry) return;
    delete pending[id];
    clearTimeout(entry.timer);
    if (inflight && inflight.req.id === id) inflight = null;
    if (error) entry.reject(new Error(error)); else entry.resolve(response);
    pump();
  }
  // Called by the shim inside the chatbot page.
  window.__cdssBridge = function (req) {
    return new Promise(function (resolve, reject) {
      var id = nonce + '-' + (++seq);
      var entry = { req: Object.assign({ id: id }, req), resolve: resolve, reject: reject };
      pending[id] = entry;
      queue.push(entry);
      pump();
    });
  };
  window.addEventListener('message', function (event) {
    var msg = event.data;
    if (!msg || msg.type !== 'streamlit:render') return;
    var args = msg.args || {};
    if (!mounted) {
      mounted = true;
      var qs = new URLSearchParams();
      if (args.patient_id) qs.set('patient_id', args.patient_id);
      if (args.name) qs.set('name', args.name);
      if (args.role) qs.set('role', args.role);
      document.getElementById('app').src = 'app.html?' + qs.toString();
    }
    if (args.response && args.response.id) finish(args.response.id, args.response, null);
    resize();
  });
  try { window.parent.addEventListener('resize', resize); } catch (e) {}
  post('streamlit:componentReady', { apiVersion: 1 });
  resize();
})();
</script></body></html>
"""

_COMPONENT = None


def _build_component_dir(index_html: str) -> str:
    """Write the wrapper + patched chatbot page to a temp folder (content-addressed)."""
    app_html = index_html
    # Every call goes through Streamlit now, so give calls more headroom than the
    # standalone page's 3s/8s startup timeouts (first request also connects to Sheets).
    app_html = app_html.replace(
        "setTimeout(() => controller.abort(), timeoutMs)",
        "setTimeout(() => controller.abort(), Math.max(timeoutMs, 45000))",
    )
    head_at = app_html.lower().find("<head>")
    insert_at = head_at + len("<head>") if head_at >= 0 else 0
    app_html = app_html[:insert_at] + _SHIM_JS + app_html[insert_at:]

    digest = hashlib.sha1((app_html + _WRAPPER_HTML).encode("utf-8")).hexdigest()[:12]
    target = Path(tempfile.gettempdir()) / f"cdss_chatbot_component_{digest}"
    if not (target / "app.html").is_file():
        target.mkdir(parents=True, exist_ok=True)
        (target / "index.html").write_text(_WRAPPER_HTML, encoding="utf-8")
        (target / "app.html").write_text(app_html, encoding="utf-8")
    return str(target)


def _component():
    global _COMPONENT
    if _COMPONENT is None:
        html = (_CHATBOT_DIR / "index.html").read_text(encoding="utf-8")
        _COMPONENT = components.declare_component(
            "cdss_chatbot_bridge", path=_build_component_dir(html)
        )
    return _COMPONENT


# --- page ----------------------------------------------------------------------

def _is_doctor() -> bool:
    return str(
        st.session_state.get("current_user", {}).get("user_type", "")
    ).strip().casefold() == "doctor"


def _configured_provider_names() -> list:
    mapping = {
        "GROQ_API_KEY": "Groq",
        "CEREBRAS_API_KEY": "Cerebras",
        "GEMINI_API_KEY": "Gemini",
        "OPENROUTER_API_KEY": "OpenRouter",
    }
    names = [name for key, name in mapping.items() if os.environ.get(key)]
    if os.environ.get("OLLAMA_BASE_URL"):
        names.append("Ollama")
    return names


def _init_state() -> None:
    st.session_state.setdefault("cb_last_req_id", None)
    st.session_state.setdefault("cb_last_resp", None)
    st.session_state.setdefault("cb_last_error", None)


_PAGE_CSS = """
<style>
  [data-testid="stMainBlockContainer"], .block-container { padding-top: 1rem; padding-bottom: 0; max-width: 100%; }
  [data-testid="stCustomComponentV1"] iframe { border: 0; }
</style>
"""


def render_clinical_chatbot_page() -> None:
    _init_state()
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    if st.button("← Back to dashboard", key="cb_back_dashboard"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    try:
        cb = ensure_chatbot_ready()
    except Exception as exc:
        st.error("The chatbot backend could not be loaded.")
        st.code(str(exc))
        return

    if not _configured_provider_names():
        st.warning(
            "No AI provider is configured yet. Add GROQ_API_KEY, CEREBRAS_API_KEY, "
            "GEMINI_API_KEY or OPENROUTER_API_KEY to your secrets (or the clinical_chatbot/.env "
            "file locally). Basic responses and safety checks still work, but full AI answers "
            "need one of these keys."
        )

    user = st.session_state.get("current_user", {}) or {}
    pid = str(user.get("user_id") or "patient-self")
    name = str(user.get("full_name") or pid)
    is_doctor = _is_doctor()
    ctx = {"pid": pid, "is_doctor": is_doctor}

    key = f"cb_bridge::{'doctor' if is_doctor else 'patient'}::{pid}"
    _handle_pending_request(cb, key, ctx)

    _component()(
        patient_id=pid,
        name=name,
        role="doctor" if is_doctor else "",
        response=st.session_state.cb_last_resp,
        key=key,
        default=None,
    )

    if st.session_state.get("cb_last_error"):
        with st.expander("Technical details"):
            st.code(str(st.session_state.cb_last_error))