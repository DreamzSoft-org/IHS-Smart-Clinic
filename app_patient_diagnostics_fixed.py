import base64
import hashlib
import hmac
import json
import os
import re
import sys
import threading
import subprocess
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Application settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMART Clinic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

BASE_DIR = Path(__file__).resolve().parent
STARTUP_SCREEN_1 = BASE_DIR / "assets" / "startup_screen_1.png"
STARTUP_SCREEN_2 = BASE_DIR / "assets" / "startup_screen_2.png"

# Medical background used by the login screen.
BACKGROUND_IMAGE_FILE = BASE_DIR / "assets" / "medical-background.webp"
SMILE_BANNER_FILE = BASE_DIR / "assets" / "smile-ihs-hd.png"
if BACKGROUND_IMAGE_FILE.exists():
    BACKGROUND_IMAGE_DATA = base64.b64encode(
        BACKGROUND_IMAGE_FILE.read_bytes()
    ).decode("ascii")
else:
    BACKGROUND_IMAGE_DATA = ""

# Dashboard diagnosis-card icons.
BREAST_CANCER_ICON_FILE = BASE_DIR / "assets" / "breast-cancer-icon.png"
HEART_DISEASE_ICON_FILE = BASE_DIR / "assets" / "heart-disease-icon.png"
BRAIN_STROKE_ICON_FILE = BASE_DIR / "assets" / "brain-stroke-icon.png"
CLINICAL_CHATBOT_DIR = BASE_DIR / "clinical_chatbot"
AGENTIC_DIAGNOSTIC_DIR = Path(
    os.environ.get(
        "AGENTIC_DIAGNOSTIC_DIR",
        str(BASE_DIR / "general_diagnostics"),
    )
)
AGENTIC_DIAGNOSTIC_PORT = 8600
# Persistent patient medical-history storage.
# The legacy project-folder file is retained only for automatic migration.
LEGACY_MEDICAL_HISTORY_FILE = BASE_DIR / "medical_history_data.json"

if os.name == "nt":
    _history_root = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    MEDICAL_HISTORY_DATA_DIR = _history_root / "IHS_Sehat_Plus"
else:
    MEDICAL_HISTORY_DATA_DIR = Path.home() / ".ihs_sehat_plus"

MEDICAL_HISTORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDICAL_HISTORY_FILE = MEDICAL_HISTORY_DATA_DIR / "medical_history_data.json"
MEDICAL_HISTORY_LOCK = threading.Lock()
DIAGNOSTIC_REPORTS_FILE = BASE_DIR / "diagnostic_reports_data.json"

DEFAULT_MALE_PROFILE_FILE = BASE_DIR / "assets" / "default-profile-male.png"
DEFAULT_FEMALE_PROFILE_FILE = BASE_DIR / "assets" / "default-profile-female.jpg"
DEFAULT_OTHER_PROFILE_FILE = BASE_DIR / "assets" / "default-profile-other.jpg"
PROFILE_PHOTO_DIR = BASE_DIR / "uploads" / "profile_photos"
PROFILE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Agentic-Diagnostic project integration
# ---------------------------------------------------------
# Recommended folder structure:
#
# SMART-CDSS/
# ├── app.py
# ├── run_app.bat
# └── Agentic-Diagnostic/
#     ├── agents/
#     │   ├── __init__.py
#     │   └── diagnostic_agent.py
#     ├── utils/
#     │   ├── __init__.py
#     │   └── pdf_generator.py
#     └── .env
#
# You may also set DIAGNOSTIC_APP_DIR in Windows before running the app.
DIAGNOSTIC_PROJECT_CANDIDATES = [
    Path(os.environ["DIAGNOSTIC_APP_DIR"]).expanduser()
    if os.environ.get("DIAGNOSTIC_APP_DIR")
    else None,
    # The bundled diagnostics source included with this distribution.
    BASE_DIR / "general_diagnostics",
    # Backward-compatible locations used by older project layouts.
    BASE_DIR / "Agentic-Diagnostic",
    BASE_DIR / "Streamlit-VS-Code" / "Agentic-Diagnostic",
    Path(
        r"C:\Users\rfanm\OneDrive\Desktop\Smart e-Health"
        r"\Streamlit-VS-Code\Agentic-Diagnostic"
    ),
]


def find_diagnostic_project() -> Path:
    """Find the project that contains the diagnostic agent and PDF generator."""
    for candidate in DIAGNOSTIC_PROJECT_CANDIDATES:
        if candidate is None:
            continue

        candidate = candidate.resolve()
        agent_file = candidate / "agents" / "diagnostic_agent.py"
        pdf_file = candidate / "utils" / "pdf_generator.py"

        if agent_file.is_file() and pdf_file.is_file():
            return candidate

    checked = "\n".join(
        f"• {candidate}"
        for candidate in DIAGNOSTIC_PROJECT_CANDIDATES
        if candidate is not None
    )

    raise FileNotFoundError(
        "The Agentic-Diagnostic source folder could not be found.\n\n"
        "It must contain both:\n"
        "  agents\\diagnostic_agent.py\n"
        "  utils\\pdf_generator.py\n\n"
        "Checked these locations:\n"
        f"{checked}\n\n"
        "Recommended fix: place the complete Agentic-Diagnostic folder "
        "beside app.py."
    )


def configure_diagnostic_imports() -> Path:
    """Add Agentic-Diagnostic to Python's module search path."""
    project_dir = find_diagnostic_project()
    project_text = str(project_dir)

    if project_text not in sys.path:
        sys.path.insert(0, project_text)

    return project_dir
def _resolve_service_account_file() -> Path:
    """Find the Google service-account key in supported project locations.

    The desktop bundle historically shipped a second copy under
    clinical_chatbot/. If the root copy is missing after extraction, use that
    bundled copy instead of failing the whole login screen.
    """
    env_value = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    candidates: list[Path] = []

    if env_value:
        env_path = Path(env_value).expanduser()
        if not env_path.is_absolute():
            env_path = BASE_DIR / env_path
        candidates.append(env_path)

    candidates.extend([
        BASE_DIR / "service_account.json",
        BASE_DIR / "clinical_chatbot" / "service_account.json",
    ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # Return the normal root path so any eventual error message remains clear.
    return BASE_DIR / "service_account.json"


SERVICE_ACCOUNT_FILE = _resolve_service_account_file()
GOOGLE_CONFIG_FILE = BASE_DIR / "google_config.json"


def _credential_payloads() -> dict[str, object]:
    """Collect the Google credentials from Streamlit secrets (any supported
    shape) or from process environment variables. Returns filename -> payload."""
    try:
        secrets = getattr(st, "secrets", None) or {}
    except Exception:
        secrets = {}

    payloads: dict[str, object] = {}

    disk = _secrets_from_disk_toml()
    disk_sa = disk.get("service_account")
    if isinstance(disk_sa, dict):
        payloads.setdefault("service_account.json", disk_sa)
    disk_gc = disk.get("google_config")
    if isinstance(disk_gc, dict):
        payloads.setdefault("google_config.json", disk_gc)

    def _coerce(value: object) -> dict | None:
        if value is None:
            return None
        try:
            return dict(value)
        except Exception:
            return None

    service_account = _coerce(secrets.get("service_account"))
    if service_account:
        payloads.setdefault("service_account.json", service_account)
    google_config = _coerce(secrets.get("google_config"))
    if google_config:
        payloads.setdefault("google_config.json", google_config)

    # Environment-variable / plain-string fallback. Both process environment
    # variables and TOML-scalar secrets are accepted so the operator never has
    # to hand-convert the JSON into nested sections.
    env_service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "") or ""
    plain_secret_account = secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if isinstance(plain_secret_account, str):
        env_service_account = env_service_account or plain_secret_account
    if env_service_account:
        payloads.setdefault("service_account.json", env_service_account)

    env_google_config = os.environ.get("GOOGLE_CONFIG_JSON", "") or ""
    plain_secret_config = secrets.get("GOOGLE_CONFIG_JSON")
    if isinstance(plain_secret_config, str):
        env_google_config = env_google_config or plain_secret_config
    if env_google_config:
        payloads.setdefault("google_config.json", env_google_config)

    return payloads


def _secrets_from_disk_toml() -> dict:
    """Read the Streamlit secrets file directly and parse it as TOML.

    Streamlit Cloud stores the secrets the operator pastes into the dashboard
    at ``~/.streamlit/secrets.toml`` inside the container's writable home
    directory. Reading that file with tomllib avoids every quirk of the
    ``st.secrets`` proxy object (which does not behave like a plain dict for
    nested sections in every Streamlit version).
    """
    try:
        import tomllib  # type: ignore[import-not-found]  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}
    try:
        path = Path.home() / ".streamlit" / "secrets.toml"
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cloud_service_account_info() -> dict | None:
    """Return an in-memory Google service-account dict taken from Streamlit
    secrets or the environment, without touching the filesystem.

    This is what makes the hosted build work: Streamlit Community Cloud runs a
    read-only app folder, so a secret-backed dict (pasted in the secrets editor)
    is converted straight into google.auth credentials.
    """
    disk = _secrets_from_disk_toml()
    sa = disk.get("service_account")
    if isinstance(sa, dict) and sa.get("client_email") and sa.get("private_key"):
        return sa

    try:
        secrets = getattr(st, "secrets", None) or {}
    except Exception:
        secrets = {}

    sa = secrets.get("service_account")
    if sa:
        try:
            sa = dict(sa)
        except Exception:
            sa = None
        if isinstance(sa, dict) and sa.get("client_email") and sa.get("private_key"):
            return sa

    candidates: list[object] = []
    plain_secret = secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if isinstance(plain_secret, str) and plain_secret.strip():
        candidates.append(plain_secret)
    env_payload = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if env_payload.strip():
        candidates.append(env_payload)

    for candidate in candidates:
        info = candidate
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(info, dict) and info.get("client_email") and info.get("private_key"):
            return info
    return None


def _materialize_cloud_files() -> None:
    """
    Streamlit Community Cloud has no persistent disk and has a read-only app
    directory, and this repository keeps the Google credentials out of git.
    When the app runs on the cloud the operator pastes those files into the
    Streamlit secrets editor. Materialize them into a writable location here so
    the Google Sheets login keeps working unchanged.

    The desktop edition is unaffected: when the files already exist locally,
    this function does nothing.
    """
    temp_dir = Path(tempfile.gettempdir())
    for file_name, payload in _credential_payloads().items():
        base_path = BASE_DIR / file_name
        if base_path.exists():
            continue
        text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)

        bound_path = base_path
        try:
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(text, encoding="utf-8")
        except OSError:
            # Read-only application folder (Streamlit Community Cloud): fall
            # back to the container temp directory, which is always writable.
            bound_path = temp_dir / file_name
            try:
                bound_path.write_text(text, encoding="utf-8")
            except OSError:
                continue

        if file_name == "service_account.json":
            globals()["SERVICE_ACCOUNT_FILE"] = bound_path
            os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = str(bound_path)
        else:
            globals()["GOOGLE_CONFIG_FILE"] = bound_path


_materialize_cloud_files()

# Hosted (Streamlit Cloud) mode: some pages spawn local companion servers and
# embed them with iframes to 127.0.0.1. That cannot work in a hosted build, so
# those pages show a clear notice instead. Set app.cloud=true in Streamlit
# secrets (or the CLOUD_DEPLOY env variable) to enable this behaviour.
def _read_cloud_deploy_flag() -> bool:
    """Detect Streamlit Cloud deployment without raising when no secrets.toml exists."""
    if os.environ.get("CLOUD_DEPLOY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            app = secrets.get("app") or {}
            return bool(app.get("cloud", False))
    except Exception:
        pass
    return False


CLOUD_DEPLOY = _read_cloud_deploy_flag()


# Supplied Heart Disease Diagnosis model and dataset.
HEART_DISEASE_DATA_FILE = (
    BASE_DIR / "heart_disease" / "cleaned_merged_heart_dataset.csv"
)
HEART_DISEASE_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalachh", "exang", "oldpeak", "slope", "ca", "thal",
]


@st.cache_resource(show_spinner="Preparing the heart disease model...")
def load_heart_disease_model():
    """Train and cache the supplied Random Forest heart-disease model."""
    dataset = pd.read_csv(HEART_DISEASE_DATA_FILE)
    missing = [
        column for column in HEART_DISEASE_FEATURES + ["target"]
        if column not in dataset.columns
    ]
    if missing:
        raise ValueError(
            "Heart disease dataset is missing required columns: "
            + ", ".join(missing)
        )

    features = dataset[HEART_DISEASE_FEATURES]
    target = dataset["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, accuracy


def generate_heart_clinical_summary(
    patient_data: dict[str, float],
    risk_probability: float,
) -> str:
    """Generate the template-based clinical summary supplied with the HDD model."""
    age = int(patient_data.get("age", 0))
    sex = "Male" if int(patient_data.get("sex", 0)) == 1 else "Female"
    bp = int(patient_data.get("trestbps", 0))
    chol = int(patient_data.get("chol", 0))
    heart_rate = int(patient_data.get("thalachh", 0))
    st_depression = float(patient_data.get("oldpeak", 0))
    exercise_angina = int(patient_data.get("exang", 0)) == 1
    major_vessels = int(patient_data.get("ca", 0))

    if risk_probability > 70:
        risk_category = "HIGH RISK"
        urgency = "immediate further clinical evaluation"
    elif risk_probability > 40:
        risk_category = "MODERATE RISK"
        urgency = "follow-up testing is recommended"
    else:
        risk_category = "LOW RISK"
        urgency = "continue routine monitoring"

    risk_factors = []
    if bp > 140:
        risk_factors.append("elevated resting blood pressure")
    if chol > 240:
        risk_factors.append("high cholesterol")
    if st_depression > 1.0:
        risk_factors.append("significant ST depression")
    if major_vessels > 1:
        risk_factors.append(f"multiple vessel involvement ({major_vessels})")
    if exercise_angina:
        risk_factors.append("exercise-induced angina")
    if heart_rate < 60:
        risk_factors.append("low maximum heart rate")

    findings = (
        "Key findings include " + ", ".join(risk_factors) + "."
        if risk_factors
        else "The entered measurements contain no additional rule-based warning flags."
    )
    return (
        f"{age}-year-old {sex} with a {risk_category} model profile "
        f"({risk_probability:.1f}% estimated probability). {findings} "
        f"Recommendation: {urgency}. This result supports, but does not replace, "
        "assessment by a qualified clinician."
    )


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        /* Main page background */
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(
                135deg,
                #f4f7fb 0%,
                #eef3ff 100%
            );
        }

        .main .block-container {
            max-width: 1320px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Make normal Streamlit text dark and visible */
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp p,
        .stApp label,
        .stApp li,
        .stApp span,
        .stApp div[data-testid="stMarkdownContainer"],
        .stApp div[data-testid="stText"],
        .stApp div[data-testid="stMetricLabel"],
        .stApp div[data-testid="stMetricValue"],
        .stApp div[data-testid="stMetricDelta"],
        .stApp div[data-testid="stCaptionContainer"] {
            color: #0f172a;
        }

        /* Sentence text and section headings */
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] strong {
            color: #0f172a !important;
        }

        /* Input boxes */
        .stTextInput input,
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        /* Input labels */
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextArea"] label {
            color: #0f172a !important;
            font-weight: 600;
        }

        /* Metric score */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 14px;
            padding: 1rem;
        }

        div[data-testid="stMetricLabel"] p {
            color: #475569 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
        }

        /* Expander text */
        details,
        details summary,
        details p,
        details li {
            color: #0f172a !important;
        }

        /* Hero section */
        .hero-box {
            padding: 2.35rem 2.5rem;
            border-radius: 24px;
            background: linear-gradient(120deg, rgba(219, 234, 254, 0.96), rgba(239, 246, 255, 0.96));
            border: 1px solid rgba(147, 197, 253, 0.30);
            margin-bottom: 1.75rem;
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.10);
        }

        .hero-box h1 {
            margin: 0;
            font-size: 2.25rem;
            color: #0f172a !important;
        }

        .hero-box p {
            margin: 0.8rem 0 0 0;
            color: #334155 !important;
            font-size: 1.05rem;
        }


        /* Dashboard welcome bar only */
        .welcome-hero {
            position: relative;
            overflow: hidden;
            padding: 2rem;
            border-radius: 28px;
            background: linear-gradient(120deg, #eef4ff 0%, #dce8ff 52%, #c7d8ff 100%);
            border: 1px solid rgba(96, 165, 250, 0.22);
            margin-bottom: 1.5rem;
            box-shadow: 0 14px 34px rgba(59, 130, 246, 0.13);
        }

        .welcome-hero::after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -90px;
            top: -115px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(167, 139, 250, 0.28), rgba(96, 165, 250, 0));
            pointer-events: none;
        }

        .welcome-hero h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            font-size: 2.25rem;
            color: #17346f !important;
        }

        .welcome-hero p {
            position: relative;
            z-index: 1;
            margin: 0.65rem 0 0 0;
            color: #4b648c !important;
            font-size: 1.05rem;
        }

        @media (max-width: 760px) {
            .welcome-hero {
                padding: 1.5rem;
                border-radius: 22px;
            }

            .welcome-hero h1 {
                font-size: 1.8rem;
            }
        }

        /* Patient search page */
        .patient-search-hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem;
            border-radius: 26px;
            background: linear-gradient(120deg, #dbeafe 0%, #e0e7ff 48%, #fae8ff 100%);
            border: 1px solid rgba(99, 102, 241, 0.18);
            margin-bottom: 1.25rem;
            box-shadow: 0 14px 35px rgba(99, 102, 241, 0.14);
        }

        .patient-search-hero::after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            right: -65px;
            top: -85px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.28), rgba(168, 85, 247, 0.22));
        }

        .patient-search-hero h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            font-size: 2.25rem;
            color: #1e3a8a !important;
        }

        .patient-search-hero p {
            position: relative;
            z-index: 1;
            margin: 0.7rem 0 0 0;
            color: #475569 !important;
            font-size: 1.05rem;
        }

        .search-tip-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.25rem 0 1.25rem 0;
        }

        .search-tip-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            color: #0f172a;
            font-weight: 600;
            box-shadow: 0 7px 20px rgba(15, 23, 42, 0.06);
        }

        .search-tip-card small {
            display: block;
            margin-top: 0.25rem;
            color: #64748b;
            font-weight: 400;
        }

        .tip-blue { background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 1px solid #bfdbfe; }
        .tip-pink { background: linear-gradient(135deg, #fdf2f8, #fae8ff); border: 1px solid #f5d0fe; }
        .tip-green { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 1px solid #a7f3d0; }

        div[data-testid="stForm"] {
            background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(248,250,252,0.94));
            border: 1px solid #dbeafe;
            border-radius: 22px;
            padding: 1.2rem 1.2rem 0.35rem 1.2rem;
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.08);
        }

        @media (max-width: 760px) {
            .search-tip-grid { grid-template-columns: 1fr; }
            .patient-search-hero { padding: 1.5rem; }
            .patient-search-hero h1 { font-size: 1.8rem; }
        }

        /* Login page */
        .login-title {
            text-align: center;
            margin-top: 0.5rem;
            color: #0f172a !important;
        }

        .login-subtitle {
            text-align: center;
            color: #64748b !important;
            margin-bottom: 1.5rem;
        }

        /* Dashboard cards */
        .tool-card {
            height: 220px !important;
            min-height: 220px !important;
            max-height: 220px !important;
            box-sizing: border-box !important;
            padding: 1.35rem !important;
            border-radius: 20px;
            background: #ffffff;
            border: 1px solid #dbe4f0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
            margin-bottom: 0.8rem;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden;
        }

        .tool-icon {
            height: 3.25rem !important;
            min-height: 3.25rem !important;
            font-size: 2rem;
            line-height: 1;
            margin-bottom: 0.55rem;
            display: flex;
            align-items: center;
        }

        .tool-icon img {
            width: 3.25rem !important;
            height: 3.25rem !important;
            max-width: 3.25rem !important;
            max-height: 3.25rem !important;
            object-fit: contain;
            display: block;
            image-rendering: auto;
        }

        .tool-title {
            font-size: 1.18rem;
            font-weight: 700;
            color: #0f172a !important;
            margin-bottom: 0.4rem;
        }

        .tool-text {
            color: #64748b !important;
            line-height: 1.5;
            min-height: 3rem;
        }

        .status-ready {
            align-self: flex-start;
            display: inline-block;
            margin-top: auto;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534 !important;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .status-soon {
            align-self: flex-start;
            display: inline-block;
            margin-top: auto;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: #475569 !important;
            font-size: 0.82rem;
            font-weight: 700;
        }

        /* Login form */
        div[data-testid="stForm"] {
            background: #ffffff;
            padding: 1.4rem;
            border: 1px solid #dbe4f0;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }

        /* Buttons: gray background with clearly visible text */
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #64748b !important;
            color: #ffffff !important;
            border: 1px solid #64748b !important;
            border-radius: 12px;
            font-weight: 700;
        }

        div.stButton > button p,
        div[data-testid="stFormSubmitButton"] > button p {
            color: #ffffff !important;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #475569 !important;
            border-color: #475569 !important;
            color: #ffffff !important;
        }

        /* Disabled dashboard buttons */
        div.stButton > button:disabled {
            background-color: #cbd5e1 !important;
            border-color: #cbd5e1 !important;
            color: #475569 !important;
            opacity: 1 !important;
        }

        div.stButton > button:disabled p {
            color: #475569 !important;
        }

        /* Password visibility button */
        div[data-testid="stTextInput"] button {
            background-color: #e2e8f0 !important;
            color: #334155 !important;
            border: 1px solid #cbd5e1 !important;
        }

        div[data-testid="stTextInput"] button svg {
            fill: #334155 !important;
            color: #334155 !important;
        }

        /* General diagnostics page */
        div[data-testid="stAlert"] {
            border-radius: 16px !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
            padding: 0.35rem 0.6rem !important;
        }

        div[data-testid="stAlert"] p {
            color: #0f172a !important;
        }

        div[data-testid="stCaptionContainer"] {
            background: rgba(239, 246, 255, 0.92);
            border: 1px solid rgba(147, 197, 253, 0.28);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin: 0.75rem 0 1rem 0;
            box-shadow: 0 8px 22px rgba(59, 130, 246, 0.05);
        }

        div[data-testid="stCodeBlock"] {
            border-radius: 16px !important;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
        }

        .stButton > button {
            border-radius: 12px !important;
        }
    
        /* White dropdown controls and menus */
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        div[role="listbox"],
        ul[role="listbox"],
        div[data-baseweb="popover"] > div {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        li[role="option"],
        div[role="option"] {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        li[role="option"]:hover,
        div[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        div[role="option"][aria-selected="true"] {
            background: #f1f5f9 !important;
            color: #0f172a !important;
        }


        /* Force every User type dropdown and opened option panel to white */
        .stSelectbox div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #0f172a !important;
            border-color: #334155 !important;
            box-shadow: none !important;
        }

        .stSelectbox [data-baseweb="select"] *,
        div[data-testid="stSelectbox"] [data-baseweb="select"] * {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        body > div[data-baseweb="popover"],
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] > div,
        [role="listbox"] {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #0f172a !important;
        }

        [role="option"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background-color: #eef2f7 !important;
            color: #0f172a !important;
        }


        /* White user-type dropdown matching the reference design */
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #d7dee8 !important;
            border-radius: 16px !important;
            min-height: 64px !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08) !important;
        }

        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="select"] > div:focus-within {
            background: #ffffff !important;
            border-color: #c4cfdd !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.10) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: #0f172a !important;
        }

        div[data-baseweb="select"] svg {
            fill: #0f172a !important;
            color: #0f172a !important;
        }

        div[role="listbox"],
        ul[role="listbox"],
        div[data-baseweb="popover"] > div {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #d7dee8 !important;
            border-radius: 14px !important;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.14) !important;
        }

        li[role="option"],
        div[role="option"] {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        li[role="option"]:hover,
        div[role="option"]:hover,
        li[aria-selected="true"],
        div[aria-selected="true"] {
            background: #f1f5f9 !important;
            color: #0f172a !important;
        }

    
        /* FINAL LOGIN SELECTBOX FIX — compatible with current Streamlit/BaseWeb */
        div[data-testid="stSelectbox"] {
            width: 100% !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"],
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #0f172a !important;
            border-color: #d5dde8 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] > div {
            min-height: 64px !important;
            padding-left: 0.75rem !important;
            border: 1px solid #d5dde8 !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08) !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
        .stSelectbox div[data-baseweb="select"] > div:hover,
        .stSelectbox div[data-baseweb="select"] > div:focus-within {
            background-color: #ffffff !important;
            border-color: #c5cfdb !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.10) !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
        .stSelectbox div[data-baseweb="select"] * {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        div[data-testid="stSelectbox"] svg,
        .stSelectbox svg {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        /* Dropdown menu is mounted elsewhere in the page, so target it globally. */
        body div[data-baseweb="popover"],
        body div[data-baseweb="popover"] > div,
        body div[data-baseweb="menu"],
        body div[data-baseweb="menu"] > div,
        body ul[role="listbox"],
        body div[role="listbox"] {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #0f172a !important;
            border-color: #d5dde8 !important;
        }

        body li[role="option"],
        body div[role="option"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        body li[role="option"]:hover,
        body div[role="option"]:hover,
        body li[role="option"][aria-selected="true"],
        body div[role="option"][aria-selected="true"] {
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
        }


        /* Login form V3: User type, username and password share one card */
        div[data-testid="stForm"] {
            background: #ffffff !important;
            border: 1px solid #dbe4f0 !important;
            border-radius: 22px !important;
            padding: 1.5rem 1.5rem 0.8rem 1.5rem !important;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.10) !important;
        }

        div[data-testid="stForm"] div[data-testid="stSelectbox"] {
            margin-top: 0 !important;
            margin-bottom: 0.8rem !important;
        }


        /* FINAL AUTH LAYOUT FIX */
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.97) !important;
            border: 1px solid rgba(219, 228, 240, 0.98) !important;
            border-radius: 32px !important;
            padding: 2rem 2.4rem 1.4rem 2.4rem !important;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.14) !important;
            overflow: visible !important;
        }

        div[data-testid="stForm"] div[data-testid="stSelectbox"] {
            margin-bottom: 1.1rem !important;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] {
            margin-bottom: 0.65rem !important;
        }

        div[data-testid="stForm"] div[data-baseweb="input"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            min-height: 58px !important;
            overflow: hidden !important;
            box-shadow: none !important;
        }

        div[data-testid="stForm"] div[data-baseweb="input"]:focus-within {
            border-color: #94a3b8 !important;
            box-shadow: 0 0 0 1px #94a3b8 !important;
        }

        div[data-testid="stForm"] div[data-baseweb="input"] input {
            background: #ffffff !important;
            border: 0 !important;
            color: #0f172a !important;
            min-height: 56px !important;
            padding-left: 1rem !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        div[data-testid="stForm"] div[data-baseweb="input"] input::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #64748b !important;
        }

        div[data-testid="stForm"] div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            min-height: 58px !important;
            box-shadow: none !important;
        }

        div[data-testid="stForm"] label,
        div[data-testid="stForm"] label p {
            color: #0f172a !important;
            font-weight: 500 !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            min-height: 62px !important;
            border-radius: 16px !important;
            margin-top: 0.4rem !important;
        }

        @media (max-width: 760px) {
            div[data-testid="stForm"] {
                padding: 1.4rem 1.2rem 1rem 1.2rem !important;
                border-radius: 24px !important;
            }
        }


        /* SMILE banner directly below login/signup panel */
        .login-smile-banner {
            width: 100%;
            margin: 18px auto 12px auto;
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.96);
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.10);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-smile-banner img {
            display: block;
            width: 100%;
            max-width: 680px;
            height: auto;
            object-fit: contain;
        }

        /* Right-side standalone module menu on the login page */
        .module-menu-heading {
            font-size: 1.15rem;
            font-weight: 800;
            color: #0f172a !important;
            margin: 0 0 0.2rem 0;
            text-align: center;
        }

        .module-menu-subtitle {
            font-size: 0.82rem;
            color: #64748b !important;
            margin: 0 0 0.85rem 0;
            text-align: center;
        }

        .module-menu-selected {
            margin-top: 0.7rem;
            padding: 0.7rem 0.8rem;
            border-radius: 12px;
            background: rgba(239,246,255,0.95);
            border: 1px solid #bfdbfe;
            color: #1e3a8a !important;
            font-size: 0.85rem;
            font-weight: 700;
            text-align: center;
        }

        /* IHS Solutions Menu colors only — no other UI is changed */
        div[data-testid="stVerticalBlock"]:has(.ihs-menu-marker) div[data-testid="stButton"] button {
            color: #172033 !important;
            border-color: rgba(71, 85, 105, 0.16) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.ihs-menu-marker) div[data-testid="stButton"] button p {
            color: #172033 !important;
            font-weight: 700 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-1):not(:has(.menu-color-2))
        div[data-testid="stButton"] > button {
            background-color: #EEF2F1 !important;
            background: #EEF2F1 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-1):not(:has(.menu-color-2))
        div[data-testid="stButton"] > button:hover {
            background-color: #EEF2F1 !important;
            background: #EEF2F1 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-2):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #D9C6B8 !important;
            background: #D9C6B8 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-2):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #D9C6B8 !important;
            background: #D9C6B8 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-3):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #F1DED0 !important;
            background: #F1DED0 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-3):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #F1DED0 !important;
            background: #F1DED0 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-4):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #F2B2A9 !important;
            background: #F2B2A9 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-4):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #F2B2A9 !important;
            background: #F2B2A9 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-5):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #F2A19E !important;
            background: #F2A19E !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-5):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #F2A19E !important;
            background: #F2A19E !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-6):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #EEF2F1 !important;
            background: #EEF2F1 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-6):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #EEF2F1 !important;
            background: #EEF2F1 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-7):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #D9C6B8 !important;
            background: #D9C6B8 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-7):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #D9C6B8 !important;
            background: #D9C6B8 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-8):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #F1DED0 !important;
            background: #F1DED0 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-8):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #F1DED0 !important;
            background: #F1DED0 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-9):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #F2B2A9 !important;
            background: #F2B2A9 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-9):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #F2B2A9 !important;
            background: #F2B2A9 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-10):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #F2A19E !important;
            background: #F2A19E !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-10):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #F2A19E !important;
            background: #F2A19E !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-11):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #EEF2F1 !important;
            background: #EEF2F1 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-11):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #EEF2F1 !important;
            background: #EEF2F1 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-12):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #D9C6B8 !important;
            background: #D9C6B8 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-12):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #D9C6B8 !important;
            background: #D9C6B8 !important;
            filter: brightness(0.97);
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-13):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button {
            background-color: #F1DED0 !important;
            background: #F1DED0 !important;
            color: #172033 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.menu-color-13):not(:has(.menu-color-1))
        div[data-testid="stButton"] > button:hover {
            background-color: #F1DED0 !important;
            background: #F1DED0 !important;
            filter: brightness(0.97);
        }
</style>
    """,
    unsafe_allow_html=True,
)

# Highly visible medical background for the authentication page.
if BACKGROUND_IMAGE_DATA:
    st.markdown(
        f"""
        <style>
            .stApp,
            [data-testid="stAppViewContainer"] {{
                background-image:
                    linear-gradient(
                        rgba(255, 255, 255, 0.68),
                        rgba(248, 251, 255, 0.68)
                    ),
                    url("data:image/webp;base64,{BACKGROUND_IMAGE_DATA}") !important;
                background-size: 800px 800px !important;
                background-repeat: repeat !important;
                background-position: center top !important;
                background-attachment: fixed !important;
            }}

            [data-testid="stHeader"] {{
                background: rgba(11, 18, 32, 0.96) !important;
            }}

            .main .block-container {{
                padding-top: 3.25rem;
            }}

            .login-title {{
                font-size: 2.45rem !important;
                font-weight: 800 !important;
                letter-spacing: 0.01em;
                color: #111827 !important;
                text-shadow: 0 1px 2px rgba(255, 255, 255, 0.85);
            }}

            .login-subtitle {{
                color: #1f2937 !important;
                font-weight: 500;
                text-shadow: 0 1px 2px rgba(255, 255, 255, 0.9);
            }}

            div[data-testid="stForm"] {{
                background: rgba(255, 255, 255, 0.94) !important;
                border: 1px solid rgba(255, 255, 255, 0.95) !important;
                box-shadow: 0 14px 38px rgba(15, 23, 42, 0.16) !important;
                backdrop-filter: blur(7px);
                -webkit-backdrop-filter: blur(7px);
            }}

            div[data-baseweb="tab-list"] {{
                background: rgba(255, 255, 255, 0.70);
                border-radius: 12px;
                padding: 0.15rem 0.35rem;
                backdrop-filter: blur(4px);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "startup_intro_complete" not in st.session_state:
    st.session_state.startup_intro_complete = False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "active_page" not in st.session_state:
    st.session_state.active_page = "dashboard"

if "current_user" not in st.session_state:
    st.session_state.current_user = {}

LEGACY_USER_HEADERS = [
    "user_id",
    "full_name",
    "email",
    "username",
    "password_hash",
    "password_salt",
    "created_at",
    "last_login",
]

USER_HEADERS = LEGACY_USER_HEADERS + [
    "user_type",
    "other_user_type",
    "age",
    "gender",
    "date_of_birth",
    "contact_info",
    "cnic_id",
    "profile_photo",
]

USER_TYPE_OPTIONS = [
    "Patient",
    "Doctor",
    "Smart Clinic",
    "Smart E-Lab",
    "Smart E-Pharmacy",
    "Smart Hospital",
    "Admin",
    "Others",
]

PATIENT_HEADERS = [
    "user_id",
    "full_name",
    "age",
    "gender",
    "symptoms",
    "disease",
    "notes",
    "updated_at",
]

PATIENT_ID_PATTERN = re.compile(r"^P-(\d+)$", re.IGNORECASE)
USER_ID_START = 1001
USER_ID_LOCK = threading.Lock()


def generate_unique_user_id(users: list[dict[str, str]]) -> str:
    """Return the next readable patient/user ID, starting at P-1001."""
    used_numbers = set()
    for user in users:
        match = PATIENT_ID_PATTERN.fullmatch(clean_text(user.get("user_id")))
        if match:
            used_numbers.add(int(match.group(1)))

    next_number = max(used_numbers, default=USER_ID_START - 1) + 1
    while next_number in used_numbers:
        next_number += 1
    return f"P-{next_number}"




def clean_text(value: object) -> str:
    """Convert a spreadsheet value to clean text."""
    return str(value or "").strip()


def normalize_username(username: str) -> str:
    """Usernames are stored in lowercase for reliable login checks."""
    return username.strip().lower()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def valid_email(email: str) -> bool:
    return re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()) is not None


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    """Hash a password using PBKDF2. The real password is never saved."""
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000,
    ).hex()
    return password_hash, salt.hex()


def password_matches(password: str, saved_hash: str, saved_salt: str) -> bool:
    try:
        calculated_hash, _ = hash_password(password, saved_salt)
        return hmac.compare_digest(calculated_hash, saved_hash)
    except (TypeError, ValueError):
        return False


def read_google_sheet_settings() -> tuple[str, str]:
    """Read Google Sheet settings from the local google_config.json file."""
    if not GOOGLE_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "google_config.json was not found.\n"
            f"Expected location: {GOOGLE_CONFIG_FILE}\n"
            f"App location: {Path(__file__).resolve()}"
        )

    try:
        config = json.loads(
            GOOGLE_CONFIG_FILE.read_text(encoding="utf-8-sig")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"google_config.json is not valid JSON: {error}"
        ) from error

    sheet_config = config.get("google_sheet", config)
    spreadsheet_id = clean_text(sheet_config.get("spreadsheet_id"))
    worksheet_name = clean_text(
        sheet_config.get("worksheet_name")
    ) or "Users"

    if (
        not spreadsheet_id
        or spreadsheet_id == "PASTE_YOUR_SPREADSHEET_ID_HERE"
    ):
        raise ValueError(
            "Add your real Google spreadsheet ID to google_config.json."
        )

    return spreadsheet_id, worksheet_name


@st.cache_resource(show_spinner=False)
def get_users_worksheet():
    """Connect to the Google Sheet used as the user database."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # Hosted build: the read-only folder cannot hold the credential file, so
    # build the credentials straight from the secret dict when available.
    # The desktop build prefers the JSON file beside app.py; the dict and the
    # file paths are equivalent, so preferring the file keeps every desktop
    # behaviour identical.
    credentials = None
    service_account_info = _cloud_service_account_info()
    if not service_account_info and SERVICE_ACCOUNT_FILE.exists():
        try:
            credentials = Credentials.from_service_account_file(
                str(SERVICE_ACCOUNT_FILE),
                scopes=scopes,
            )
        except Exception as error:
            raise ValueError(
                "The service-account file was found, but Google could not "
                "read it.\n"
                f"File being used: {SERVICE_ACCOUNT_FILE}\n"
                f"Original error: {error}"
            ) from error
    else:
        if service_account_info:
            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=scopes,
            )
        else:
            files_in_app_folder = ", ".join(
                sorted(path.name for path in BASE_DIR.iterdir())
            )
            secret_names: list[str] = []
            try:
                items = list(getattr(st.secrets, "items", lambda: [])())
                secret_names = sorted(str(key) for key, _ in items)
            except Exception:
                secret_names = []
            env_flags = {
                key: bool(os.environ.get(key))
                for key in (
                    "CLOUD_DEPLOY",
                    "GOOGLE_SERVICE_ACCOUNT_FILE",
                    "GOOGLE_SERVICE_ACCOUNT_JSON",
                    "GOOGLE_CONFIG_JSON",
                )
            }
            raise FileNotFoundError(
                "No Google service-account key could be found.\n"
                f"Checked: {BASE_DIR / 'service_account.json'}\n"
                f"Also checked: {BASE_DIR / 'clinical_chatbot' / 'service_account.json'}\n"
                "You may also set GOOGLE_SERVICE_ACCOUNT_FILE to the key path.\n"
                f"App being executed: {Path(__file__).resolve()}\n"
                f"Files found in that folder: {files_in_app_folder}\n"
                "Streamlit secrets keys present: "
                + (", ".join(secret_names) if secret_names else "(none detected)")
                + "\n"
                "Env presence: "
                + ", ".join(f"{key}={flag}" for key, flag in env_flags.items())
            )

    client = gspread.authorize(credentials)
    spreadsheet_id, worksheet_name = read_google_sheet_settings()

    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except gspread.exceptions.SpreadsheetNotFound as error:
        raise ValueError(
            "The spreadsheet could not be opened. Check that the "
            "spreadsheet ID is correct and that the Google Sheet is "
            "shared with the client_email from service_account.json "
            "with Editor access."
        ) from error
    except (gspread.exceptions.APIError, PermissionError) as error:
        # Provide a clearer message when Google returns a 403 / permission error
        client_email = None
        try:
            sa_info = _cloud_service_account_info()
            if not sa_info and SERVICE_ACCOUNT_FILE.exists():
                sa_info = json.loads(
                    SERVICE_ACCOUNT_FILE.read_text(encoding="utf-8")
                )
            if sa_info:
                client_email = sa_info.get("client_email")
        except Exception:
            client_email = None

        email_msg = (
            f" Share the sheet with {client_email} (client_email in service_account.json) with Editor access."
            if client_email
            else " Ensure the service account has access to the sheet with Editor rights."
        )

        raise ValueError(
            "The Google API returned a permission error when opening the spreadsheet." + email_msg
        ) from error

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=1000,
            cols=len(USER_HEADERS),
        )

    first_row = [clean_text(value) for value in worksheet.row_values(1)]

    if not first_row:
        worksheet.append_row(USER_HEADERS, value_input_option="RAW")
    elif first_row != USER_HEADERS:
        # Upgrade every previous SMART CDSS user-sheet version by appending
        # only the missing columns. Existing accounts and values are preserved.
        if first_row == USER_HEADERS[:len(first_row)]:
            worksheet.resize(cols=len(USER_HEADERS))
            for column_number, header in enumerate(
                USER_HEADERS[len(first_row):],
                start=len(first_row) + 1,
            ):
                worksheet.update_cell(1, column_number, header)
        else:
            raise ValueError(
                f'The first row of the "{worksheet_name}" worksheet must begin with: '
                + ", ".join(LEGACY_USER_HEADERS)
            )

    return worksheet


@st.cache_resource(show_spinner=False)
def get_patients_worksheet():
    """Connect to the Patients worksheet in the configured Google spreadsheet."""
    import gspread
    from google.oauth2.service_account import Credentials

    if not SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(
            "No Google service-account key was found in the project root or "
            "clinical_chatbot folder. Restore the private key file, or set "
            "GOOGLE_SERVICE_ACCOUNT_FILE, before using patient search."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=scopes,
    )
    client = gspread.authorize(credentials)

    spreadsheet_id, _ = read_google_sheet_settings()
    config = json.loads(GOOGLE_CONFIG_FILE.read_text(encoding="utf-8-sig"))
    sheet_config = config.get("google_sheet", config)
    patient_worksheet_name = clean_text(
        sheet_config.get("patient_worksheet_name")
    ) or "Patients"

    spreadsheet = client.open_by_key(spreadsheet_id)
    try:
        worksheet = spreadsheet.worksheet(patient_worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=patient_worksheet_name,
            rows=1000,
            cols=len(PATIENT_HEADERS),
        )
        worksheet.append_row(PATIENT_HEADERS, value_input_option="RAW")

    first_row = [clean_text(value) for value in worksheet.row_values(1)]
    if not first_row:
        worksheet.append_row(PATIENT_HEADERS, value_input_option="RAW")
    else:
        # Keep existing patient-sheet data/columns intact and add only any
        # required search columns that are missing.
        missing_headers = [
            header for header in PATIENT_HEADERS
            if header not in first_row
        ]
        if missing_headers:
            required_columns = len(first_row) + len(missing_headers)
            if worksheet.col_count < required_columns:
                worksheet.resize(cols=required_columns)
            for column_number, header in enumerate(
                missing_headers,
                start=len(first_row) + 1,
            ):
                worksheet.update_cell(1, column_number, header)

    return worksheet


# ---------------------------------------------------------
# Service subscriptions (payment records live in the shared
# spreadsheet). Admins mark a service Active / Pending / Revoked
# after verifying payment; the app then unlocks or blocks it for
# the patient. A patient keeps working by default until a row
# exists for their account, so no access regressions.
# ---------------------------------------------------------
PAYMENT_SHEET_NAME = "Payments"
PAYMENT_HEADERS = [
    "patient_id",
    "patient_name",
    "service",
    "status",
    "amount",
    "payment_reference",
    "verified_by",
    "updated_at",
]
SERVICE_KEY_TO_LABEL = {
    "clinical_chatbot": "Clinical Chatbot",
    "full_diagnostics": "Full Diagnostics",
    "brain_stroke": "Brain Stroke Diagnosis",
    "heart_disease": "Heart Disease Diagnosis",
    "breast_cancer": "Breast Cancer Diagnosis",
}


def _authorized_sheets_client():
    """Return an authorized gspread client, or None when unavailable."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    service_account_info = _cloud_service_account_info()
    if not service_account_info and SERVICE_ACCOUNT_FILE.exists():
        try:
            credentials = Credentials.from_service_account_file(
                str(SERVICE_ACCOUNT_FILE), scopes=scopes
            )
            return gspread.authorize(credentials)
        except Exception:
            return None
    if service_account_info:
        try:
            credentials = Credentials.from_service_account_info(
                service_account_info, scopes=scopes
            )
            return gspread.authorize(credentials)
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def get_payments_worksheet():
    """Connect to (and create if needed) the Payments worksheet."""
    import gspread

    client = _authorized_sheets_client()
    if client is None:
        raise ValueError(
            "Service records unavailable: no Google service-account credentials found."
        )
    spreadsheet_id, _ = read_google_sheet_settings()
    spreadsheet = client.open_by_key(spreadsheet_id)
    try:
        worksheet = spreadsheet.worksheet(PAYMENT_SHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=PAYMENT_SHEET_NAME, rows=1000, cols=len(PAYMENT_HEADERS)
        )
    first_row = [clean_text(value) for value in worksheet.row_values(1)]
    if not first_row:
        worksheet.append_row(PAYMENT_HEADERS, value_input_option="RAW")
    elif first_row != PAYMENT_HEADERS:
        if first_row == PAYMENT_HEADERS[: len(first_row)]:
            worksheet.resize(cols=len(PAYMENT_HEADERS))
            for column_number, header in enumerate(
                PAYMENT_HEADERS[len(first_row):],
                start=len(first_row) + 1,
            ):
                worksheet.update_cell(1, column_number, header)
        else:
            raise ValueError(
                f'The first row of the "{PAYMENT_SHEET_NAME}" worksheet must be: '
                + ", ".join(PAYMENT_HEADERS)
            )
    return worksheet


def list_service_subscriptions() -> list[dict[str, object]]:
    """All service-subscription rows (data rows only). Never raises."""
    rows: list[dict[str, object]] = []
    try:
        worksheet = get_payments_worksheet()
        records = worksheet.get_all_records(value_render_option="FORMATTED_VALUE")
    except Exception:
        return rows
    for index, record in enumerate(records, start=2):
        patient_id = clean_text(record.get("patient_id"))
        if not patient_id:
            continue
        rows.append(
            {
                "row": index,
                "patient_id": patient_id,
                "patient_name": clean_text(record.get("patient_name")),
                "service": clean_text(record.get("service")),
                "status": clean_text(record.get("status")).lower(),
                "amount": clean_text(record.get("amount")),
                "payment_reference": clean_text(record.get("payment_reference")),
                "verified_by": clean_text(record.get("verified_by")),
                "updated_at": clean_text(record.get("updated_at")),
            }
        )
    return rows


def get_service_subscription(patient_id: str, service: str) -> dict[str, object] | None:
    """Latest subscription row for a patient+service, or None."""
    patient_id = clean_text(patient_id)
    service = clean_text(service).lower()
    if not patient_id or not service:
        return None
    latest = None
    for row in list_service_subscriptions():
        if row["patient_id"] == patient_id and row["service"].lower() == service:
            latest = row
    return latest


def upsert_service_subscription(
    patient_id: str,
    patient_name: str,
    service: str,
    status: str,
    amount: str = "",
    payment_reference: str = "",
    verified_by: str = "admin",
) -> bool:
    """Update an existing patient+service row, or append a new one."""
    patient_id = clean_text(patient_id)
    service_key = clean_text(service).lower() or "other"
    status = clean_text(status).lower()
    if not patient_id:
        return False
    try:
        worksheet = get_payments_worksheet()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        values = [
            patient_id,
            clean_text(patient_name) or patient_id,
            service_key,
            status,
            clean_text(amount),
            clean_text(payment_reference),
            clean_text(verified_by) or "admin",
            now,
        ]
        current = get_service_subscription(patient_id, service_key)
        if current and current.get("row"):
            row_number = int(current["row"])
            worksheet.update(
                [[value for value in values]],
                f"A{row_number}:H{row_number}",
                value_input_option="RAW",
            )
        else:
            worksheet.append_row(values, value_input_option="RAW")
        return True
    except Exception:
        return False


def service_access_allowed(
    patient_id: str, service: str
) -> tuple[bool, str, dict[str, object]]:
    """Allow by default (no rows yet); block only when an explicit row exists
    and its status is not 'active'. Doctors and admins always have access."""
    if current_user_is_admin() or current_user_is_doctor():
        return True, "", {}
    patient_id = clean_text(patient_id)
    if not patient_id:
        return True, "", {}
    subscription = get_service_subscription(patient_id, service)
    if subscription is None:
        return True, "", {}
    status = str(subscription.get("status", "active")).lower()
    if status == "active":
        return True, "", subscription
    label = SERVICE_KEY_TO_LABEL.get(str(service).lower(), str(service))
    message = (
        f"The **{label}** service is currently **{status.title()}** for your "
        "account. Contact your clinic or administrator to activate it, or "
        "check your payment status."
    )
    return False, message, subscription


def get_all_patients() -> list[dict[str, str]]:
    """Return clinical records plus newly registered patient accounts."""
    worksheet = get_patients_worksheet()
    records = worksheet.get_all_records(expected_headers=PATIENT_HEADERS)
    patients = [
        {key: clean_text(value) for key, value in record.items()}
        for record in records
    ]

    # A patient is searchable immediately after sign-up, even before a
    # clinical record has been added to the Patients worksheet.
    existing_ids = {
        clean_text(patient.get("user_id")).casefold()
        for patient in patients
    }
    for user in get_all_users():
        user_id = clean_text(user.get("user_id"))
        if (
            clean_text(user.get("user_type")).casefold() == "patient"
            and user_id
            and user_id.casefold() not in existing_ids
        ):
            patients.append(
                {
                    "user_id": user_id,
                    "full_name": clean_text(user.get("full_name")),
                    "age": clean_text(user.get("age")),
                    "gender": clean_text(user.get("gender")),
                    "symptoms": "",
                    "disease": "",
                    "notes": "",
                    "updated_at": clean_text(user.get("created_at")),
                }
            )

    return patients


def _contains_query(value: object, query: str) -> bool:
    """Case-insensitive substring matching for patient search fields."""
    return query.casefold() in clean_text(value).casefold()


def search_patients(
    user_id: str = "",
    symptoms: str = "",
    disease: str = "",
) -> list[dict[str, str]]:
    """Search patients using any combination of ID, symptoms, and disease."""
    user_id = clean_text(user_id)
    symptoms = clean_text(symptoms)
    disease = clean_text(disease)

    results = []
    for patient in get_all_patients():
        if user_id and clean_text(patient.get("user_id")).casefold() != user_id.casefold():
            continue
        if symptoms and not _contains_query(patient.get("symptoms"), symptoms):
            continue
        if disease and not _contains_query(patient.get("disease"), disease):
            continue
        results.append(patient)

    return results


def google_sheet_ready() -> tuple[bool, str]:
    try:
        get_users_worksheet()
        return True, ""
    except Exception as error:
        return False, str(error)


def get_all_users() -> list[dict[str, str]]:
    worksheet = get_users_worksheet()
    records = worksheet.get_all_records(expected_headers=USER_HEADERS)
    return [
        {key: clean_text(value) for key, value in record.items()}
        for record in records
    ]


def _normalized_header(value: object) -> str:
    """Normalize legacy Google Sheet column names."""
    name = clean_text(value).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    aliases = {
        "name": "full_name",
        "full_name": "full_name",
        "fullname": "full_name",
        "email_address": "email",
        "email": "email",
        "user_name": "username",
        "username": "username",
        "password_hash": "password_hash",
        "passwordhash": "password_hash",
        "password_salt": "password_salt",
        "passwordsalt": "password_salt",
        "created_at": "created_at",
        "last_login": "last_login",
        "user_id": "user_id",
        "userid": "user_id",
        "user_type": "user_type",
        "other_user_type": "other_user_type",
        "gender": "gender",
        "age": "age",
        "date_of_birth": "date_of_birth",
        "contact_info": "contact_info",
        "cnic_id": "cnic_id",
        "profile_photo": "profile_photo",
    }
    return aliases.get(name, name)


def get_legacy_users() -> list[dict[str, str]]:
    """
    Read accounts created by older releases from their previous worksheet.

    Earlier builds used "User Registration Responses".  The current build uses
    the dedicated "Users" worksheet.  This compatibility reader prevents old
    accounts from becoming impossible to log into after the worksheet upgrade.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    spreadsheet_id, current_worksheet_name = read_google_sheet_settings()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=scopes,
    )
    spreadsheet = gspread.authorize(credentials).open_by_key(spreadsheet_id)

    legacy_names = [
        "User Registration Responses",
        "Form Responses 1",
        "Form Responses",
    ]

    legacy_users: list[dict[str, str]] = []
    for worksheet_name in legacy_names:
        if worksheet_name == current_worksheet_name:
            continue
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            continue

        values = worksheet.get_all_values()
        if len(values) < 2:
            continue

        headers = [_normalized_header(value) for value in values[0]]
        # Only treat a sheet as an old account database when it contains the
        # fields needed for secure password verification.
        required = {"email", "username", "password_hash", "password_salt"}
        if not required.issubset(set(headers)):
            continue

        for row_number, values_row in enumerate(values[1:], start=2):
            padded = list(values_row) + [""] * max(0, len(headers) - len(values_row))
            record = {
                header: clean_text(padded[index])
                for index, header in enumerate(headers)
                if header
            }
            if not record.get("username") and not record.get("email"):
                continue
            record["__legacy_worksheet"] = worksheet_name
            record["__legacy_row"] = str(row_number)
            legacy_users.append(record)

    return legacy_users


def find_user(identifier: str) -> dict[str, str] | None:
    normalized = identifier.strip().lower()

    # Current account database first.
    for user in get_all_users():
        if normalize_username(user.get("username", "")) == normalized:
            return user
        if normalize_email(user.get("email", "")) == normalized:
            return user

    # Backward compatibility for accounts made before the Users worksheet
    # migration.  These are moved into Users after a successful login.
    for user in get_legacy_users():
        if normalize_username(user.get("username", "")) == normalized:
            return user
        if normalize_email(user.get("email", "")) == normalized:
            return user

    return None



def find_user_by_id(user_id: str) -> dict[str, str] | None:
    """Find a current account by its assigned unique User ID."""
    normalized_user_id = clean_text(user_id).casefold()
    if not normalized_user_id:
        return None

    for user in get_all_users():
        if clean_text(user.get("user_id")).casefold() == normalized_user_id:
            return user
    return None


def reset_password_by_user_id(
    user_id: str,
    account_identifier: str,
    new_password: str,
) -> None:
    """
    Reset a current account password after matching its unique User ID and
    the username/email already stored on that account.
    """
    user = find_user_by_id(user_id)
    if not user:
        raise ValueError("No account was found with that User ID.")

    normalized_identifier = account_identifier.strip().lower()
    username_matches = (
        normalize_username(user.get("username", "")) == normalized_identifier
    )
    email_matches = (
        normalize_email(user.get("email", "")) == normalized_identifier
    )
    if not (username_matches or email_matches):
        raise ValueError(
            "The username/email does not match the account for this User ID."
        )

    if len(new_password) < 8:
        raise ValueError("The new password must be at least 8 characters long.")

    password_hash, password_salt = hash_password(new_password)
    worksheet = get_users_worksheet()
    user_id_cell = worksheet.find(clean_text(user.get("user_id")), in_column=1)

    password_hash_column = USER_HEADERS.index("password_hash") + 1
    password_salt_column = USER_HEADERS.index("password_salt") + 1
    worksheet.update_cell(user_id_cell.row, password_hash_column, password_hash)
    worksheet.update_cell(user_id_cell.row, password_salt_column, password_salt)

def migrate_legacy_user(
    user: dict[str, str],
    selected_user_type: str = "",
    selected_other_type: str = "",
) -> dict[str, str]:
    """Move a successfully authenticated legacy account into the Users sheet."""
    if not user.get("__legacy_worksheet"):
        return user

    worksheet = get_users_worksheet()
    existing_users = get_all_users()

    legacy_email = normalize_email(user.get("email", ""))
    legacy_username = normalize_username(user.get("username", ""))

    # If another run already migrated it, use that canonical row.
    for existing in existing_users:
        if legacy_username and normalize_username(existing.get("username", "")) == legacy_username:
            return existing
        if legacy_email and normalize_email(existing.get("email", "")) == legacy_email:
            return existing

    user_id = clean_text(user.get("user_id"))
    used_ids = {clean_text(item.get("user_id")).casefold() for item in existing_users}
    if not user_id or user_id.casefold() in used_ids:
        user_id = generate_unique_user_id(existing_users)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    saved_user_type = clean_text(user.get("user_type")) or clean_text(selected_user_type)
    saved_other_type = clean_text(user.get("other_user_type"))
    if saved_user_type == "Others" and not saved_other_type:
        saved_other_type = clean_text(selected_other_type)

    migrated = {
        "user_id": user_id,
        "full_name": clean_text(user.get("full_name")),
        "email": legacy_email,
        "username": legacy_username,
        "password_hash": clean_text(user.get("password_hash")),
        "password_salt": clean_text(user.get("password_salt")),
        "created_at": clean_text(user.get("created_at")) or now,
        "last_login": now,
        "user_type": saved_user_type,
        "other_user_type": saved_other_type,
        "age": clean_text(user.get("age")),
        "gender": clean_text(user.get("gender")),
        "date_of_birth": clean_text(user.get("date_of_birth")),
        "contact_info": clean_text(user.get("contact_info")),
        "cnic_id": clean_text(user.get("cnic_id")),
        "profile_photo": clean_text(user.get("profile_photo")),
    }

    worksheet.append_row(
        [migrated.get(header, "") for header in USER_HEADERS],
        value_input_option="RAW",
    )
    return migrated


def create_user(
    full_name: str,
    email: str,
    username: str,
    password: str,
    user_type: str,
    gender: str,
    other_user_type: str = "",
) -> dict[str, str]:
    worksheet = get_users_worksheet()

    normalized_email = normalize_email(email)
    normalized_username = normalize_username(username)

    # Keep ID allocation and account insertion together inside this process so
    # two simultaneous sign-ups cannot receive the same readable ID.
    with USER_ID_LOCK:
        users = get_all_users()

        for existing_user in users:
            if normalize_email(existing_user.get("email", "")) == normalized_email:
                raise ValueError("An account with this email already exists.")
            if normalize_username(existing_user.get("username", "")) == normalized_username:
                raise ValueError("This username is already taken.")

        user_id = generate_unique_user_id(users)
        password_hash, password_salt = hash_password(password)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    normalized_user_type = clean_text(user_type)
    normalized_other_type = (
        clean_text(other_user_type)
        if normalized_user_type == "Others"
        else ""
    )

    row = [
        user_id,
        full_name.strip(),
        normalized_email,
        normalized_username,
        password_hash,
        password_salt,
        now,
        now,
        normalized_user_type,
        normalized_other_type,
        "",  # age
        clean_text(gender),
        "",  # date_of_birth
        "",  # contact_info
        "",  # cnic_id
        "",  # profile_photo
    ]
    with USER_ID_LOCK:
        # Recheck immediately before writing in case this function is changed
        # later to do more work between ID generation and insertion.
        current_ids = {
            clean_text(user.get("user_id")).casefold()
            for user in get_all_users()
        }
        if user_id.casefold() in current_ids:
            user_id = generate_unique_user_id(get_all_users())
            row[0] = user_id
        worksheet.append_row(row, value_input_option="RAW")

    return {
        "user_id": user_id,
        "full_name": full_name.strip(),
        "email": normalized_email,
        "username": normalized_username,
        "user_type": normalized_user_type,
        "other_user_type": normalized_other_type,
        "age": "",
        "gender": clean_text(gender),
        "date_of_birth": "",
        "contact_info": "",
        "cnic_id": "",
        "profile_photo": "",
    }


def update_last_login(user_id: str) -> None:
    """Update the user's last login time. Login still works if this update fails."""
    try:
        worksheet = get_users_worksheet()
        user_id_cell = worksheet.find(user_id, in_column=1)
        last_login_column = USER_HEADERS.index("last_login") + 1
        worksheet.update_cell(
            user_id_cell.row,
            last_login_column,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except Exception:
        pass


def save_login_session(user: dict[str, str]) -> None:
    st.session_state.logged_in = True
    st.session_state.current_user = {
        key: clean_text(user.get(key, ""))
        for key in [
            "user_id",
            "full_name",
            "email",
            "username",
            "user_type",
            "other_user_type",
            "age",
            "gender",
            "date_of_birth",
            "contact_info",
            "cnic_id",
            "profile_photo",
        ]
    }
    st.session_state.active_page = "dashboard"


def update_user_profile(user_id: str, profile: dict[str, str]) -> dict[str, str]:
    """Update editable profile fields while keeping the assigned User ID fixed."""
    worksheet = get_users_worksheet()
    user_id_cell = worksheet.find(user_id, in_column=1)
    editable_fields = [
        "full_name",
        "email",
        "age",
        "gender",
        "date_of_birth",
        "contact_info",
        "cnic_id",
        "profile_photo",
    ]

    for field in editable_fields:
        column_number = USER_HEADERS.index(field) + 1
        worksheet.update_cell(
            user_id_cell.row,
            column_number,
            clean_text(profile.get(field, "")),
        )

    updated_user = next(
        (
            user for user in get_all_users()
            if clean_text(user.get("user_id")).casefold() == user_id.casefold()
        ),
        None,
    )
    if not updated_user:
        raise ValueError("The updated user account could not be reloaded.")
    return updated_user


def save_profile_photo(user_id: str, uploaded_file) -> str:
    """Validate and save a profile image locally; return its relative path."""
    if uploaded_file is None:
        return ""

    allowed_types = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    suffix = allowed_types.get(clean_text(uploaded_file.type).lower())
    file_bytes = uploaded_file.getvalue()

    if suffix is None:
        raise ValueError("Profile photo must be a JPG, PNG, or WEBP image.")
    if len(file_bytes) > 5 * 1024 * 1024:
        raise ValueError("Profile photo must be 5 MB or smaller.")

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", user_id)
    for old_file in PROFILE_PHOTO_DIR.glob(f"{safe_id}.*"):
        try:
            old_file.unlink()
        except OSError:
            pass

    destination = PROFILE_PHOTO_DIR / f"{safe_id}{suffix}"
    destination.write_bytes(file_bytes)
    return str(destination.relative_to(BASE_DIR)).replace("\\", "/")


def profile_photo_path(user: dict[str, str]) -> Path:
    saved_path = clean_text(user.get("profile_photo"))
    if saved_path:
        candidate = (BASE_DIR / saved_path).resolve()
        try:
            candidate.relative_to(BASE_DIR.resolve())
            if candidate.is_file():
                return candidate
        except ValueError:
            pass

    normalized_gender = clean_text(user.get("gender")).casefold()
    if normalized_gender == "female":
        return DEFAULT_FEMALE_PROFILE_FILE
    if normalized_gender == "other":
        return DEFAULT_OTHER_PROFILE_FILE
    return DEFAULT_MALE_PROFILE_FILE


def login_page() -> None:
    # UI VERSION: USER_TYPE_INSIDE_LOGIN_FORM_V3
    left, center, right = st.columns([0.65, 1.35, 1.0], gap="large")

    with center:
        st.markdown("<h1 class='login-title'>Integrated Health Services (IHS) | Sehat Plus</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='login-subtitle'>Sign in with the username and password issued by your clinic or administrator.</p>",
            unsafe_allow_html=True,
        )

        sheet_ready, sheet_error = google_sheet_ready()
        if not sheet_ready:
            st.error("Google Sheets connection/setup failed.")
            st.info(
                "The app found its local Google Sheets configuration, but could not "
                "finish opening or preparing the user database. See the exact cause below."
            )
            with st.expander("Technical error", expanded=True):
                st.code(sheet_error)
            st.stop()

        login_tab, = st.tabs(["Log in"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                login_user_type = st.selectbox(
                    "User type",
                    USER_TYPE_OPTIONS,
                    key="login_user_type",
                )

                login_other_user_type = ""
                if login_user_type == "Others":
                    login_other_user_type = st.text_input(
                        "Specify user type",
                        placeholder="Enter your user type",
                        key="login_other_user_type",
                    )

                identifier = st.text_input(
                    "Username or email",
                    placeholder="Enter username or email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter password",
                )
                submitted = st.form_submit_button("Log in", use_container_width=True)

            if submitted:
                if not identifier.strip() or not password:
                    st.warning("Enter your username/email and password.")
                else:
                    try:
                        user = find_user(identifier)
                        password_is_valid = bool(user) and password_matches(
                            password,
                            user.get("password_hash", ""),
                            user.get("password_salt", ""),
                        )

                        saved_user_type = clean_text(user.get("user_type", "")) if user else ""
                        saved_other_type = clean_text(user.get("other_user_type", "")) if user else ""

                        # Existing accounts created before this update may have no saved type.
                        type_is_valid = (
                            not saved_user_type
                            or saved_user_type == login_user_type
                        )
                        if login_user_type == "Others" and saved_user_type:
                            type_is_valid = (
                                type_is_valid
                                and bool(login_other_user_type.strip())
                                and saved_other_type.lower() == login_other_user_type.strip().lower()
                            )

                        if password_is_valid and type_is_valid:
                            # Old releases stored accounts in a different worksheet.
                            # After the password is verified, migrate that account into
                            # the current Users worksheet automatically.
                            user = migrate_legacy_user(
                                user,
                                selected_user_type=login_user_type,
                                selected_other_type=login_other_user_type,
                            )
                            save_login_session(user)
                            update_last_login(user.get("user_id", ""))
                            st.rerun()
                        elif password_is_valid and not type_is_valid:
                            st.error("The selected user type does not match this account.")
                        else:
                            if not user:
                                st.error(
                                    "Account not found. If this is an older account, make sure the "
                                    'original "User Registration Responses" worksheet is still in '
                                    "the same Google spreadsheet."
                                )
                            else:
                                st.error("Incorrect username, email, or password.")
                    except Exception as error:
                        st.error("Login could not be checked because Google Sheets is unavailable.")
                        with st.expander("Technical error"):
                            st.code(str(error))


        st.markdown("<div style='height: 0.35rem'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <style>
            .st-key-reset_password_block details {
                background-color: #DEE0E4 !important;
                border-color: #BFC3C9 !important;
            }
            .st-key-reset_password_block details > summary {
                background-color: #DEE0E4 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="reset_password_block"):
            with st.expander("Forgot password? Reset it here", expanded=False):
                st.caption(
                    "Use your unique User ID and the username or email already linked "
                    "to your account to choose a new password."
                )
                with st.form("reset_password_form", clear_on_submit=True):
                    reset_user_id = st.text_input(
                        "Unique User ID",
                        placeholder="Example: P-1001",
                        key="reset_user_id",
                    )
                    reset_identifier = st.text_input(
                        "Registered username or email",
                        placeholder="Enter the username or email on this account",
                        key="reset_identifier",
                    )
                    reset_new_password = st.text_input(
                        "New password",
                        type="password",
                        placeholder="Minimum 8 characters",
                        key="reset_new_password",
                    )
                    reset_confirm_password = st.text_input(
                        "Confirm new password",
                        type="password",
                        placeholder="Re-enter the new password",
                        key="reset_confirm_password",
                    )
                    reset_submitted = st.form_submit_button(
                        "Reset password",
                        use_container_width=True,
                    )

                if reset_submitted:
                    if (
                        not reset_user_id.strip()
                        or not reset_identifier.strip()
                        or not reset_new_password
                        or not reset_confirm_password
                    ):
                        st.warning("Complete all password reset fields.")
                    elif reset_new_password != reset_confirm_password:
                        st.error("The new passwords do not match.")
                    elif len(reset_new_password) < 8:
                        st.warning("The new password must be at least 8 characters long.")
                    else:
                        try:
                            reset_password_by_user_id(
                                reset_user_id,
                                reset_identifier,
                                reset_new_password,
                            )
                            st.success(
                                "Password reset successfully. You can now log in with "
                                "your new password."
                            )
                        except ValueError as error:
                            st.error(str(error))
                        except Exception as error:
                            st.error(
                                "Password reset could not be completed because the "
                                "user database is unavailable."
                            )
                            with st.expander("Technical error"):
                                st.code(str(error))



        # Restore the original SMILE banner below the login panel.
        if SMILE_BANNER_FILE.exists():
            smile_banner_data = base64.b64encode(
                SMILE_BANNER_FILE.read_bytes()
            ).decode("ascii")
            st.markdown(
                f"""
                <div class="login-smile-banner">
                    <img src="data:image/png;base64,{smile_banner_data}" alt="SMILE - Care of Smart Clinic">
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Restore the original IHS Solutions side menu.
    with right:
        with st.container(border=True, height=575):
            st.markdown(
                "<div class='ihs-menu-marker'></div>"
                "<div class='module-menu-heading'>IHS Solutions Menu</div>"
                "<div class='module-menu-subtitle'>Standalone modules</div>",
                unsafe_allow_html=True,
            )

            menu_items = [
                ("Smart Clinic", "menu-color-3"),
                ("Nutrition Advisor", "menu-color-1"),
                ("Medication Safety Assistant", "menu-color-2"),
                ("Early Warning System", "menu-color-3"),
                ("Patient Experience", "menu-color-4"),
                ("AI Models / Data Pipeline", "menu-color-5"),
                ("Infection Control Voice Bot", "menu-color-6"),
                ("Patient Fall Detection Alerts", "menu-color-7"),
                ("Workforce Monitoring / Optimization", "menu-color-9"),
                ("Personalized Pain Relief Agent", "menu-color-10"),
                ("Smart Consent", "menu-color-11"),
                ("Smart Clinics Command Center", "menu-color-12"),
                ("IHS Channel Partners", "menu-color-13"),
                ("Health-Tech Marketplace", "menu-color-14"),
                ("Get Insured", "menu-color-15"),
            ]

            if "login_selected_module" not in st.session_state:
                st.session_state.login_selected_module = "Smart Clinic"

            for menu_label, color_class in menu_items:
                with st.container():
                    st.markdown(
                        f"<span class='{color_class}' style='display:none'></span>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        menu_label,
                        key=f"login_menu_{re.sub(r'[^a-z0-9]+', '_', menu_label.lower()).strip('_')}",
                        use_container_width=True,
                    ):
                        st.session_state.login_selected_module = menu_label

            st.markdown(
                f"<div class='module-menu-selected'>Selected: "
                f"{st.session_state.login_selected_module}</div>",
                unsafe_allow_html=True,
            )

def top_navigation(title: str) -> None:
    title_column, user_column, button_column = st.columns([4.2, 1.4, 1])

    with title_column:
        if title:
            st.markdown(f"### {title}")

    with user_column:
        display_name = st.session_state.current_user.get("full_name", "User")
        st.caption(f"Signed in as {display_name}")

    with button_column:
        if st.button("Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = {}
            st.session_state.active_page = "dashboard"
            st.rerun()


def tool_card(
    icon: str,
    title: str,
    description: str,
    ready: bool = False,
    icon_image=None,
) -> None:
    status_class = "status-ready" if ready else "status-soon"
    status_text = "Ready" if ready else "Coming soon"

    icon_html = icon
    if icon_image is not None and icon_image.is_file():
        encoded_icon = base64.b64encode(icon_image.read_bytes()).decode("ascii")
        icon_html = (
            f'<img src="data:image/png;base64,{encoded_icon}" '
            f'alt="{title} icon">'
        )

    st.markdown(
        f"""
        <div class="tool-card">
            <div class="tool-icon">{icon_html}</div>
            <div class="tool-title">{title}</div>
            <div class="tool-text">{description}</div>
            <span class="{status_class}">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def current_user_is_doctor() -> bool:
    """Return True only when the signed-in account is registered as a doctor."""
    return clean_text(
        st.session_state.get("current_user", {}).get("user_type", "")
    ).casefold() == "doctor"


def current_user_is_admin() -> bool:
    """Return True only when the signed-in account is registered as an admin."""
    return clean_text(
        st.session_state.get("current_user", {}).get("user_type", "")
    ).casefold() == "admin"



def _medical_history_defaults() -> dict:
    """Return a new blank medical-history record."""
    return {
        "conditions": "",
        "past_events": "",
        "current_medications": "",
        "past_medications": "",
        "allergies": "",
        "family_history": "",
        "smoking": "Never smoked",
        "alcohol_use": "None",
        "exercise_level": "Low",
        "diet": "",
        "sleep_pattern": "",
        "vaccination_history": "",
        "pregnancy_history": "",
        "blood_transfusion_history": "",
        "mental_health_history": "",
        "implanted_device": "",
        "other_information": "",
        "updated_at": "",
    }


def _read_medical_history_records_unlocked() -> dict:
    """Read saved records and migrate an older local file once if present."""
    MEDICAL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Automatically migrate medical history saved by an older build
    # when that older data file is present beside app.py.
    if not MEDICAL_HISTORY_FILE.is_file() and LEGACY_MEDICAL_HISTORY_FILE.is_file():
        try:
            legacy_records = json.loads(
                LEGACY_MEDICAL_HISTORY_FILE.read_text(encoding="utf-8")
            )
            if isinstance(legacy_records, dict):
                temp_file = MEDICAL_HISTORY_FILE.with_suffix(".json.tmp")
                temp_file.write_text(
                    json.dumps(legacy_records, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                temp_file.replace(MEDICAL_HISTORY_FILE)
        except (OSError, ValueError, TypeError):
            pass

    if not MEDICAL_HISTORY_FILE.is_file():
        return {}

    try:
        records = json.loads(MEDICAL_HISTORY_FILE.read_text(encoding="utf-8"))
        return records if isinstance(records, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def load_medical_history(user_id: str) -> dict:
    """Load the latest permanently saved history for one patient."""
    history = _medical_history_defaults()
    patient_key = clean_text(user_id)
    if not patient_key:
        return history

    with MEDICAL_HISTORY_LOCK:
        records = _read_medical_history_records_unlocked()
        saved = records.get(patient_key, {})

    if isinstance(saved, dict):
        for key in history:
            if key in saved:
                history[key] = clean_text(saved.get(key))

    return history


def load_latest_diagnostic_report(user_id: str) -> dict:
    """Load the latest saved diagnostic report for one patient."""
    default = {
        "report_content": "",
        "generated_at": "",
        "patient_info": {},
    }
    try:
        if DIAGNOSTIC_REPORTS_FILE.is_file():
            records = json.loads(DIAGNOSTIC_REPORTS_FILE.read_text(encoding="utf-8"))
            saved = records.get(clean_text(user_id), {})
            if isinstance(saved, dict):
                default.update(saved)
    except (OSError, ValueError, TypeError):
        pass
    return default


def save_latest_diagnostic_report(
    user_id: str,
    report_content: str,
    patient_info: dict,
) -> None:
    """Persist the newest diagnostic report for a patient."""
    patient_key = clean_text(user_id)
    if not patient_key:
        return

    records = {}
    try:
        if DIAGNOSTIC_REPORTS_FILE.is_file():
            loaded = json.loads(DIAGNOSTIC_REPORTS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                records = loaded
    except (OSError, ValueError, TypeError):
        records = {}

    records[patient_key] = {
        "report_content": str(report_content),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patient_info": {
            key: clean_text(value)
            for key, value in patient_info.items()
        },
    }

    temporary_file = DIAGNOSTIC_REPORTS_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_file.replace(DIAGNOSTIC_REPORTS_FILE)


def save_medical_history(user_id: str, history: dict) -> None:
    """Persist medical history until the same patient explicitly saves an edit."""
    patient_key = clean_text(user_id)
    if not patient_key:
        raise ValueError("A valid signed-in user ID is required.")

    clean_history = {
        key: clean_text(value)
        for key, value in history.items()
    }
    clean_history["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    with MEDICAL_HISTORY_LOCK:
        records = _read_medical_history_records_unlocked()
        records[patient_key] = clean_history

        MEDICAL_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = MEDICAL_HISTORY_FILE.with_suffix(".json.tmp")
        temporary_file.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_file.replace(MEDICAL_HISTORY_FILE)


def medical_history_tab(user_id: str) -> None:
    """Render the editable patient medical-history interface."""
    history = load_medical_history(user_id)

    st.markdown(
        """
        <div class="hero-box">
            <h1>🩺 Patient Medical History</h1>
            <p>Maintain conditions, medications, allergies, family history and lifestyle information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history.get("updated_at"):
        st.caption(f"Last updated: {history['updated_at']}")

    with st.form("medical_history_form", border=True, clear_on_submit=False):
        st.subheader("Medical History")
        left, right = st.columns(2, gap="large")
        with left:
            conditions = st.text_area(
                "Medical Conditions",
                value=history["conditions"],
                height=140,
                placeholder="One condition per line, e.g. Diabetes — Type 2 (Diagnosed 2018)",
            )
            current_medications = st.text_area(
                "Current Medications",
                value=history["current_medications"],
                height=140,
                placeholder="Medicine | Dose | Frequency",
            )
        with right:
            past_events = st.text_area(
                "Past Medical Events",
                value=history["past_events"],
                height=140,
                placeholder="Event | Details | Date",
            )
            allergies = st.text_area(
                "Allergies",
                value=history["allergies"],
                height=140,
                placeholder="Allergen | Reaction type | Reaction",
            )

        second_left, second_right = st.columns(2, gap="large")
        with second_left:
            past_medications = st.text_area(
                "Past Medications",
                value=history["past_medications"],
                height=110,
            )
            family_history = st.text_area(
                "Family History",
                value=history["family_history"],
                height=140,
                placeholder="Condition | Relation | Notes",
            )
        with second_right:
            st.subheader("Lifestyle History")
            smoking = st.selectbox(
                "Smoking",
                ["Never smoked", "Former smoker", "Current smoker", "Prefer not to say"],
                index=["Never smoked", "Former smoker", "Current smoker", "Prefer not to say"].index(history["smoking"])
                if history["smoking"] in ["Never smoked", "Former smoker", "Current smoker", "Prefer not to say"] else 0,
            )
            alcohol_use = st.text_input("Alcohol Use", value=history["alcohol_use"])
            exercise_level = st.selectbox(
                "Exercise Level",
                ["Low", "Moderate", "High"],
                index=["Low", "Moderate", "High"].index(history["exercise_level"])
                if history["exercise_level"] in ["Low", "Moderate", "High"] else 0,
            )
            diet = st.text_input("Diet", value=history["diet"])
            sleep_pattern = st.text_input("Sleep Pattern", value=history["sleep_pattern"])

        st.subheader("Other Useful Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            vaccination_history = st.text_area(
                "Vaccination History", value=history["vaccination_history"], height=100
            )
            pregnancy_history = st.text_area(
                "Pregnancy History", value=history["pregnancy_history"], height=100
            )
        with c2:
            blood_transfusion_history = st.text_area(
                "Blood Transfusion History",
                value=history["blood_transfusion_history"],
                height=100,
            )
            mental_health_history = st.text_area(
                "Mental Health History", value=history["mental_health_history"], height=100
            )
        with c3:
            implanted_device = st.text_area(
                "Implanted Medical Device", value=history["implanted_device"], height=100
            )
            other_information = st.text_area(
                "Other Information", value=history["other_information"], height=100
            )

        save_history = st.form_submit_button(
            "Save Medical History",
            type="primary",
            use_container_width=True,
        )

    if save_history:
        try:
            save_medical_history(
                user_id,
                {
                    "conditions": conditions,
                    "past_events": past_events,
                    "current_medications": current_medications,
                    "past_medications": past_medications,
                    "allergies": allergies,
                    "family_history": family_history,
                    "smoking": smoking,
                    "alcohol_use": alcohol_use,
                    "exercise_level": exercise_level,
                    "diet": diet,
                    "sleep_pattern": sleep_pattern,
                    "vaccination_history": vaccination_history,
                    "pregnancy_history": pregnancy_history,
                    "blood_transfusion_history": blood_transfusion_history,
                    "mental_health_history": mental_health_history,
                    "implanted_device": implanted_device,
                    "other_information": other_information,
                },
            )
            st.success("Medical history saved successfully.")
        except Exception as error:
            st.error("The medical history could not be saved.")
            with st.expander("Technical error"):
                st.code(str(error))




# ---------------------------------------------------------
# Breast Cancer Diagnosis integration
# ---------------------------------------------------------
BREAST_CANCER_MODEL_FILE = (
    BASE_DIR / "breast_cancer" / "model" / "breast_cancer_model.joblib"
)
BREAST_CANCER_CLASS_INDEX_FILE = (
    BASE_DIR / "breast_cancer" / "model" / "class_indices.json"
)
BREAST_CANCER_IMG_SIZE = (128, 128)


@st.cache_resource(show_spinner="Loading the Breast Cancer Diagnosis model...")
def load_breast_cancer_model():
    """Load and cache the supplied breast-ultrasound classifier."""
    import joblib

    if not BREAST_CANCER_MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Model file not found: {BREAST_CANCER_MODEL_FILE}"
        )
    if not BREAST_CANCER_CLASS_INDEX_FILE.is_file():
        raise FileNotFoundError(
            f"Class index file not found: {BREAST_CANCER_CLASS_INDEX_FILE}"
        )

    bundle = joblib.load(BREAST_CANCER_MODEL_FILE)
    with BREAST_CANCER_CLASS_INDEX_FILE.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)
    idx_to_class = {int(value): str(key) for key, value in class_to_idx.items()}
    return bundle, idx_to_class


def extract_breast_cancer_features(pil_image):
    """Extract the HOG/LBP/intensity features expected by the supplied model."""
    import cv2
    import numpy as np
    from skimage.feature import hog, local_binary_pattern

    gray = np.array(pil_image.convert("L"))
    gray = cv2.resize(gray, BREAST_CANCER_IMG_SIZE)
    gray_eq = cv2.equalizeHist(gray)

    hog_feat = hog(
        gray_eq,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    lbp = local_binary_pattern(gray_eq, P=24, R=3, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )

    gray_hist = cv2.calcHist([gray_eq], [0], None, [32], [0, 256]).flatten()
    gray_hist = gray_hist / (gray_hist.sum() + 1e-8)

    stats = np.array(
        [
            gray.mean(),
            gray.std(),
            np.median(gray),
            gray_eq.mean(),
            gray_eq.std(),
        ]
    )

    return np.concatenate([hog_feat, lbp_hist, gray_hist, stats]).astype(
        "float32"
    )


def predict_breast_cancer(bundle, idx_to_class, pil_image):
    """Classify an uploaded breast-ultrasound image."""
    features = extract_breast_cancer_features(pil_image).reshape(1, -1)
    features_scaled = bundle["scaler"].transform(features)
    probs = bundle["model"].predict_proba(features_scaled)[0]
    probabilities = {
        idx_to_class[index]: float(probs[index])
        for index in range(len(probs))
    }
    predicted_class = max(probabilities, key=probabilities.get)
    return predicted_class, probabilities


def make_breast_cancer_pdf(
    image_path,
    predicted_class,
    probabilities,
    patient_name,
    patient_age,
    notes,
):
    """Create the same downloadable PDF report as the standalone interface."""
    from fpdf import FPDF

    class BreastCancerReportPDF(FPDF):
        def header(self):
            self.set_fill_color(173, 20, 87)
            self.rect(0, 0, 210, 25, style="F")
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 16)
            self.set_xy(10, 7)
            self.cell(
                0,
                10,
                "Breast Ultrasound Classification Report",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            self.set_text_color(0, 0, 0)
            self.ln(15)

        def footer(self):
            self.set_y(-18)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.multi_cell(
                0,
                4,
                "Disclaimer: This report is generated by an AI/Machine Learning "
                "model for clinical decision support / research only. It is not "
                "a medical diagnosis. Consult a qualified radiologist or doctor.",
                align="C",
            )

    pdf = BreastCancerReportPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Scan Information", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(
        0,
        7,
        f"Patient Name : {patient_name or 'N/A'}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        7,
        f"Age          : {patient_age or 'N/A'}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        7,
        f"Report Date  : {datetime.now().strftime('%d-%m-%Y %H:%M')}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        8,
        "Uploaded Ultrasound Image",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    try:
        pdf.image(image_path, x=65, w=80)
    except Exception:
        pass
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Result", new_x="LMARGIN", new_y="NEXT")

    colors = {
        "benign": (0, 140, 60),
        "malignant": (200, 30, 30),
        "normal": (30, 100, 200),
    }
    red, green, blue = colors.get(
        predicted_class.lower(),
        (0, 0, 0),
    )
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(red, green, blue)
    pdf.cell(
        0,
        10,
        f"Predicted Class: {predicted_class.upper()}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(90, 8, "Class", border=1)
    pdf.cell(60, 8, "Probability", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for class_name, probability in sorted(
        probabilities.items(),
        key=lambda item: -item[1],
    ):
        pdf.cell(90, 8, class_name.capitalize(), border=1)
        pdf.cell(
            60,
            8,
            f"{probability * 100:.2f}%",
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
        )

    if notes:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Notes", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, notes)

    output = pdf.output()
    return bytes(output) if not isinstance(output, bytes) else output


def breast_cancer_diagnosis_page():
    """Integrated version of the supplied standalone breast-cancer interface."""
    import tempfile
    from PIL import Image

    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] { display: none !important; }

            .stApp,
            [data-testid="stAppViewContainer"] {
                background: #FCE4EC !important;
            }

            .block-container {
                max-width: 1120px;
                padding-top: 2.2rem;
                padding-bottom: 3rem;
            }

            .hero-box {
                background: #FCE4EC;
                border: 1px solid rgba(173, 20, 87, 0.22);
                border-radius: 22px;
                padding: 28px 32px;
                margin-bottom: 18px;
                box-shadow: 0 12px 30px rgba(173, 20, 87, 0.12);
            }

            .hero-box h1 {
                color: #5d1636;
                margin: 0 0 8px 0;
                font-size: 2.25rem;
            }

            .hero-box p {
                color: #765267;
                font-size: 1.05rem;
                margin: 0;
            }

            div[data-testid="stExpander"],
            div[data-testid="stFileUploader"] {
                border-color: rgba(173, 20, 87, 0.20) !important;
            }

            .result-card {
                background: rgba(255,255,255,0.78);
                border: 1px solid rgba(173,20,87,0.16);
                border-radius: 18px;
                padding: 18px 22px;
                margin-top: 18px;
            }

            .small-note {
                color: #77596a;
                font-size: 0.92rem;
            }

            button[kind="primary"] {
                background: #ad1457 !important;
                border-color: #ad1457 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Back to dashboard", key="bc_back_to_dashboard"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    st.markdown(
        """
        <div class="hero-box">
            <h1>🎀 Breast Cancer Diagnosis</h1>
            <p>Upload a breast ultrasound image for Benign / Malignant / Normal classification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Clinical decision support only. Do not use this model as a standalone "
        "diagnosis or as a substitute for a radiologist's or doctor's assessment."
    )

    try:
        model_bundle, idx_to_class = load_breast_cancer_model()
    except Exception as error:
        st.error("The Breast Cancer Diagnosis model could not be loaded.")
        st.code(str(error))
        return

    identity = logged_in_patient_identity()
    with st.expander(
        "Patient / scan details (linked to the logged-in account)"
    ):
        st.caption("Patient identity is automatically linked to the currently logged-in account.")
        patient_name = st.text_input(
            "Patient Name",
            value=str(identity["name"]),
            key="bc_patient_name",
            disabled=True,
        )
        patient_age = st.text_input(
            "Age",
            value=str(identity["age"]) if int(identity["age"]) > 0 else "",
            key="bc_patient_age",
            disabled=True,
        )
        notes = st.text_area(
            "Additional Notes",
            value="",
            key="bc_notes",
        )

    uploaded_file = st.file_uploader(
        "Upload a breast ultrasound image",
        type=["png", "jpg", "jpeg"],
        key="bc_ultrasound_uploader",
    )

    if uploaded_file is not None:
        try:
            current_bytes = uploaded_file.getvalue()
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("The uploaded file could not be read as an image.")
            return

        st.image(
            pil_image,
            caption="Uploaded Image",
            use_container_width=True,
        )

        if st.button(
            "🔍 Predict",
            type="primary",
            use_container_width=True,
            key="bc_predict",
        ):
            with st.spinner("Model is analyzing the image..."):
                predicted_class, probabilities = predict_breast_cancer(
                    model_bundle,
                    idx_to_class,
                    pil_image,
                )

            st.session_state["bc_predicted_class"] = predicted_class
            st.session_state["bc_probabilities"] = probabilities
            st.session_state["bc_image_bytes"] = current_bytes
            st.session_state["bc_image_name"] = uploaded_file.name

    if "bc_predicted_class" in st.session_state:
        predicted_class = st.session_state["bc_predicted_class"]
        probabilities = st.session_state["bc_probabilities"]

        color = {
            "benign": "green",
            "malignant": "red",
            "normal": "blue",
        }.get(predicted_class.lower(), "gray")

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### Result: :{color}[{predicted_class.upper()}]")

        top_confidence = max(probabilities.values())
        if top_confidence < 0.50:
            st.info(
                "ℹ️ Confidence is quite low. This image may not be a breast "
                "ultrasound scan, or the scan quality/angle may be unusual for "
                "the model. Interpret this result with caution."
            )

        st.subheader("Class Probabilities")
        st.bar_chart(probabilities)
        for class_name, probability in sorted(
            probabilities.items(),
            key=lambda item: -item[1],
        ):
            st.write(
                f"**{class_name.capitalize()}**: "
                f"{probability * 100:.2f}%"
            )
            st.progress(float(probability))

        try:
            image_bytes = st.session_state.get("bc_image_bytes")
            if image_bytes:
                with tempfile.TemporaryDirectory() as temporary_dir:
                    suffix = (
                        Path(
                            st.session_state.get(
                                "bc_image_name",
                                "scan.png",
                            )
                        ).suffix
                        or ".png"
                    )
                    image_path = (
                        Path(temporary_dir) / f"uploaded_image{suffix}"
                    )
                    image_path.write_bytes(image_bytes)

                    pdf_bytes = make_breast_cancer_pdf(
                        str(image_path),
                        predicted_class,
                        probabilities,
                        patient_name,
                        patient_age,
                        notes,
                    )

                safe_name = re.sub(
                    r"[^A-Za-z0-9_-]+",
                    "_",
                    patient_name.strip() or "patient",
                )
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=(
                        f"breast_ultrasound_report_{safe_name}.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                    key="bc_download_pdf",
                )
        except Exception as error:
            st.warning(f"PDF generation is unavailable: {error}")

        st.markdown("</div>", unsafe_allow_html=True)


def render_signup_form(widget_prefix: str, auto_login: bool = False) -> None:
    """Render the shared account-creation form.

    ``widget_prefix`` keeps Streamlit widget keys unique when the form is shown
    more than once in the same script run (public login page + admin panel).
    With ``auto_login`` the new account is signed in immediately.
    """
    with st.form(f"{widget_prefix}_form", clear_on_submit=False, border=True):
        signup_user_type = st.selectbox(
            "User type",
            USER_TYPE_OPTIONS,
            key=f"{widget_prefix}_user_type",
        )

        signup_other_user_type = ""
        if signup_user_type == "Others":
            signup_other_user_type = st.text_input(
                "Specify user type",
                placeholder="Enter your user type",
                key=f"{widget_prefix}_other_user_type",
            )

        full_name = st.text_input(
            "Full name",
            placeholder="Enter your full name",
            key=f"{widget_prefix}_full_name",
        )
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other", "Prefer not to say"],
            key=f"{widget_prefix}_gender",
        )
        email = st.text_input(
            "Email",
            placeholder="name@example.com",
            key=f"{widget_prefix}_email",
        )
        username = st.text_input(
            "Choose username",
            placeholder="Choose a username",
            key=f"{widget_prefix}_username",
        )
        new_password = st.text_input(
            "Choose password",
            type="password",
            placeholder="At least 6 characters",
            key=f"{widget_prefix}_password",
        )
        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            placeholder="Write the password again",
            key=f"{widget_prefix}_confirm_password",
        )
        signup_submitted = st.form_submit_button(
            "Create account",
            use_container_width=True,
        )

    if signup_submitted:
        username_clean = normalize_username(username)

        if not all([
            full_name.strip(),
            gender.strip(),
            email.strip(),
            username_clean,
            new_password,
            confirm_password,
        ]):
            st.warning("Please complete every field.")
        elif signup_user_type == "Others" and not signup_other_user_type.strip():
            st.warning("Please specify your user type.")
        elif not valid_email(email):
            st.warning("Enter a valid email address.")
        elif not re.fullmatch(r"[a-z0-9_.-]{3,30}", username_clean):
            st.warning(
                "Username must be 3–30 characters and use only letters, numbers, _, . or -."
            )
        elif len(new_password) < 6:
            st.warning("Password must contain at least 6 characters.")
        elif new_password != confirm_password:
            st.warning("The two passwords do not match.")
        else:
            try:
                user = create_user(
                    full_name,
                    email,
                    username_clean,
                    new_password,
                    signup_user_type,
                    gender,
                    signup_other_user_type,
                )
                if auto_login:
                    save_login_session(user)
                    update_last_login(user["user_id"])
                    st.session_state.signup_user_id = user["user_id"]
                    st.session_state.active_page = "dashboard"
                    st.rerun()
                else:
                    st.success(
                        f"Account created successfully. Your unique User ID is "
                        f"**{user['user_id']}**."
                    )
            except ValueError as error:
                st.warning(str(error))
            except Exception as error:
                st.error("The account could not be saved to Google Sheets.")
                with st.expander("Technical error"):
                    st.code(str(error))


def _render_service_assignment_tab() -> None:
    """Admin screen for verifying payment and assigning/revoking services."""
    st.subheader("Service Assignment (Payment Verification)")
    st.caption(
        "After a payment is verified, mark the service **Active** to unlock it "
        "for the patient. Use **Pending** while awaiting confirmation and "
        "**Revoke** to block access immediately."
    )

    try:
        users = get_all_users()
    except Exception:
        users = []

    if not users:
        st.info("No patient accounts are currently visible (Google Sheets unreachable).")
        st.stop()

    user_labels = {
        f"{clean_text(user.get('user_id'))} — {clean_text(user.get('full_name')) or clean_text(user.get('username'))}": user
        for user in users
    }
    selected_label = st.selectbox(
        "Patient",
        list(user_labels.keys()),
        key="ss_patient_select",
        format_func=lambda label: label,
    )
    selected_user = user_labels.get(selected_label, {})

    service_key_options = ["clinical_chatbot", "full_diagnostics", "brain_stroke", "heart_disease", "breast_cancer"]
    service_key = st.selectbox(
        "Service",
        service_key_options,
        key="ss_service_select",
        format_func=lambda key: SERVICE_KEY_TO_LABEL.get(key, key),
    )
    other_service = st.text_input(
        "Other service name (only if not listed above)",
        key="ss_other_service",
        placeholder="e.g. lab_package",
    )
    effective_service = (clean_text(other_service).lower() or service_key).strip()

    amount = st.text_input("Amount paid (e.g. 560)", key="ss_amount")
    payment_reference = st.text_input(
        "Payment reference (from receipt / payment gateway)",
        key="ss_payment_reference",
    )

    action_col, feedback_col = st.columns([2, 5])
    with action_col:
        status = None
        if st.button("✓ Mark Active", key="ss_status_active", use_container_width=True):
            status = "active"
        if st.button("◔ Mark Pending", key="ss_status_pending", use_container_width=True):
            status = "pending"
        if st.button("✕ Revoke Access", key="ss_status_revoked", use_container_width=True):
            status = "revoked"
    with feedback_col:
        if not status:
            st.caption("Select an action to save or update this service record.")
        elif not effective_service:
            st.error("Please type a non-empty service name.")
        else:
            patient_id = clean_text(selected_user.get("user_id"))
            patient_name = clean_text(selected_user.get("full_name")) or clean_text(selected_user.get("username"))
            saved = upsert_service_subscription(
                patient_id=patient_id,
                patient_name=patient_name,
                service=effective_service,
                status=status,
                amount=amount,
                payment_reference=payment_reference,
                verified_by=clean_text(st.session_state.get("current_user", {}).get("full_name")) or "admin",
            )
            if saved:
                st.success(
                    f"{SERVICE_KEY_TO_LABEL.get(effective_service, effective_service)} "
                    f"set to **{status.title()}** for {patient_name} ({patient_id})."
                )
            else:
                st.error("Could not save the service record. Check Google Sheets access.")

    st.divider()
    st.subheader("Current service records")
    records = list_service_subscriptions()
    if not records:
        st.caption("No payment/service records yet. Assign the first one above.")
        return
    display_records = [
        {
            "Patient ID": record["patient_id"],
            "Patient": record["patient_name"],
            "Service": SERVICE_KEY_TO_LABEL.get(str(record["service"]).lower(), record["service"]),
            "Status": record["status"].title(),
            "Amount": record["amount"],
            "Payment Ref": record["payment_reference"],
            "Verified By": record["verified_by"],
            "Updated At": record["updated_at"],
        }
        for record in records
    ]
    st.dataframe(
        pd.DataFrame(display_records),
        use_container_width=True,
        hide_index=True,
    )


def admin_page() -> None:
    """Show administrators a Sign up tab and a Service Assignment tab."""
    top_navigation("")

    signup_tab, services_tab = st.tabs(["Sign up", "Service Assignment"])

    with signup_tab:
        render_signup_form("admin_signup", auto_login=False)

    with services_tab:
        _render_service_assignment_tab()


def dashboard_page() -> None:
    top_navigation("")

    signup_user_id = st.session_state.pop("signup_user_id", "")
    if signup_user_id:
        st.success(
            f"Account created successfully. Your unique User ID is "
            f"**{signup_user_id}**. Save it for future patient searches."
        )

    st.markdown(
        """
        <div class="welcome-hero">
            <h1>Welcome to SMART CLINIC</h1>
            <p>Your account is connected. Select a block below to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_doctor = current_user_is_doctor()
    first, second, third = st.columns(3)

    with first:
        if is_doctor:
            tool_card(
                "🫀",
                "Heart Disease Diagnosis",
                "Enter cardiovascular health values for heart-disease risk analysis and support.",
                ready=True,
                icon_image=HEART_DISEASE_ICON_FILE,
            )
            if st.button(
                "Open Heart Disease Diagnosis",
                key="open_heart_diagnosis_doctor",
                use_container_width=True,
            ):
                st.session_state.active_page = "heart_disease_diagnosis"
                st.rerun()
        else:
            tool_card(
                "🫀",
                "Heart Disease Diagnosis",
                "Enter cardiovascular health values for heart-disease risk analysis and support.",
                ready=True,
                icon_image=HEART_DISEASE_ICON_FILE,
            )
            if st.button(
                "Open Heart Disease Diagnosis",
                key="open_heart_diagnosis_patient",
                use_container_width=True,
            ):
                st.session_state.active_page = "heart_disease_diagnosis"
                st.rerun()

    with second:
        if is_doctor:
            tool_card(
                "🫀",
                "Brain Stroke Diagnosis",
                "Analyze brain-stroke related clinical information and medical imaging findings.",
                ready=True,
                icon_image=BRAIN_STROKE_ICON_FILE,
            )
            if st.button("Open Brain Stroke Diagnosis", key="open_stroke_doctor", use_container_width=True):
                st.session_state.active_page = "brain_stroke_diagnosis"
                st.rerun()
        else:
            tool_card(
                "🫀",
                "Brain Stroke Diagnosis",
                "Analyze brain-stroke related clinical information and medical imaging findings.",
                ready=True,
                icon_image=BRAIN_STROKE_ICON_FILE,
            )
            if st.button("Open Brain Stroke Diagnosis", key="open_stroke_patient", use_container_width=True):
                st.session_state.active_page = "brain_stroke_diagnosis"
                st.rerun()

    with third:
        if is_doctor:
            tool_card(
                "💬",
                "Clinical Chatbot",
                "Ask medical questions, book doctor appointments, and schedule lab tests through the Clinical Chatbot.",
                ready=True,
            )
            if st.button("Open Clinical Chatbot", key="open_chatbot_doctor", use_container_width=True):
                st.session_state.active_page = "clinical_chatbot"
                st.rerun()
        else:
            tool_card(
                "💬",
                "Clinical Chatbot",
                "Ask medical questions, book doctor appointments, and schedule lab tests through the Clinical Chatbot.",
                ready=True,
            )
            if st.button("Open Clinical Chatbot", key="open_chatbot_patient", use_container_width=True):
                st.session_state.active_page = "clinical_chatbot"
                st.rerun()

    def render_breast_cancer_card() -> None:
        tool_card(
            "🎀",
            "Breast Cancer Diagnosis",
            "Analyze breast-cancer related clinical information and medical imaging findings.",
            ready=True,
            icon_image=BREAST_CANCER_ICON_FILE,
        )
        button_key = "open_breast_cancer_doctor" if is_doctor else "open_breast_cancer_patient"
        if st.button("Open Breast Cancer Diagnosis", key=button_key, use_container_width=True):
            st.session_state.active_page = "breast_cancer_diagnosis"
            st.rerun()

    def render_user_dashboard_card() -> None:
        tool_card(
            "👤",
            "User Dashboard",
            "View and edit your personal information, contact details, and profile photo.",
            ready=True,
        )
        if st.button("Open User Dashboard", key="open_user_dashboard", use_container_width=True):
            st.session_state.active_page = "user_dashboard"
            st.rerun()

    def render_search_patient_card() -> None:
        tool_card(
            "🔎",
            "Search Patient",
            "Search patient with ID, disease or symptoms.",
            ready=True,
        )
        if st.button("Open Search Patient", key="open_patient_search", use_container_width=True):
            st.session_state.active_page = "patient_search"
            st.rerun()

    def render_agentic_diagnostic_card() -> None:
        tool_card(
            "🧠",
            "Full Diagnostics Scan",
            "Multi-agent diagnostic assistant with medication, safety, and readmission analysis.",
            ready=True,
        )
        # On hosted builds the embedded diagnostic Station is replaced by the
        # in-process Full Diagnostics page; the desktop launcher keeps the
        # original bundled sub-application.
        route = "general_diagnostics" if CLOUD_DEPLOY else "agentic_diagnostic"
        if st.button("Open Full Diagnostics Scan", key="open_agentic_diagnostic", use_container_width=True):
            st.session_state.active_page = route
            st.rerun()

    # Full Diagnostics Scan is available to both Patient and Doctor accounts.
    # Search Patient remains doctor-only.
    remaining_cards = [
        render_breast_cancer_card,
        render_user_dashboard_card,
        render_agentic_diagnostic_card,
    ]
    if is_doctor:
        remaining_cards.append(render_search_patient_card)

    for row_start in range(0, len(remaining_cards), 3):
        row_cards = remaining_cards[row_start : row_start + 3]
        row_columns = st.columns(3)
        for render_card, column in zip(row_cards, row_columns):
            with column:
                render_card()


def user_dashboard_page() -> None:
    top_navigation("User Dashboard")

    # Page-specific pastel lavender palette requested for the User Dashboard.
    st.markdown(
        """
        <style>
            .stApp,
            [data-testid="stAppViewContainer"] {
                background: #EDE5FF !important;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"],
            div[data-testid="stForm"],
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-color: rgba(180, 117, 61, 0.24) !important;
            }

            div[data-testid="stForm"],
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.58) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Back to Smart Clinic", key="back_from_user_dashboard"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    user = st.session_state.current_user
    user_id = clean_text(user.get("user_id"))
    profile_tab, history_tab = st.tabs(["👤 Profile", "🩺 Medical History"])

    with profile_tab:
        photo_column, details_column = st.columns([1, 2.4], gap="large")

        with photo_column:
            st.subheader("Profile photo")
            current_photo = profile_photo_path(user)
            if current_photo.is_file():
                st.image(str(current_photo), use_container_width=True)
            st.caption(
                "A default photo is shown according to gender until a profile photo is uploaded."
            )

        with details_column:
            st.subheader("Personal information")
            st.info(f"Unique User ID: **{user_id}** (cannot be edited)")

            gender_options = ["Male", "Female", "Other", "Prefer not to say"]
            current_gender = clean_text(user.get("gender"))
            gender_index = (
                gender_options.index(current_gender)
                if current_gender in gender_options
                else 0
            )

            saved_dob = clean_text(user.get("date_of_birth"))
            try:
                initial_dob = datetime.strptime(saved_dob, "%Y-%m-%d").date()
            except ValueError:
                initial_dob = datetime(2000, 1, 1).date()

            with st.form("user_profile_form", border=True):
                full_name = st.text_input(
                    "Full name",
                    value=clean_text(user.get("full_name")),
                )
                email = st.text_input(
                    "Email",
                    value=clean_text(user.get("email")),
                )

                left_field, right_field = st.columns(2)
                with left_field:
                    age = st.number_input(
                        "Age",
                        min_value=0,
                        max_value=130,
                        value=int(user.get("age")) if clean_text(user.get("age")).isdigit() else 0,
                        step=1,
                    )
                    date_of_birth = st.date_input(
                        "Date of birth",
                        value=initial_dob,
                        min_value=datetime(1900, 1, 1).date(),
                        max_value=datetime.now().date(),
                    )
                    contact_info = st.text_input(
                        "Contact information",
                        value=clean_text(user.get("contact_info")),
                        placeholder="Phone number or other contact details",
                    )

                with right_field:
                    gender = st.selectbox(
                        "Gender",
                        gender_options,
                        index=gender_index,
                    )
                    cnic_id = st.text_input(
                        "CNIC ID",
                        value=clean_text(user.get("cnic_id")),
                        placeholder="Example: 12345-1234567-1",
                    )
                    uploaded_photo = st.file_uploader(
                        "Upload profile photo",
                        type=["jpg", "jpeg", "png", "webp"],
                        help="JPG, PNG, or WEBP; maximum size 5 MB.",
                    )

                save_profile = st.form_submit_button(
                    "Save profile",
                    use_container_width=True,
                )

            if save_profile:
                if not full_name.strip():
                    st.warning("Full name is required.")
                elif not valid_email(email):
                    st.warning("Enter a valid email address.")
                elif cnic_id.strip() and not re.fullmatch(r"\d{5}-\d{7}-\d", cnic_id.strip()):
                    st.warning("CNIC ID must use the format 12345-1234567-1.")
                else:
                    try:
                        saved_photo = clean_text(user.get("profile_photo"))
                        if uploaded_photo is not None:
                            saved_photo = save_profile_photo(user_id, uploaded_photo)

                        updated_user = update_user_profile(
                            user_id,
                            {
                                "full_name": full_name,
                                "email": normalize_email(email),
                                "age": str(age),
                                "gender": gender,
                                "date_of_birth": date_of_birth.isoformat(),
                                "contact_info": contact_info,
                                "cnic_id": cnic_id,
                                "profile_photo": saved_photo,
                            },
                        )
                        save_login_session(updated_user)
                        st.session_state.active_page = "user_dashboard"
                        st.success("Your profile has been updated.")
                        st.rerun()
                    except ValueError as error:
                        st.warning(str(error))
                    except Exception as error:
                        st.error("The profile could not be saved.")
                        with st.expander("Technical error"):
                            st.code(str(error))

    with history_tab:
        medical_history_tab(user_id)


def logged_in_patient_identity() -> dict[str, object]:
    """Return normalized identity fields for the currently authenticated user."""
    user = st.session_state.get("current_user", {}) or {}
    name = clean_text(user.get("full_name")) or clean_text(user.get("username")) or "Logged-in user"
    user_id = clean_text(user.get("user_id"))
    raw_age = clean_text(user.get("age"))
    try:
        age = max(0, min(150, int(float(raw_age)))) if raw_age else 0
    except (TypeError, ValueError):
        age = 0

    # Older accounts may have a date of birth saved before the separate age
    # field was populated. Use it only as an account-derived fallback.
    if age == 0:
        raw_dob = clean_text(user.get("date_of_birth"))
        if raw_dob:
            for dob_format in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    dob = datetime.strptime(raw_dob, dob_format).date()
                    today = datetime.now().date()
                    age = today.year - dob.year - (
                        (today.month, today.day) < (dob.month, dob.day)
                    )
                    age = max(0, min(150, age))
                    break
                except ValueError:
                    continue

    gender = clean_text(user.get("gender"))
    if gender not in {"Male", "Female", "Other", "Prefer not to say"}:
        gender = "Other"
    return {"user_id": user_id, "name": name, "age": age, "gender": gender}


# ---------------------------------------------------------
# Smart e-Health General Diagnostics integration
# ---------------------------------------------------------
def _truncate_medical_text(text: str, limit: int = 40000) -> str:
    """Keep uploaded content within a practical size for analysis."""
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "\n\n[Content shortened because the uploaded file was very large.]"


def extract_uploaded_medical_text(uploaded_file) -> tuple[str, str]:
    """Extract readable text or metadata from a supported uploaded file."""
    from io import BytesIO

    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    file_type = suffix.lstrip(".").upper() or "UNKNOWN"

    if suffix == ".pdf":
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return _truncate_medical_text(text), "PDF document"

    if suffix == ".docx":
        from docx import Document

        document = Document(BytesIO(file_bytes))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return _truncate_medical_text("\n".join(parts)), "Word document"

    if suffix in {".txt", ".md", ".rtf"}:
        return _truncate_medical_text(file_bytes.decode("utf-8", errors="replace")), "Text document"

    if suffix == ".csv":
        dataframe = pd.read_csv(BytesIO(file_bytes))
        return _truncate_medical_text(dataframe.to_string(index=False)), "CSV laboratory data"

    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(BytesIO(file_bytes))
        sheets = []
        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(workbook, sheet_name=sheet_name)
            sheets.append(f"SHEET: {sheet_name}\n{dataframe.to_string(index=False)}")
        return _truncate_medical_text("\n\n".join(sheets)), "Excel laboratory data"

    if suffix in {".dcm", ".dicom"}:
        import pydicom

        dataset = pydicom.dcmread(BytesIO(file_bytes), force=True)
        metadata_fields = [
            "Modality",
            "StudyDescription",
            "SeriesDescription",
            "BodyPartExamined",
            "PatientPosition",
            "Rows",
            "Columns",
        ]
        metadata = []
        for field in metadata_fields:
            if hasattr(dataset, field):
                metadata.append(f"{field}: {getattr(dataset, field)}")
        metadata.append(
            "The DICOM pixels are not diagnosed automatically by this text-based agent. "
            "Add the radiologist's findings in the Image Findings box."
        )
        return "\n".join(metadata), "DICOM medical image metadata"

    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}:
        from PIL import Image

        image = Image.open(BytesIO(file_bytes))
        base_description = (
            f"Image file: {uploaded_file.name}; size: {image.width} x {image.height}; "
            f"format: {image.format or file_type}."
        )

        try:
            import pytesseract

            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                return _truncate_medical_text(
                    f"{base_description}\nOCR text extracted from image:\n{ocr_text}"
                ), "Medical image with OCR text"
        except Exception:
            pass

        return (
            base_description
            + " No reliable text was extracted. Add the clinical or radiology findings manually."
        ), "Medical image"

    raise ValueError(f"Unsupported file type: {suffix or uploaded_file.name}")


@st.cache_resource(show_spinner=False)
def load_general_diagnostic_agent():
    """Load the Smart e-Health diagnostic agent once."""
    diagnostic_project = configure_diagnostic_imports()

    try:
        from dotenv import load_dotenv

        # Prefer the diagnostic project's API configuration, while still
        # allowing the main dashboard to contain its own .env file.
        load_dotenv(diagnostic_project / ".env")
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass

    from importlib import import_module

    DiagnosticAgent = import_module(
        "agents.diagnostic_agent"
    ).DiagnosticAgent

    return DiagnosticAgent()


@st.cache_resource(show_spinner=False)
def load_embedded_diagnostics_dashboard():
    """Import the bundled Full Diagnostics sidebar app for in-process rendering.

    The desktop build launches general_diagnostics\\app.py as a second Streamlit
    server (port 8600) and embeds it in an iframe. A hosted Streamlit Cloud build
    can only run a single app, so the very same dashboard is rendered inside this
    process instead. Importing the module is safe because its st.set_page_config
    call only fires when it is executed as its own entry point.
    """
    configure_diagnostic_imports()

    from importlib import import_module

    return import_module("app")


def _general_diagnostics_pdf(report_content: str, patient_info: dict[str, object]) -> bytes:
    """Generate a PDF while supporting different fpdf2 return types."""
    configure_diagnostic_imports()
    from importlib import import_module

    PDFReportGenerator = import_module(
        "utils.pdf_generator"
    ).PDFReportGenerator

    output = PDFReportGenerator().generate_pdf_report(report_content, patient_info)
    if isinstance(output, bytes):
        return output
    if isinstance(output, bytearray):
        return bytes(output)
    if isinstance(output, str):
        return output.encode("latin-1", errors="ignore")
    return bytes(output)


def general_diagnostics_page() -> None:
    top_navigation("Full Diagnostics Scan")

    identity = logged_in_patient_identity()
    allowed, message, subscription = service_access_allowed(
        str(identity["user_id"]), "full_diagnostics"
    )
    if not allowed:
        st.warning(message)
        if subscription.get("payment_reference"):
            st.caption(
                "Payment reference on record: "
                + clean_text(subscription.get("payment_reference"))
            )
        if st.button("← Back to dashboard", key="gd_blocked_back"):
            st.session_state.active_page = "dashboard"
            st.rerun()
        st.stop()

    if st.button("← Back to dashboard", key="gd_back"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    # On the hosted single-app build, render the very same Full Diagnostics
    # sidebar dashboard that the desktop build shows on port 8600. It is
    # imported and drawn inside this process because Streamlit Cloud can only
    # run one app. If anything about the embedded app is unavailable, we fall
    # back to the simplified in-process form below.
    if CLOUD_DEPLOY:
        try:
            embedded_dashboard = load_embedded_diagnostics_dashboard()
        except Exception as error:
            st.warning(
                "The full diagnostics dashboard could not be loaded, so the "
                f"simplified form is shown instead. ({error})"
            )
        else:
            embedded_dashboard.main()
            return

    st.markdown(
        """
        <div class="hero-box">
            <h1>🩺 Smart e-Health Full Diagnostics Scan</h1>
            <p>Enter the clinical case, notes, laboratory data and imaging findings to create a structured diagnostic-support report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Educational and research use only. This tool can make mistakes and must not replace a qualified doctor, "
        "emergency service, or formal medical diagnosis."
    )
    st.caption(
        "When an OpenAI or Groq API key is configured in the .env file, entered information may be sent to that "
        "provider for analysis. Without an API key, the supplied model uses its local keyword-based fallback."
    )

    try:
        diagnostic_agent = load_general_diagnostic_agent()
        if diagnostic_agent.llm_available:
            provider_name = (diagnostic_agent.model_type or "online AI").title()
            st.success(f"Smart e-Health agent ready — {provider_name} mode.")
        else:
            st.info("Smart e-Health agent ready — local analysis mode. Add an API key for the full AI report.")
    except Exception as error:
        st.error("The Smart e-Health diagnostic agent could not be loaded.")
        st.code(str(error))
        st.stop()

    identity = logged_in_patient_identity()
    patient_column, clinical_column = st.columns([1, 2])

    with patient_column:
        st.subheader("👤 Patient information")
        st.caption("These identity fields are linked to the currently logged-in account.")
        patient_id = st.text_input(
            "Patient ID",
            value=str(identity["user_id"]),
            key="gd_patient_id",
            disabled=True,
        )
        patient_name = st.text_input(
            "Patient name",
            value=str(identity["name"]),
            key="gd_patient_name",
            disabled=True,
        )
        patient_age = st.number_input(
            "Age",
            min_value=0,
            max_value=150,
            value=int(identity["age"]),
            step=1,
            key="gd_patient_age",
            disabled=True,
        )
        gender_options = ["Male", "Female", "Other", "Not specified"]
        identity_gender = str(identity["gender"])
        if identity_gender == "Prefer not to say":
            identity_gender = "Not specified"
        patient_gender = st.selectbox(
            "Gender",
            gender_options,
            index=gender_options.index(identity_gender) if identity_gender in gender_options else 2,
            key="gd_patient_gender",
            disabled=True,
        )

    with clinical_column:
        st.subheader("📄 Clinical input")
        clinical_case = st.text_area(
            "Clinical case description",
            key="gd_clinical_case",
            height=160,
            placeholder="Describe the symptoms, duration, medical history and chief complaint.",
        )
        clinical_notes = st.text_area(
            "Clinical notes",
            key="gd_clinical_notes",
            height=160,
            placeholder="Add physician observations, nursing notes, medicines or other relevant details.",
        )

    st.subheader("📁 Clinical documents and laboratory data")
    document_column, laboratory_column = st.columns(2)

    with document_column:
        uploaded_documents = st.file_uploader(
            "Upload clinical documents",
            type=["pdf", "docx", "txt", "md", "rtf", "jpg", "jpeg", "png", "tiff", "webp"],
            accept_multiple_files=True,
            key="gd_document_upload",
            help="Text is extracted from supported documents. Image OCR is used only when available.",
        )
        manual_documents = st.text_area(
            "Paste report or document text",
            key="gd_manual_documents",
            height=180,
            placeholder="Paste discharge summaries, prescriptions, reports or other supporting text.",
        )

    with laboratory_column:
        uploaded_lab_file = st.file_uploader(
            "Upload laboratory results",
            type=["csv", "xlsx", "xls"],
            key="gd_lab_upload",
        )
        manual_lab_data = st.text_area(
            "Enter lab values and vital signs",
            key="gd_manual_lab_data",
            height=180,
            placeholder="Example: Glucose 240 mg/dL, Hemoglobin 10.2 g/dL, BP 150/95 mmHg.",
        )

    st.subheader("🖼️ Medical image information")
    uploaded_images = st.file_uploader(
        "Upload medical images or DICOM files",
        type=["dcm", "dicom", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        key="gd_image_upload",
    )

    previewable_images = [
        image
        for image in (uploaded_images or [])
        if Path(image.name).suffix.lower() not in {".dcm", ".dicom"}
    ]
    if previewable_images:
        preview_columns = st.columns(min(3, len(previewable_images)))
        for index, image in enumerate(previewable_images):
            with preview_columns[index % len(preview_columns)]:
                st.image(image, caption=image.name, use_container_width=True)

    image_findings = st.text_area(
        "Image or radiology findings",
        key="gd_image_findings",
        height=140,
        placeholder="Enter the radiologist's findings or describe relevant observations. The agent does not reliably diagnose image pixels by itself.",
    )

    analyze_column, clear_column = st.columns(2)

    with analyze_column:
        analyze_clicked = st.button(
            "🩺 Analyze clinical case",
            key="gd_analyze",
            type="primary",
            use_container_width=True,
        )

    with clear_column:
        if st.button("🗑️ Clear General Diagnostics", key="gd_clear", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("gd_"):
                    del st.session_state[key]
            st.rerun()

    if analyze_clicked:
        has_input = any(
            [
                clinical_case.strip(),
                clinical_notes.strip(),
                manual_documents.strip(),
                manual_lab_data.strip(),
                image_findings.strip(),
                uploaded_documents,
                uploaded_lab_file,
                uploaded_images,
            ]
        )

        if not has_input:
            st.warning("Enter some clinical information or upload a supported file before analysis.")
        else:
            document_parts = [manual_documents.strip()] if manual_documents.strip() else []
            lab_parts = [manual_lab_data.strip()] if manual_lab_data.strip() else []
            image_parts = [image_findings.strip()] if image_findings.strip() else []
            processed_types = []
            extraction_errors = []

            with st.spinner("Reading files and analyzing the clinical case..."):
                for uploaded_file in uploaded_documents or []:
                    try:
                        extracted_text, document_type = extract_uploaded_medical_text(uploaded_file)
                        document_parts.append(f"--- {uploaded_file.name} ---\n{extracted_text}")
                        processed_types.append(f"{uploaded_file.name}: {document_type}")
                    except Exception as error:
                        extraction_errors.append(f"{uploaded_file.name}: {error}")

                if uploaded_lab_file is not None:
                    try:
                        extracted_text, document_type = extract_uploaded_medical_text(uploaded_lab_file)
                        lab_parts.append(f"--- {uploaded_lab_file.name} ---\n{extracted_text}")
                        processed_types.append(f"{uploaded_lab_file.name}: {document_type}")
                    except Exception as error:
                        extraction_errors.append(f"{uploaded_lab_file.name}: {error}")

                for uploaded_file in uploaded_images or []:
                    try:
                        extracted_text, document_type = extract_uploaded_medical_text(uploaded_file)
                        image_parts.append(f"--- {uploaded_file.name} ---\n{extracted_text}")
                        processed_types.append(f"{uploaded_file.name}: {document_type}")
                    except Exception as error:
                        extraction_errors.append(f"{uploaded_file.name}: {error}")

                patient_context = (
                    f"Patient ID: {patient_id or 'Not provided'}\n"
                    f"Patient name: {patient_name or 'Not provided'}\n"
                    f"Age: {patient_age}\n"
                    f"Gender: {patient_gender}\n\n"
                    f"{clinical_case.strip()}"
                )

                results = diagnostic_agent.analyze_case(
                    clinical_case=patient_context,
                    clinical_notes=clinical_notes.strip(),
                    lab_data=_truncate_medical_text("\n\n".join(lab_parts), 60000),
                    image_findings=_truncate_medical_text("\n\n".join(image_parts), 60000),
                    documents=_truncate_medical_text("\n\n".join(document_parts), 60000),
                    document_types="\n".join(processed_types),
                )

            st.session_state.gd_analysis_results = results
            st.session_state.gd_extraction_errors = extraction_errors
            st.session_state.gd_report_patient_info = {
                "patient_id": patient_id or "Not provided",
                "patient_name": patient_name or "Not provided",
                "patient_age": patient_age,
                "patient_gender": patient_gender,
            }

    results = st.session_state.get("gd_analysis_results")
    if results:
        st.divider()
        st.subheader("📈 Diagnostic-support results")

        for extraction_error in st.session_state.get("gd_extraction_errors", []):
            st.warning(f"A file could not be fully read: {extraction_error}")

        if results.get("error"):
            st.error(results["error"])
        else:
            summary_tab, analysis_tab, report_tab = st.tabs(
                ["Summary", "Detailed analysis", "Formatted report"]
            )

            with summary_tab:
                st.info(results.get("summary", "No summary was generated."))

            with analysis_tab:
                st.markdown(results.get("diagnostic_analysis", "No detailed analysis was generated."))

            with report_tab:
                report_content = results.get("formatted_report", "No report was generated.")
                st.markdown(report_content)

            report_content = results.get("formatted_report", "No report was generated.")
            patient_info = st.session_state.get("gd_report_patient_info", {})
            report_patient_id = clean_text(patient_info.get("patient_id"))
            if report_patient_id and report_patient_id != "Not provided":
                try:
                    save_latest_diagnostic_report(
                        report_patient_id,
                        report_content,
                        patient_info,
                    )
                except OSError as error:
                    st.warning(f"The report could not be saved for later access: {error}")
            safe_patient_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(patient_info.get("patient_id", "unknown")))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            download_pdf_column, download_text_column = st.columns(2)
            with download_pdf_column:
                try:
                    pdf_bytes = _general_diagnostics_pdf(report_content, patient_info)
                    st.download_button(
                        "📥 Download report as PDF",
                        data=pdf_bytes,
                        file_name=f"diagnostic_report_{safe_patient_id}_{timestamp}.pdf",
                        mime="application/pdf",
                        key="gd_pdf_download",
                        use_container_width=True,
                    )
                except Exception as error:
                    st.warning(f"PDF generation is unavailable: {error}")

            with download_text_column:
                st.download_button(
                    "📄 Download report as text",
                    data=report_content,
                    file_name=f"diagnostic_report_{safe_patient_id}_{timestamp}.txt",
                    mime="text/plain",
                    key="gd_text_download",
                    use_container_width=True,
                )



def heart_disease_diagnosis_page() -> None:
    """Render the supplied HDD model inside the SMART CDSS interface."""
    top_navigation("Heart Disease Diagnosis")

    # Page-specific pastel pink palette requested for Heart Disease Diagnosis.
    st.markdown(
        """
        <style>
            .stApp,
            [data-testid="stAppViewContainer"] {
                background: #EDBDD5 !important;
            }

            .hero-box {
                background: #EDBDD5 !important;
                border-color: rgba(130, 65, 103, 0.22) !important;
                box-shadow: 0 12px 30px rgba(130, 65, 103, 0.12) !important;
            }

            div[data-testid="stForm"] {
                border-color: rgba(130, 65, 103, 0.20) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Back to Smart Clinic", key="back_from_heart_diagnosis"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    st.markdown(
        """
        <div class="hero-box">
            <h1>🫀 Heart Disease Diagnosis</h1>
            <p>Enter the patient's cardiovascular measurements for Random Forest risk analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Clinical decision support only. Do not use this model as a standalone "
        "diagnosis or as a substitute for emergency or specialist care."
    )

    try:
        model, test_accuracy = load_heart_disease_model()
    except Exception as error:
        st.error("The supplied Heart Disease Diagnosis model could not be loaded.")
        st.code(str(error))
        return

    st.caption(f"Supplied Random Forest model loaded • validation accuracy: {test_accuracy:.1%}")
    identity = logged_in_patient_identity()
    st.info(
        f"Assessment for **{identity['name']}**"
        + (f" • User ID: **{identity['user_id']}**" if identity["user_id"] else "")
    )
    heart_default_age = int(identity["age"]) if int(identity["age"]) > 0 else 50
    heart_default_sex = 1 if identity["gender"] == "Male" else 0

    with st.form("heart_disease_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(
                "Age (years)", min_value=1, max_value=120,
                value=max(1, min(120, heart_default_age)), disabled=True
            )
            sex_options = [("Female", 0), ("Male", 1)]
            sex = st.selectbox(
                "Sex", sex_options, index=heart_default_sex,
                format_func=lambda x: x[0], disabled=True
            )
            cp = st.selectbox(
                "Chest pain type",
                [("Typical angina", 0), ("Atypical angina", 1),
                 ("Non-anginal pain", 2), ("Asymptomatic", 3)],
                format_func=lambda x: x[0],
            )
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)", min_value=60, max_value=260, value=120
            )
            chol = st.number_input(
                "Serum cholesterol (mg/dL)", min_value=80, max_value=700, value=200
            )
        with c2:
            fbs = st.selectbox(
                "Fasting blood sugar > 120 mg/dL",
                [("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0],
            )
            restecg = st.selectbox(
                "Resting ECG",
                [("Normal", 0), ("ST-T abnormality", 1),
                 ("Left ventricular hypertrophy", 2)],
                format_func=lambda x: x[0],
            )
            thalachh = st.number_input(
                "Maximum heart rate achieved", min_value=40, max_value=250, value=150
            )
            exang = st.selectbox(
                "Exercise-induced angina",
                [("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0],
            )
        with c3:
            oldpeak = st.number_input(
                "ST depression (oldpeak)", min_value=0.0, max_value=10.0,
                value=0.0, step=0.1
            )
            slope = st.selectbox(
                "Peak exercise ST slope",
                [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                format_func=lambda x: x[0],
            )
            ca = st.selectbox("Major vessels colored (0–4)", [0, 1, 2, 3, 4])
            thal = st.selectbox(
                "Thalassemia result",
                [("Unknown/normal code 0", 0), ("Normal", 1),
                 ("Fixed defect", 2), ("Reversible defect", 3)],
                format_func=lambda x: x[0],
            )

        submitted = st.form_submit_button(
            "Analyze Heart Disease Risk",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    patient_data = {
        "age": float(age),
        "sex": float(sex[1]),
        "cp": float(cp[1]),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": float(fbs[1]),
        "restecg": float(restecg[1]),
        "thalachh": float(thalachh),
        "exang": float(exang[1]),
        "oldpeak": float(oldpeak),
        "slope": float(slope[1]),
        "ca": float(ca),
        "thal": float(thal[1]),
    }
    input_frame = pd.DataFrame(
        [[patient_data[name] for name in HEART_DISEASE_FEATURES]],
        columns=HEART_DISEASE_FEATURES,
    )
    prediction = int(model.predict(input_frame)[0])
    probability = float(model.predict_proba(input_frame)[0][1]) * 100

    if probability > 70:
        risk_level = "High Risk"
    elif probability > 40:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    st.subheader("Analysis result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model prediction", "Heart Disease Detected" if prediction else "No Heart Disease Detected")
    m2.metric("Estimated probability", f"{probability:.1f}%")
    m3.metric("Risk category", risk_level)

    summary = generate_heart_clinical_summary(patient_data, probability)
    if prediction:
        st.warning(summary)
    else:
        st.success(summary)


def _agentic_diagnostic_health(port: int) -> bool:
    """Return True only if the Streamlit server on this port is alive."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/_stcore/health", timeout=1.5
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _ensure_agentic_diagnostic_server() -> tuple[bool, str, int]:
    """Start/reuse the bundled Agentic Diagnostic Streamlit app."""
    preferred = AGENTIC_DIAGNOSTIC_PORT

    cached_port = st.session_state.get("_agentic_diagnostic_port")
    if isinstance(cached_port, int) and _agentic_diagnostic_health(cached_port):
        return True, "", cached_port
    st.session_state.pop("_agentic_diagnostic_port", None)

    # Do NOT reuse arbitrary healthy Streamlit servers discovered on these
    # ports. A previous/older Agentic Diagnostic process can remain alive after
    # app.py is edited; embedding it makes newly added UI blocks appear to be
    # "missing" even though they exist in the current source. Only a port that
    # this Streamlit session previously started/verified (cached above) may be
    # reused. Otherwise start the current project on a genuinely free port.
    port = next(
        (candidate for candidate in range(preferred, preferred + 12)
         if _port_is_free(candidate)),
        None,
    )
    if port is None:
        return False, "No free local port was available for Agentic Diagnostic.", preferred

    agentic_app = AGENTIC_DIAGNOSTIC_DIR / "app.py"
    if not agentic_app.is_file():
        return False, f"Agentic Diagnostic files were not found at: {AGENTIC_DIAGNOSTIC_DIR}", port

    log_path = AGENTIC_DIAGNOSTIC_DIR / "agentic_diagnostic_server.log"
    try:
        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        log_handle = open(log_path, "a", encoding="utf-8")
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.port",
                str(port),
                "--server.address",
                "127.0.0.1",
                "--server.headless",
                "true",
                "--server.enableXsrfProtection",
                "false",
                "--server.enableCORS",
                "false",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=str(AGENTIC_DIAGNOSTIC_DIR),
            stdout=log_handle,
            stderr=log_handle,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )

        for _ in range(40):
            if _agentic_diagnostic_health(port):
                st.session_state["_agentic_diagnostic_port"] = port
                return True, "", port
            time.sleep(0.5)

        return False, (
            "The Agentic Diagnostic service did not become healthy. "
            f"See {log_path.name} in the agentic_diagnostic folder."
        ), port
    except Exception as error:
        return False, str(error), port


def agentic_diagnostic_page() -> None:
    """Render the Agentic Diagnostic app inside SMART CDSS."""
    top_navigation("Full Diagnostics Scan")

    identity = logged_in_patient_identity()
    allowed, message, subscription = service_access_allowed(
        str(identity["user_id"]), "full_diagnostics"
    )
    if not allowed:
        st.warning(message)
        if subscription.get("payment_reference"):
            st.caption(
                "Payment reference on record: "
                + clean_text(subscription.get("payment_reference"))
            )
        if st.button("← Back to dashboard", key="agentic_blocked_back"):
            st.session_state.active_page = "dashboard"
            st.rerun()
        st.stop()

    if st.button("← Back to dashboard", key="agentic_back_dashboard"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    if CLOUD_DEPLOY:
        st.info(
            "The Full Diagnostics / Agentic Scan runs as a bundled local Streamlit "
            "service in the desktop version (RUN_SMART_CDSS.bat) and is not "
            "available in this hosted cloud build."
        )
        return

    running, error_message, agentic_port = _ensure_agentic_diagnostic_server()
    if not running:
        st.error("The Agentic Diagnostic interface could not be loaded.")
        with st.expander("Technical error"):
            st.code(error_message)
        return

    import streamlit.components.v1 as components

    # Include a source-version token in the iframe URL so the browser does not
    # keep displaying a cached frontend after the embedded app source changes.
    try:
        agentic_source = AGENTIC_DIAGNOSTIC_DIR / "app.py"
        source_version = str(agentic_source.stat().st_mtime_ns)
    except OSError:
        source_version = str(int(time.time()))

    components.iframe(
        f"http://127.0.0.1:{agentic_port}/?source_version={source_version}",
        height=1100,
        scrolling=True,
    )
def _port_is_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def clinical_chatbot_page() -> None:
    """Render the bundled HTML chatbot through Streamlit's in-process bridge."""
    identity = logged_in_patient_identity()
    allowed, message, subscription = service_access_allowed(
        str(identity["user_id"]), "clinical_chatbot"
    )
    if not allowed:
        top_navigation("Clinical Chatbot")
        st.warning(message)
        if subscription.get("payment_reference"):
            st.caption(
                "Payment reference on record: "
                + clean_text(subscription.get("payment_reference"))
            )
        if st.button("← Back to dashboard", key="cb_blocked_back"):
            st.session_state.active_page = "dashboard"
            st.rerun()
        st.stop()

    from clinical_chatbot_streamlit import render_clinical_chatbot_page
    render_clinical_chatbot_page()


def brain_stroke_diagnosis_page() -> None:
    top_navigation("Brain Stroke Diagnosis")

    if st.button("← Back to dashboard", key="stroke_back_dashboard"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    try:
        from stroke_agent.interface import render_stroke_agent
        render_stroke_agent()
    except Exception as error:
        st.error("The Brain Stroke Diagnosis interface could not be loaded.")
        st.code(str(error))


def patient_search_page() -> None:
    if not current_user_is_doctor():
        st.session_state.active_page = "dashboard"
        st.error("Patient search is available only to doctor accounts.")
        if st.button("Return to dashboard", use_container_width=True):
            st.rerun()
        return

    top_navigation("Patient Search")

    # Page-specific pastel mint palette requested for Patient Search.
    st.markdown(
        """
        <style>
            .stApp,
            [data-testid="stAppViewContainer"] {
                background: #D0E4EE !important;
            }

            .patient-search-hero {
                background: rgba(255, 255, 255, 0.64) !important;
                border-color: rgba(39, 126, 96, 0.20) !important;
            }

            .search-tip-card,
            div[data-testid="stForm"],
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.62) !important;
                border-color: rgba(39, 126, 96, 0.20) !important;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                background: rgba(255, 255, 255, 0.62) !important;
                border-radius: 12px;
                padding: 0.15rem 0.35rem;
            }

            /* Keep read-only medical-history text fully visible. */
            div[data-testid="stTextInput"] input:disabled,
            div[data-testid="stTextArea"] textarea:disabled {
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                opacity: 1 !important;
                background: #ffffff !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Back to dashboard"):
        st.session_state.active_page = "dashboard"
        st.rerun()

    st.markdown(
        """
        <div class="patient-search-hero">
            <h1>🔎 Search Patient Records</h1>
            <p>Find a patient using an exact user ID or partial symptom and disease text.</p>
        </div>
        <div class="search-tip-grid">
            <div class="search-tip-card tip-blue">
                🪪 User ID
                <small>Search with an exact patient ID.</small>
            </div>
            <div class="search-tip-card tip-pink">
                🩺 Symptoms
                <small>Enter one or more symptom keywords.</small>
            </div>
            <div class="search-tip-card tip-green">
                💚 Disease
                <small>Search using a full or partial disease name.</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("patient_search_form"):
        id_column, symptom_column, disease_column = st.columns(3)

        with id_column:
            user_id = st.text_input(
                "User ID",
                placeholder="Example: P-1001",
                help="User ID matching is exact and case-insensitive.",
            )

        with symptom_column:
            symptoms = st.text_input(
                "Symptoms",
                placeholder="Example: fever",
                help="Partial, case-insensitive matching is supported.",
            )

        with disease_column:
            disease = st.text_input(
                "Disease",
                placeholder="Example: diabetes",
                help="Partial, case-insensitive matching is supported.",
            )

        submitted = st.form_submit_button(
            "Search patients",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not any(clean_text(value) for value in (user_id, symptoms, disease)):
            st.warning("Enter a user ID, symptom, or disease before searching.")
            st.session_state.pop("patient_search_results", None)
            return

        try:
            with st.spinner("Searching patient records..."):
                st.session_state.patient_search_results = search_patients(
                    user_id=user_id,
                    symptoms=symptoms,
                    disease=disease,
                )
        except Exception as error:
            st.error("Patient records could not be searched.")
            st.code(str(error))
            st.session_state.pop("patient_search_results", None)
            return

    results = st.session_state.get("patient_search_results")

    if results is None:
        st.info("Enter at least one search field, then select **Search patients**.")
        return

    if not results:
        st.warning("No matching patient records were found.")
        return

    st.success(f"Found {len(results)} matching patient record(s).")

    records_tab, history_tab, report_tab = st.tabs(
        ["📋 Search Results", "🩺 Patient Medical History", "📄 Diagnostic Report"]
    )

    with records_tab:
        display_columns = {
            "user_id": "User ID",
            "full_name": "Full name",
            "age": "Age",
            "gender": "Gender",
            "updated_at": "Updated at",
        }
        visible_columns = list(display_columns.keys())
        result_table = pd.DataFrame(results)
        result_table = result_table.reindex(columns=visible_columns).rename(
            columns=display_columns
        )
        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True,
        )

    with history_tab:
        patient_options = {
            f"{clean_text(record.get('full_name')) or 'Unnamed patient'} "
            f"({clean_text(record.get('user_id'))})": record
            for record in results
        }
        selected_label = st.selectbox(
            "Select patient",
            options=list(patient_options.keys()),
            help="Choose a patient from the current search results.",
        )
        selected_patient = patient_options[selected_label]
        selected_user_id = clean_text(selected_patient.get("user_id"))
        selected_history = load_medical_history(selected_user_id)

        st.info(
            f"Medical history for **{clean_text(selected_patient.get('full_name')) or 'Unnamed patient'}** "
            f"· User ID: **{selected_user_id}**"
        )

        if not selected_history.get("updated_at"):
            st.warning("This patient has not saved a medical history yet.")
        else:
            st.caption(f"Last updated: {selected_history['updated_at']}")

            first_left, first_right = st.columns(2, gap="large")
            with first_left:
                st.text_area(
                    "Medical Conditions",
                    value=selected_history["conditions"],
                    height=130,
                    disabled=True,
                    key=f"doctor_history_conditions_{selected_user_id}",
                )
                st.text_area(
                    "Current Medications",
                    value=selected_history["current_medications"],
                    height=130,
                    disabled=True,
                    key=f"doctor_history_current_medications_{selected_user_id}",
                )
                st.text_area(
                    "Past Medications",
                    value=selected_history["past_medications"],
                    height=110,
                    disabled=True,
                    key=f"doctor_history_past_medications_{selected_user_id}",
                )
                st.text_area(
                    "Family History",
                    value=selected_history["family_history"],
                    height=130,
                    disabled=True,
                    key=f"doctor_history_family_{selected_user_id}",
                )

            with first_right:
                st.text_area(
                    "Past Medical Events",
                    value=selected_history["past_events"],
                    height=130,
                    disabled=True,
                    key=f"doctor_history_events_{selected_user_id}",
                )
                st.text_area(
                    "Allergies",
                    value=selected_history["allergies"],
                    height=130,
                    disabled=True,
                    key=f"doctor_history_allergies_{selected_user_id}",
                )
                st.text_input(
                    "Smoking",
                    value=selected_history["smoking"],
                    disabled=True,
                    key=f"doctor_history_smoking_{selected_user_id}",
                )
                st.text_input(
                    "Alcohol Use",
                    value=selected_history["alcohol_use"],
                    disabled=True,
                    key=f"doctor_history_alcohol_{selected_user_id}",
                )
                st.text_input(
                    "Exercise Level",
                    value=selected_history["exercise_level"],
                    disabled=True,
                    key=f"doctor_history_exercise_{selected_user_id}",
                )
                st.text_input(
                    "Diet",
                    value=selected_history["diet"],
                    disabled=True,
                    key=f"doctor_history_diet_{selected_user_id}",
                )
                st.text_input(
                    "Sleep Pattern",
                    value=selected_history["sleep_pattern"],
                    disabled=True,
                    key=f"doctor_history_sleep_{selected_user_id}",
                )

            st.subheader("Additional Medical Information")
            extra_one, extra_two, extra_three = st.columns(3)
            with extra_one:
                st.text_area(
                    "Vaccination History",
                    value=selected_history["vaccination_history"],
                    height=100,
                    disabled=True,
                    key=f"doctor_history_vaccination_{selected_user_id}",
                )
                st.text_area(
                    "Pregnancy History",
                    value=selected_history["pregnancy_history"],
                    height=100,
                    disabled=True,
                    key=f"doctor_history_pregnancy_{selected_user_id}",
                )
            with extra_two:
                st.text_area(
                    "Blood Transfusion History",
                    value=selected_history["blood_transfusion_history"],
                    height=100,
                    disabled=True,
                    key=f"doctor_history_transfusion_{selected_user_id}",
                )
                st.text_area(
                    "Mental Health History",
                    value=selected_history["mental_health_history"],
                    height=100,
                    disabled=True,
                    key=f"doctor_history_mental_{selected_user_id}",
                )
            with extra_three:
                st.text_area(
                    "Implanted Medical Device",
                    value=selected_history["implanted_device"],
                    height=100,
                    disabled=True,
                    key=f"doctor_history_device_{selected_user_id}",
                )
                st.text_area(
                    "Other Information",
                    value=selected_history["other_information"],
                    height=100,
                    disabled=True,
                    key=f"doctor_history_other_{selected_user_id}",
                )


    with report_tab:
        report_patient_options = {
            f"{clean_text(record.get('full_name')) or 'Unnamed patient'} "
            f"({clean_text(record.get('user_id'))})": record
            for record in results
        }
        report_selected_label = st.selectbox(
            "Select patient",
            options=list(report_patient_options.keys()),
            help="Choose a patient from the current search results.",
            key="diagnostic_report_patient_select",
        )
        report_patient = report_patient_options[report_selected_label]
        report_user_id = clean_text(report_patient.get("user_id"))
        latest_report = load_latest_diagnostic_report(report_user_id)

        st.info(
            f"Latest diagnostic report for "
            f"**{clean_text(report_patient.get('full_name')) or 'Unnamed patient'}** "
            f"· User ID: **{report_user_id}**"
        )

        report_content = str(latest_report.get("report_content", "")).strip()
        if not report_content:
            st.warning("No saved diagnostic report is available for this patient.")
        else:
            generated_at = clean_text(latest_report.get("generated_at"))
            if generated_at:
                st.caption(f"Report generated: {generated_at}")

            st.markdown(report_content)

            report_patient_info = latest_report.get("patient_info", {})
            if not isinstance(report_patient_info, dict):
                report_patient_info = {}
            report_patient_info = {
                **report_patient_info,
                "patient_id": report_user_id,
                "patient_name": clean_text(report_patient.get("full_name")) or "Unnamed patient",
                "patient_age": clean_text(report_patient.get("age")),
                "patient_gender": clean_text(report_patient.get("gender")),
            }

            try:
                report_pdf = _general_diagnostics_pdf(
                    report_content,
                    report_patient_info,
                )
                safe_report_user_id = re.sub(
                    r"[^A-Za-z0-9_-]+",
                    "_",
                    report_user_id or "unknown",
                )
                st.download_button(
                    "📥 Download last report as PDF",
                    data=report_pdf,
                    file_name=f"diagnostic_report_{safe_report_user_id}.pdf",
                    mime="application/pdf",
                    key=f"download_latest_report_{safe_report_user_id}",
                    use_container_width=True,
                )
            except Exception as error:
                st.warning(f"PDF generation is unavailable: {error}")



# ---------------------------------------------------------
# Two-photo startup interface shown before SMART CDSS login
# ---------------------------------------------------------
def startup_intro_page():
    if "startup_second_screen_ready" not in st.session_state:
        st.session_state.startup_second_screen_ready = False

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        footer {
            display: none !important;
        }

        html, body, .stApp {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            background: #ffffff !important;
        }

        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        div[data-testid="stImage"] {
            position: fixed !important;
            inset: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
            z-index: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: #ffffff !important;
        }

        div[data-testid="stImage"] img {
            width: 100vw !important;
            height: 100vh !important;
            object-fit: contain !important;
            object-position: center !important;
        }

        div[data-testid="stButton"] {
            position: fixed !important;
            right: 4.2vw !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            z-index: 999999 !important;
            width: auto !important;
        }

        div[data-testid="stButton"] > button {
            min-width: 200px !important;
            min-height: 76px !important;
            padding: 18px 38px !important;
            border-radius: 18px !important;
            border: 3px solid #ffffff !important;
            background: #e00000 !important;
            color: #ffffff !important;
            font-size: 46px !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 26px rgba(0,0,0,0.32) !important;
        }

        div[data-testid="stButton"] > button p {
            font-size: 46px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stButton"] > button:hover {
            background: #b80000 !important;
            color: #ffffff !important;
            transform: scale(1.04);
        }

        div[data-testid="stButton"] > button:focus {
            outline: 4px solid #f0bd38 !important;
            outline-offset: 4px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not STARTUP_SCREEN_1.exists() or not STARTUP_SCREEN_2.exists():
        st.error("Startup images are missing.")
        return

    # Process Enter immediately on its rerun so the intro is not replayed.
    if st.button("Enter", key="startup_enter_button"):
        st.session_state.startup_intro_complete = True
        st.rerun()

    image_slot = st.empty()

    # Show the first startup image only once per session.
    if not st.session_state.startup_second_screen_ready:
        image_slot.image(str(STARTUP_SCREEN_1))
        time.sleep(2)
        st.session_state.startup_second_screen_ready = True

    # Keep the second image visible while waiting for Enter.
    image_slot.image(str(STARTUP_SCREEN_2))


# ---------------------------------------------------------
# App router
# ---------------------------------------------------------
if not st.session_state.startup_intro_complete:
    startup_intro_page()
elif not st.session_state.logged_in:
    login_page()
elif current_user_is_admin():
    admin_page()
elif (
    st.session_state.active_page == "patient_search"
    and current_user_is_doctor()
):
    patient_search_page()
elif st.session_state.active_page == "patient_search":
    st.session_state.active_page = "dashboard"
    st.warning("Patient search is available only to doctor accounts.")
    dashboard_page()
elif st.session_state.active_page == "brain_stroke_diagnosis":
    brain_stroke_diagnosis_page()
elif st.session_state.active_page == "heart_disease_diagnosis":
    heart_disease_diagnosis_page()
elif st.session_state.active_page == "breast_cancer_diagnosis":
    breast_cancer_diagnosis_page()
elif st.session_state.active_page == "clinical_chatbot":
    clinical_chatbot_page()
elif st.session_state.active_page == "general_diagnostics":
    general_diagnostics_page()
elif st.session_state.active_page == "agentic_diagnostic":
    agentic_diagnostic_page()
elif st.session_state.active_page == "user_dashboard":
    user_dashboard_page()
else:
    dashboard_page()
