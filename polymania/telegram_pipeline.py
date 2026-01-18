import os
import sys
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def _err(msg: str, code: int = 1) -> None:
    print(f"[PolyMania][TG PIPELINE] ERROR: {msg}")
    sys.exit(code)


def _info(msg: str) -> None:
    print(f"[PolyMania][TG PIPELINE] {msg}")


def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        _err(f"Missing {name} in .env")
    return val


def _find_session_file(session_name: str) -> Path | None:
    """
    Telethon stores sessions in the CWD by default as <name>.session
    (sometimes also creates <name>.session-journal).
    We'll check project root first.
    """
    candidates = [
        Path(f"{session_name}.session"),
        Path("polymania") / f"{session_name}.session",  # fallback if someone ran from inside folder
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _ensure_telethon_installed() -> None:
    try:
        import telethon  # noqa: F401
    except Exception:
        _err("Telethon is not installed in the active venv. Run: python -m pip install -r requirements.txt")


def _run_module(module: str) -> None:
    _info(f"Running: python -m {module}")
    subprocess.run([sys.executable, "-m", module], check=True)


def main() -> None:
    load_dotenv()

    _info("Starting Telegram pipeline (login-if-needed -> scan-once)")

    # 1) Dependency check
    _ensure_telethon_installed()

    # 2) Env validation (login)
    api_id = _require_env("TG_API_ID")
    api_hash = _require_env("TG_API_HASH")
    session_name = os.getenv("TG_SESSION_NAME", "polymania_user").strip()

    _info(f"TG_API_ID set: {bool(api_id)} | TG_API_HASH set: {bool(api_hash)} | session: {session_name!r}")

    # 3) Env validation (scan)
    channels = os.getenv("TG_WATCH_CHANNELS", "").strip()
    keywords = os.getenv("TG_WATCH_KEYWORDS", "").strip()
    hits_csv = os.getenv("TG_HITS_CSV", "telegram_hits.csv").strip()

    if not channels:
        _err("TG_WATCH_CHANNELS is empty. Put comma-separated channel usernames/links in .env.")
    if not keywords:
        _err("TG_WATCH_KEYWORDS is empty. Put comma-separated keywords in .env.")

    _info(f"Channels configured: {len([c for c in channels.split(',') if c.strip()])}")
    _info(f"Keywords configured: {len([k for k in keywords.split(',') if k.strip()])}")
    _info(f"Hits output CSV: {hits_csv}")

    # 4) Session presence check -> login if missing
    session_file = _find_session_file(session_name)
    if session_file is None:
        _info("No .session file found -> starting FIRST LOGIN (you will need to enter phone/code/password).")
        _run_module("polymania.telegram_first_login")
        session_file = _find_session_file(session_name)
        if session_file is None:
            _err("Login finished but session file still not found. Check TG_SESSION_NAME and rerun.")
    else:
        _info(f"Session found: {session_file}")

    # 5) Run scan once
    _info("Running channel scan once...")
    _run_module("polymania.telegram_watch_channels")

    _info("Done.")
    _info(f"Check results in: {hits_csv}")
    _info("Tip: if you get 'Cannot find entity', run: python -m polymania.telegram_list_dialogs and copy the exact channel identifiers.")


if __name__ == "__main__":
    main()
