from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

from .config import MAX_SESSION_TRANSACTIONS


def init_state() -> None:
    defaults = {
        "key_material": None,
        "identity": None,
        "transactions": [],
        "last_transaction": None,
        "tamper_mode": False,
    }
    if "transactions" not in st.session_state:
        st.session_state.transactions = []
    if "used_nonces" not in st.session_state:
        st.session_state.used_nonces = set()
    if "auth_events" not in st.session_state:
        st.session_state.auth_events = []
    if "session_started_at" not in st.session_state:
        import time
        st.session_state.session_started_at = time.time()
    if "used_nonces" not in st.session_state:
        st.session_state.used_nonces = set()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_request_id() -> str:
    return "AUTH-" + uuid4().hex[:10].upper()


def new_signature_id() -> str:
    return "SIG-" + uuid4().hex[:10].upper()


def new_session_id() -> str:
    return "SES-" + uuid4().hex[:10].upper()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_transaction(tx: dict) -> None:
    tx = dict(tx)
    st.session_state.transactions.insert(0, tx)
    st.session_state.transactions = st.session_state.transactions[:MAX_SESSION_TRANSACTIONS]
    st.session_state.last_transaction = tx


def transaction_frame() -> pd.DataFrame:
    rows = st.session_state.transactions
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
