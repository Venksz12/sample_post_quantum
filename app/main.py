from __future__ import annotations

# ============================================================
# PQ-AUTH SENTINEL
# Streamlit Application Entry Point
# ============================================================

import sys
from pathlib import Path

# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 2. THIRD-PARTY IMPORTS
# ============================================================

import streamlit as st

# ============================================================
# 3. APPLICATION IMPORTS
# ============================================================

from app.config import APP_TITLE, APP_SUBTITLE
from app.ml_engine import train_model
from app.session import init_state
from app.ui import inject_css
from app import views

# ============================================================
# 4. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 5. GLOBAL UI
# ============================================================

inject_css()

# ============================================================
# 6. SESSION INITIALIZATION
# ============================================================

init_state()

# ============================================================
# 7. MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Training the behavioural security model..."
)
def get_model():
    """Train the behavioural security model once per runtime."""
    return train_model()


try:
    bundle = get_model()

except FileNotFoundError as exc:
    st.error("Dataset not found.")

    st.markdown(
        f"""
        ### Expected dataset

        ```
        {PROJECT_ROOT / "data" / "pq_ai_authentication_dataset_20000.csv"}
        ```

        Make sure the supplied CSV exists inside the `data` folder.
        """
    )

    st.exception(exc)
    st.stop()

except Exception as exc:
    st.error(
        "The behavioural security model could not be initialized."
    )

    st.markdown(
        """
        The interface loaded, but model initialization failed.
        Check the traceback below for the dependency or dataset issue.
        """
    )

    st.exception(exc)
    st.stop()

# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ◈ PQ-Auth Sentinel")
    st.caption("Post-quantum authentication")
    st.caption("Behavioural threat intelligence")

    st.divider()

    # Navigation
    page = st.radio(
        "Application",
        [
            "Command Center",
            "Identity",
            "Request Portal",
            "Verifier",
            "Demo Lab",
            "Transactions",
            "Scenario Tests & Reports",  # NEW
            "Model Lab",
            "Design Notes",
        ],
    )

    st.divider()

    # Identity status
    key = st.session_state.get("key_material")

    if key is not None:
        st.success("Identity active")
        st.caption(f"Key: {key.key_fingerprint}")
        st.caption(f"Algorithm: {key.algorithm}")
    else:
        st.warning("No active identity")

    st.divider()

    # System status
    st.caption("DATABASE")
    st.write("Session-only")

    st.caption("BEHAVIOURAL MODEL")
    st.write(bundle.model.__class__.__name__)

    st.caption("DATASET")
    st.write(f"{bundle.metrics['dataset_rows']:,} records")

# ============================================================
# 9. PAGE ROUTING
# ============================================================

if page == "Command Center":
    views.render_overview(bundle)

elif page == "Identity":
    views.render_identity()

elif page == "Request Portal":
    views.render_request(bundle)

elif page == "Verifier":
    views.render_verifier(bundle)

elif page == "Demo Lab":
    views.render_demo_lab(bundle)

elif page == "Transactions":
    views.render_transactions()

elif page == "Scenario Tests & Reports":
    # NEW: Scenario testing, graphs, Excel and PDF downloads
    from app.testing_dashboard import render_test_dashboard

    render_test_dashboard(bundle)

elif page == "Model Lab":
    views.render_model(bundle)

elif page == "Design Notes":
    views.render_about()
