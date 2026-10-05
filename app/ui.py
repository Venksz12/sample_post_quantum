from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
      --ink: #10221b;
      --muted: #60756d;
      --paper: #f5f7f1;
      --panel: rgba(255,255,255,.86);
      --line: #dce5df;
      --accent: #0f766e;
      --accent-2: #155e75;
      --danger: #b42318;
      --warn: #a15c00;
      --good: #087443;
    }

    html, body, [class*="css"] {
      font-family: "DM Sans", sans-serif;
    }
    .stApp {
      background:
        radial-gradient(circle at 8% 2%, rgba(20, 184, 166, .12), transparent 28%),
        radial-gradient(circle at 94% 12%, rgba(14, 116, 144, .10), transparent 25%),
        var(--paper);
      color: var(--ink);
    }
    h1, h2, h3 {
      font-family: "Space Grotesk", sans-serif !important;
      letter-spacing: -.025em;
    }
    [data-testid="stSidebar"] {
      background: #10221b;
      border-right: 0;
    }
    [data-testid="stSidebar"] * {
      color: #edf5f1 !important;
    }
    .brand {
      padding: 10px 0 24px;
    }
    .brand-mark {
      display: inline-flex;
      width: 44px; height: 44px;
      align-items: center; justify-content: center;
      border: 1px solid rgba(255,255,255,.22);
      border-radius: 14px;
      font-weight: 800;
      font-size: 20px;
      background: linear-gradient(145deg, #164e63, #0f766e);
    }
    .brand-name { margin-top: 12px; font-size: 19px; font-weight: 700; }
    .brand-sub { color: #a7bbb3 !important; font-size: 12px; line-height: 1.5; }
    .hero {
      padding: 28px 30px;
      border: 1px solid var(--line);
      border-radius: 26px;
      background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(237,246,241,.88));
      box-shadow: 0 16px 50px rgba(16,34,27,.07);
      margin-bottom: 22px;
    }
    .eyebrow {
      font-size: 11px; font-weight: 800; letter-spacing: .16em;
      color: var(--accent); text-transform: uppercase;
    }
    .hero h1 { font-size: 40px; margin: 6px 0 8px; }
    .hero p { color: var(--muted); max-width: 850px; font-size: 15px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 10px 34px rgba(16,34,27,.055);
      height: 100%;
    }
    .kpi {
      font-family: "Space Grotesk", sans-serif;
      font-size: 30px; font-weight: 700; margin-top: 6px;
    }
    .label {
      color: var(--muted); font-size: 12px; text-transform: uppercase;
      letter-spacing: .08em; font-weight: 700;
    }
    .pill {
      display: inline-block; padding: 5px 10px; border-radius: 999px;
      font-size: 11px; font-weight: 800; letter-spacing: .04em;
    }
    .pill-good { background:#dff7eb; color:#087443; }
    .pill-warn { background:#fff0d8; color:#8a4b00; }
    .pill-bad { background:#fee4e2; color:#a61b12; }
    .pill-neutral { background:#e7eef1; color:#37515a; }
    .mono {
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px; word-break: break-all;
    }
    .flow {
      display:flex; align-items:center; gap:10px; flex-wrap:wrap;
      margin: 12px 0 4px;
    }
    .node {
      padding: 10px 13px; border-radius: 12px; background:#fff;
      border:1px solid var(--line); font-size:12px; font-weight:700;
    }
    .arrow { color: var(--accent); font-weight:900; }
    div[data-testid="stButton"] > button {
      border-radius: 12px !important;
      font-weight: 700 !important;
      min-height: 42px;
    }
    .section-note {
      color: var(--muted); font-size: 13px; line-height: 1.6;
    }
    .stMetric {
    background: rgba(255,255,255,.92) !important;
    border: 1px solid #dce5df !important;
    padding: 16px !important;
    border-radius: 16px !important;
}

[data-testid="stMetricLabel"] {
    color: #60756d !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #10221b !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #60756d !important;
}
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "PQ-AUTH SENTINEL") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """, unsafe_allow_html=True
    )


def status_pill(text: str) -> str:
    t = str(text).upper()
    cls = "pill-neutral"
    if t in {"VALID", "ALLOW", "LOW", "SUCCESS"}:
        cls = "pill-good"
    elif t in {"FLAG", "MEDIUM", "PENDING"}:
        cls = "pill-warn"
    elif t in {"INVALID", "BLOCK", "HIGH", "FAILED"}:
        cls = "pill-bad"
    return f'<span class="pill {cls}">{text}</span>'


def info_card(title: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="label">{title}</div>
          <div class="kpi">{value}</div>
          <div class="section-note">{detail}</div>
        </div>
        """, unsafe_allow_html=True
    )
