from __future__ import annotations

import time


WINDOW_SECONDS = 60.0


def _ensure_event_state(st) -> None:

    if "auth_events" not in st.session_state:
        st.session_state.auth_events = []

    if "session_started_at" not in st.session_state:
        st.session_state.session_started_at = time.time()


def record_auth_event(
    st,
    *,
    success: bool,
    device_context: str,
    certificate_valid: int = 1,
) -> dict:

    _ensure_event_state(st)

    now = time.time()
    events = st.session_state.auth_events

    if events:
        previous_time = events[-1]["time"]
        inter_request_time = max(
            now - previous_time,
            0.1,
        )
    else:
        inter_request_time = 30.0

    events.append(
        {
            "time": now,
            "success": bool(success),
            "device_context": device_context,
            "certificate_valid": int(certificate_valid),
        }
    )

    cutoff = now - 300.0

    st.session_state.auth_events = [
        event
        for event in events
        if event["time"] >= cutoff
    ]

    recent_events = st.session_state.auth_events

    recent_60 = [
        event
        for event in recent_events
        if event["time"] >= now - WINDOW_SECONDS
    ]

    request_frequency = len(recent_60)

    if len(recent_events) >= 2:

        intervals = [
            recent_events[i]["time"]
            - recent_events[i - 1]["time"]
            for i in range(1, len(recent_events))
        ]

        inter_request_time_sec = (
            sum(intervals) / len(intervals)
        )

    else:

        inter_request_time_sec = inter_request_time

    failed_attempts = sum(
        1
        for event in recent_events
        if not event["success"]
    )

    session_duration_min = (
        now - st.session_state.session_started_at
    ) / 60.0

    device_change = 0

    if len(recent_events) >= 2:

        previous_device = recent_events[-2]["device_context"]

        if previous_device != device_context:
            device_change = 1

    certificate_state = int(certificate_valid)

    frequency_signal = min(
        request_frequency / 20.0,
        1.0,
    )

    timing_signal = (
        1.0
        if inter_request_time_sec < 2.0
        else min(
            2.0 / inter_request_time_sec,
            1.0,
        )
    )

    failure_signal = min(
        failed_attempts / 5.0,
        1.0,
    )

    device_signal = float(device_change)

    certificate_signal = (
        1.0 - float(certificate_state)
    )

    request_pattern_score = (
        0.30 * frequency_signal
        + 0.25 * timing_signal
        + 0.20 * failure_signal
        + 0.15 * device_signal
        + 0.10 * certificate_signal
    )

    request_pattern_score = max(
        0.0,
        min(
            1.0,
            request_pattern_score,
        ),
    )

    return {
        "request_frequency": float(request_frequency),
        "inter_request_time_sec": float(
            max(inter_request_time_sec, 0.1)
        ),
        "failed_attempts": int(failed_attempts),
        "session_duration_min": float(
            max(session_duration_min, 0.1)
        ),
        "device_change": int(device_change),
        "certificate_valid": int(certificate_state),
        "request_pattern_score": float(
            round(request_pattern_score, 3)
        ),
    }


def get_live_metrics(st) -> dict:

    _ensure_event_state(st)

    now = time.time()

    events = st.session_state.auth_events

    recent = [
        event
        for event in events
        if event["time"] >= now - WINDOW_SECONDS
    ]

    request_frequency = len(recent)

    if len(events) >= 2:

        intervals = [
            events[i]["time"]
            - events[i - 1]["time"]
            for i in range(1, len(events))
        ]

        avg_interval = (
            sum(intervals) / len(intervals)
        )

    else:

        avg_interval = 0.0

    failed_attempts = sum(
        1
        for event in events
        if not event["success"]
    )

    session_duration_min = (
        now - st.session_state.session_started_at
    ) / 60.0

    return {
        "request_frequency": request_frequency,
        "inter_request_time_sec": round(
            avg_interval,
            2,
        ),
        "failed_attempts": failed_attempts,
        "session_duration_min": round(
            session_duration_min,
            2,
        ),
    }


# ============================================================
# DEMONSTRATION / SECURITY LAB TELEMETRY
# ============================================================

def generate_normal_telemetry() -> dict:

    return {
        "request_frequency": 4.5,
        "inter_request_time_sec": 18.0,
        "failed_attempts": 0,
        "session_duration_min": 24.0,
        "device_change": 0,
        "certificate_valid": 1,
        "request_pattern_score": 0.12,
    }


def generate_suspicious_telemetry() -> dict:

    return {
        "request_frequency": 6.5,
        "inter_request_time_sec": 3.5,
        "failed_attempts": 2,
        "session_duration_min": 5.0,
        "device_change": 1,
        "certificate_valid": 1,
        "request_pattern_score": 0.62,
    }


def generate_flooding_telemetry() -> dict:

    return {
        "request_frequency": 18.0,
        "inter_request_time_sec": 0.7,
        "failed_attempts": 5,
        "session_duration_min": 2.0,
        "device_change": 0,
        "certificate_valid": 1,
        "request_pattern_score": 0.91,
    }