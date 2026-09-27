"""SETTING_SPECS-Permalink-Muster, Presets und Regler-Grenzen (Standardmuster des Portfolios)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import vi_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "rows_slider": SettingSpec("rows", int, C.DEFAULT_ROWS, C.ROWS_MIN, C.ROWS_MAX),
    "cols_slider": SettingSpec("cols", int, C.DEFAULT_COLS, C.COLS_MIN, C.COLS_MAX),
    "slip_slider": SettingSpec("slip", float, C.DEFAULT_SLIP, C.SLIP_MIN, C.SLIP_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
}
PRESET_KEYS = {"rows": "rows_slider", "cols": "cols_slider", "slip": "slip_slider", "gamma": "gamma_slider"}
STEPS = {"slip_slider": C.SLIP_STEP, "gamma_slider": C.GAMMA_STEP}


def _p(**kw):
    base = {"rows": C.DEFAULT_ROWS, "cols": C.DEFAULT_COLS, "slip": C.DEFAULT_SLIP, "gamma": C.DEFAULT_GAMMA}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Kein Rutschen (deterministisch)": _p(slip=0.0),
    "Starkes Rutschen": _p(slip=0.30),
    "Kleines Raster": _p(rows=3, cols=4),
    "Großes Raster": _p(rows=6, cols=12),
    "Niedriger Diskontfaktor": _p(gamma=0.80),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = round(float(snapped), 3)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Standardfall": "4×8-Raster, Rutschen 0,10: V(Start) −9,23. Wertiteration 44 Sweeps, Politikiteration nur 5 äußere Schritte, aber 1158 Bewertungs-Sweeps (insgesamt 1163) - weniger äußere Schritte heißt nicht weniger Arbeit.",
    "Kein Rutschen (deterministisch)": "Ohne Rutschen ist der direkte Weg entlang der Klippe optimal (V(Start) −0,10). Politikiteration braucht hier ungewöhnlich viele äußere Schritte (11 statt der sonst üblichen 4-5) - viele gleich gute Wege ohne Risiko erzeugen Gleichstände, die erst nach und nach aufgelöst werden.",
    "Starkes Rutschen": "Rutschen 0,30: V(Start) fällt auf −10,12, kaum schlechter als bei 0,10 (−9,23) - die optimale Politik hat sich da schon ganz von der Klippe zurückgezogen und verliert kaum noch etwas an weiterem Rutschen. Wertiteration braucht dafür deutlich mehr Sweeps (102).",
    "Kleines Raster": "3×4-Raster (12 Zustände): V(Start) −4,60, Wertiteration 38 Sweeps - kaum weniger als beim viel größeren Standardraster (44), weil die Sweep-Zahl vor allem vom Diskontfaktor abhängt, nicht von der Zustandszahl.",
    "Großes Raster": "6×12-Raster (72 Zustände): V(Start) −12,77 (der längere Weg kostet mehr Schritte), Wertiteration 49 Sweeps - nur wenig mehr als beim 4×8-Standardraster (44).",
    "Niedriger Diskontfaktor": "γ=0,80: Politikiteration braucht dafür nur 389 Bewertungs-Sweeps statt 1158 bei γ=0,95 - ein niedrigerer Diskontfaktor macht die Politikbewertung viel schneller konvergent, ändert die Politik selbst aber kaum.",
}
