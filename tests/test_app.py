"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Sweep-Slider, Permalink-Grenzen/-Raster, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import vi_constants as C
import vi_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 4 and len(at.get("plotly_chart")) >= 2 and len(at.success) + len(at.warning) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == pytest.approx(p[key]) if isinstance(p[key], float) else at.session_state[state_key] == p[key]


def test_step_selection_survives_a_smaller_grid():
    at = _run(vi_step=30)
    _ok(at)
    at.slider(key="cols_slider").set_value(C.COLS_MIN).run()
    _ok(at)
    assert at.session_state["vi_step"] <= 200                                                  # kein Absturz, der Sweep-Slider bleibt gültig


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rows"] = "999"
    at.query_params["cols"] = "-5"
    at.query_params["slip"] = "abc"
    at.query_params["gamma"] = "5"
    at.run()
    _ok(at)
    s = at.session_state
    assert s["rows_slider"] == C.ROWS_MAX and s["cols_slider"] == C.COLS_MIN
    assert s["slip_slider"] == C.DEFAULT_SLIP and s["gamma_slider"] == C.GAMMA_MAX


@pytest.mark.parametrize("kw", [dict(rows_slider=C.ROWS_MIN, cols_slider=C.COLS_MIN, slip_slider=0.0), dict(rows_slider=C.ROWS_MAX, cols_slider=C.COLS_MAX, slip_slider=C.SLIP_MAX),
                                dict(gamma_slider=C.GAMMA_MIN), dict(gamma_slider=C.GAMMA_MAX, slip_slider=C.SLIP_MAX)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_sweeps_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SIZES", ((3, 4), (4, 6)))
    at = _run()
    _click(at, "sweeps_start")
    assert at.session_state["sweeps_on"] and any("Befund" in w.value for w in at.warning)


def test_slip_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SLIP_LEVELS", (0.0, 0.1, 0.2))
    at = _run()
    _click(at, "slip_start")
    assert at.session_state["slip_on"] and any("Mehrkosten" in w.value for w in at.warning)


def test_gamma_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_GAMMA_LEVELS", (0.8, 0.9, 0.95))
    at = _run()
    _click(at, "gamma_start")
    assert at.session_state["gamma_on"] and any("Diskontfaktor" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
