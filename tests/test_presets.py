"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import vi_constants as C
import vi_evaluation as E
import vi_presets as P


def _settings(p):
    return E.Settings(p["rows"], p["cols"], p["slip"], p["gamma"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("slip", "slip_slider"), ("gamma", "gamma_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6, (key, p[key])


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("slip_slider") == (C.SLIP_MIN, C.SLIP_MAX) and P.bounds("rows_slider") == (C.ROWS_MIN, C.ROWS_MAX)
    assert set(P.STEPS) == {"slip_slider", "gamma_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
