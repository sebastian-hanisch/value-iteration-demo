"""Jede Zahl aus README.md und den Preset-Hilfen wird hier nachgerechnet. Das Modell ist deterministisch - jede Zahl ist exakt, keine Bänder nötig
(nur eine kleine Fließkomma-Toleranz)."""

import pytest

import vi_evaluation as E
import vi_presets as P


def _settings(p):
    return E.Settings(p["rows"], p["cols"], p["slip"], p["gamma"])


@pytest.fixture(scope="module")
def sweeps():
    return E.sweeps_experiment()


@pytest.fixture(scope="module")
def slip():
    return E.slip_experiment()


@pytest.fixture(scope="module")
def gamma():
    return E.gamma_experiment()


@pytest.fixture(scope="module")
def presets():
    return {name: E.analyse(_settings(p)) for name, p in P.PRESETS.items()}


def test_sweeps_by_size(sweeps):
    rows = {(r["rows"], r["cols"]): r for r in sweeps["rows"]}
    r0, r1 = rows[(3, 4)], rows[(6, 12)]
    assert r0["vi_sweeps"] == 38 and r0["pi_outer"] == 4 and r0["pi_total"] == 940
    assert r1["vi_sweeps"] == 49 and r1["pi_outer"] == 5 and r1["pi_total"] == 1010
    for r in sweeps["rows"]:
        assert r["pi_outer"] < r["vi_sweeps"] and r["pi_total"] > r["vi_sweeps"]


def test_slip_moves_the_policy_away_from_the_cliff(slip):
    rows = {r["slip"]: r for r in slip["rows"]}
    r0, r1 = rows[0.0], rows[0.30]
    assert r0["gap"] == pytest.approx(0.0, abs=1e-3) and r0["east_cells"] == 7
    assert r1["gap"] == pytest.approx(141.79, abs=0.1) and r1["east_cells"] == 0
    gaps = [rows[lvl]["gap"] for lvl in sorted(rows)]
    assert gaps == sorted(gaps)                                                              # monoton wachsend mit dem Rutschen
    easts = [rows[lvl]["east_cells"] for lvl in sorted(rows)]
    assert easts == sorted(easts, reverse=True)                                              # monoton fallend


def test_gamma_barely_changes_the_policy_but_changes_convergence_speed(gamma):
    total_diff = sum(r["policy_diff_vs_previous"] for r in gamma["rows"])
    assert total_diff == 1
    r0, r1 = gamma["rows"][0], gamma["rows"][-1]
    assert r0["gamma"] == pytest.approx(0.80) and r0["pi_eval_sweeps"] == 389
    assert r1["gamma"] == pytest.approx(0.99) and r1["pi_eval_sweeps"] == 4029
    assert r0["vi_sweeps"] == 68 and r1["vi_sweeps"] == 47


def test_standard_preset(presets):
    a = presets["Standardfall"]
    s0 = a.grid.state_of(a.grid.start)
    assert a.V_vi[s0] == pytest.approx(-9.23, abs=0.02)
    assert a.sweeps_vi == 44 and a.outer_pi == 5 and a.eval_sweeps_pi == 1158 and a.outer_pi + a.eval_sweeps_pi == 1163


def test_no_slip_preset(presets):
    a = presets["Kein Rutschen (deterministisch)"]
    s0 = a.grid.state_of(a.grid.start)
    assert a.V_vi[s0] == pytest.approx(-0.10, abs=0.02) and a.outer_pi == 11


def test_strong_slip_preset(presets):
    a = presets["Starkes Rutschen"]
    s0 = a.grid.state_of(a.grid.start)
    assert a.V_vi[s0] == pytest.approx(-10.12, abs=0.02) and a.sweeps_vi == 102


def test_all_value_and_policy_iteration_agree_across_presets(presets):
    for name, a in presets.items():
        reachable = [s for s in range(a.grid.n_states) if a.grid.rc_of(s) not in a.grid.cliff]
        assert a.V_vi[reachable] == pytest.approx(a.V_pi[reachable], abs=1e-3), name
