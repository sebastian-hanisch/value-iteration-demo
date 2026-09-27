"""Analyse und die drei Experimente: Aufbau, Übereinstimmung von Wert- und Politikiteration, die erwarteten Muster (Zahlen stehen in test_claims.py)."""

import numpy as np
import pytest

import vi_evaluation as E
import vi_grid as G


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings())


def test_analysis_agrees_between_value_and_policy_iteration(analysis):
    a = analysis
    assert np.allclose(a.V_vi, a.V_pi, atol=1e-3) and np.array_equal(a.policy_vi, a.policy_pi)
    assert a.hist_vi.shape[0] == a.sweeps_vi + 1                                                # Zeile 0 ist der Start bei 0
    assert a.grid.n_states == a.settings.rows * a.settings.cols


def test_analyse_is_deterministic_and_cached():
    s = E.Settings(rows=4, cols=6, slip=0.05, gamma=0.9)
    a1, a2 = E.analyse(s), E.analyse(s)
    assert np.array_equal(a1.policy_vi, a2.policy_vi) and a1.V_vi.tolist() == a2.V_vi.tolist()
    assert E._model((4, 6, 0.05, 0.9)) is E._model((4, 6, 0.05, 0.9))


def test_sweeps_experiment_shapes_and_that_policy_iteration_needs_fewer_outer_steps():
    exp = E.sweeps_experiment(sizes=((3, 4), (4, 8)))
    assert len(exp["rows"]) == 2
    for row in exp["rows"]:
        assert row["n_states"] == row["rows"] * row["cols"]
        assert row["pi_outer"] < row["vi_sweeps"]                                               # Politikiteration braucht weniger äußere Schritte
        assert row["pi_total"] == row["pi_outer"] + row["pi_eval_sweeps"]


def test_slip_experiment_the_gap_grows_and_the_policy_moves_away_from_the_cliff():
    exp = E.slip_experiment(levels=(0.0, 0.1, 0.3))
    rows = {r["slip"]: r for r in exp["rows"]}
    assert rows[0.0]["gap"] == pytest.approx(0.0, abs=1e-3)                                     # ohne Rutschen ist der naive Weg schon optimal
    assert rows[0.1]["gap"] > rows[0.0]["gap"] and rows[0.3]["gap"] > rows[0.1]["gap"]
    assert rows[0.0]["east_cells"] >= rows[0.1]["east_cells"] >= rows[0.3]["east_cells"]         # die Politik weicht mit mehr Rutschen weiter aus


def test_gamma_experiment_shapes_and_policy_diff_is_defined_from_the_second_row_on():
    exp = E.gamma_experiment(levels=(0.8, 0.9, 0.95))
    assert exp["rows"][0]["policy_diff_vs_previous"] == 0
    assert all("policy_diff_vs_previous" in r for r in exp["rows"])


def test_naive_policy_matches_optimal_only_without_slip():
    # Klippenzellen sind unter jeder vernünftigen Politik unerreichbar (jeder Übergang dorthin wird sofort zum Start umgeleitet) - ihre
    # eigene "Politik" ist bedeutungslos und wird deshalb aus dem Vergleich ausgeschlossen.
    a = E.analyse(E.Settings(slip=0.0))
    naive = G.naive_policy(a.grid)
    reachable = [s for s in range(a.grid.n_states) if a.grid.rc_of(s) not in a.grid.cliff]
    assert np.array_equal(naive[reachable], a.policy_vi[reachable])
