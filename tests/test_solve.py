"""Wert- und Politikiteration: die Bellman-Formel von Hand, ein entarteter Fall mit geschlossener Lösung, und die Gegenprobe beider Verfahren
gegeneinander sowie gegen Vollaufzählung auf einem winzigen Raster."""

import itertools

import numpy as np
import pytest

import vi_grid as G
import vi_solve as S


def test_q_values_by_hand():
    P = np.array([[[0.5, 0.5], [1.0, 0.0]]])                                                   # 1 Zustand, 2 Aktionen, 2 Folgezustände
    R = np.array([[1.0, 2.0]])
    V = np.array([10.0, 20.0])
    Q = S.q_values(P, R, V, gamma=0.9)
    assert Q[0, 0] == pytest.approx(1.0 + 0.9 * (0.5 * 10 + 0.5 * 20)) and Q[0, 1] == pytest.approx(2.0 + 0.9 * 10.0)


def test_degenerate_single_row_grid_gives_up_and_bounces_forever():
    # Ein Raster mit nur einer Reihe: der einzige Weg zur Packstation führt zwangsläufig durch die Klippenzelle (kein Ausweichen möglich).
    # Ohne Rutschen ist die optimale Politik, ewig gegen die Wand zu laufen (Kosten -1 je Schritt) statt die Klippe zu riskieren (-100).
    g = G.Grid(rows=1, cols=3, slip=0.0, gamma=0.95)
    P, R = G.build_model(g)
    V, policy, hist, sweeps = S.value_iteration(P, R, g.gamma)
    s0 = g.state_of(g.start)
    assert V[s0] == pytest.approx(-1.0 / (1.0 - g.gamma), abs=1e-4)
    assert policy[s0] == G.NORTH                                                                # gibt auf und prallt gegen die Wand, statt die Klippe zu riskieren


def test_value_and_policy_iteration_agree_on_the_optimum():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, R = G.build_model(g)
    V_vi, pi_vi, _, _ = S.value_iteration(P, R, g.gamma)
    V_pi, pi_pi, _, _, _ = S.policy_iteration(P, R, g.gamma)
    assert np.allclose(V_vi, V_pi, atol=1e-3)
    assert np.array_equal(pi_vi, pi_pi)


def test_value_iteration_matches_exhaustive_enumeration_on_a_tiny_grid():
    # 1x3-Raster: nur 3 Zustände, je 4 Aktionen -> 4^3 = 64 stationäre Politiken vollständig durchprobierbar (die Klippenzelle selbst
    # wird unter der optimalen Politik nie besucht, ihre eigene Aktion ist irrelevant und wird deshalb nicht mitgezählt).
    g = G.Grid(rows=1, cols=3, slip=0.1, gamma=0.9)
    P, R = G.build_model(g)
    V_vi, pi_vi, _, _ = S.value_iteration(P, R, g.gamma)
    s0 = g.state_of(g.start)
    best_v0 = -np.inf
    for pol in itertools.product(G.ACTIONS, repeat=3):
        Vp, _ = S.policy_evaluation(P, R, np.array(pol), g.gamma, max_iter=20000)
        best_v0 = max(best_v0, Vp[s0])
    assert V_vi[s0] == pytest.approx(best_v0, abs=1e-3)


def test_policy_evaluation_of_a_fixed_policy_by_hand():
    # 1 Zustand, ein Selbstübergang mit Belohnung 1, Diskont 0.5: V = 1 / (1 - 0.5) = 2.
    P = np.array([[[1.0]]])
    R = np.array([[1.0]])
    V, sweeps = S.policy_evaluation(P, R, np.array([0]), gamma=0.5)
    assert V[0] == pytest.approx(2.0, abs=1e-6)


def test_policy_iteration_terminates_when_the_policy_stops_changing():
    g = G.Grid(rows=3, cols=4, slip=0.05, gamma=0.9)
    P, R = G.build_model(g)
    _, policy, outer, eval_sweeps, history = S.policy_iteration(P, R, g.gamma)
    assert np.array_equal(history[-1], policy) and np.array_equal(history[-2], policy)          # letzter Verbesserungsschritt ändert nichts mehr
    assert outer >= 1 and eval_sweeps >= outer                                                  # mindestens ein Sweep je Verbesserungsschritt


def test_more_slip_never_helps_the_optimal_value_at_the_start():
    g0 = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    g1 = G.Grid(rows=4, cols=8, slip=0.2, gamma=0.95)
    V0, _, _, _ = S.value_iteration(*G.build_model(g0), g0.gamma)
    V1, _, _, _ = S.value_iteration(*G.build_model(g1), g1.gamma)
    assert V1[g1.state_of(g1.start)] <= V0[g0.state_of(g0.start)] + 1e-6                        # mehr Zufall kann den optimalen Ertrag nur verschlechtern
