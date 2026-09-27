"""Das Vehikel von Hand nachgerechnet: Zustände, Zellenarten, das Übergangsmodell (Wahrscheinlichkeiten, erwartete Belohnung)."""

import numpy as np
import pytest

import vi_grid as G


def test_start_goal_and_cliff_positions():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    assert g.start == (3, 0) and g.goal == (3, 7)
    assert g.cliff == frozenset((3, c) for c in range(1, 7))
    assert g.n_states == 32


def test_cell_kind_by_hand():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    assert g.cell_kind(g.start) == "start" and g.cell_kind(g.goal) == "goal"
    assert g.cell_kind((3, 3)) == "cliff" and g.cell_kind((0, 0)) == "free"


def test_state_index_round_trip():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    for r in range(g.rows):
        for c in range(g.cols):
            assert g.rc_of(g.state_of((r, c))) == (r, c)


def test_transition_probabilities_sum_to_one_everywhere():
    g = G.Grid(rows=4, cols=8, slip=0.15, gamma=0.95)
    P, R = G.build_model(g)
    assert np.allclose(P.sum(axis=2), 1.0)


def test_goal_is_absorbing_with_zero_reward():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, R = G.build_model(g)
    goal_s = g.state_of(g.goal)
    for a in G.ACTIONS:
        assert P[goal_s, a, goal_s] == pytest.approx(1.0) and R[goal_s, a] == pytest.approx(0.0)


def test_deterministic_move_by_hand_no_slip():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    P, R = G.build_model(g)
    s = g.state_of((0, 3))                                                                   # eine freie Zelle, weit weg von Ziel/Klippe
    ns = g.state_of((0, 4))
    assert P[s, G.EAST, ns] == pytest.approx(1.0) and R[s, G.EAST] == pytest.approx(-1.0)


def test_boundary_bounce_back_by_hand():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    P, R = G.build_model(g)
    s = g.state_of((0, 0))                                                                   # obere linke Ecke
    assert P[s, G.NORTH, s] == pytest.approx(1.0) and R[s, G.NORTH] == pytest.approx(-1.0)     # gegen die Wand: bleibt stehen, kostet trotzdem -1
    assert P[s, G.WEST, s] == pytest.approx(1.0)


def test_slip_splits_into_the_two_orthogonal_directions_by_hand():
    g = G.Grid(rows=4, cols=8, slip=0.2, gamma=0.95)
    P, R = G.build_model(g)
    s = g.state_of((1, 3))                                                                    # frei nach allen vier Seiten
    east, north, south = g.state_of((1, 4)), g.state_of((0, 3)), g.state_of((2, 3))
    assert P[s, G.EAST, east] == pytest.approx(0.8) and P[s, G.EAST, north] == pytest.approx(0.1) and P[s, G.EAST, south] == pytest.approx(0.1)
    assert R[s, G.EAST] == pytest.approx(-1.0)                                                 # alle drei Ausgänge sind gewöhnliche Zellen: -1 in jedem Fall


def test_entering_the_cliff_costs_the_penalty_and_resets_to_start():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    P, R = G.build_model(g)
    s = g.state_of((2, 3))                                                                     # direkt über der Klippenzelle (3,3)
    start_s = g.state_of(g.start)
    assert P[s, G.SOUTH, start_s] == pytest.approx(1.0) and R[s, G.SOUTH] == pytest.approx(-100.0)


def test_entering_the_goal_gives_the_reward():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    P, R = G.build_model(g)
    s = g.state_of((2, 7))                                                                     # direkt über der Packstation
    goal_s = g.state_of(g.goal)
    assert P[s, G.SOUTH, goal_s] == pytest.approx(1.0) and R[s, G.SOUTH] == pytest.approx(10.0)


def test_naive_policy_walks_the_row_above_the_cliff_then_south():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    pol = G.naive_policy(g)
    row = g.rows - 2
    for c in range(g.cols - 1):
        assert pol[g.state_of((row, c))] == G.EAST
    assert pol[g.state_of((row, g.cols - 1))] == G.SOUTH
    assert pol[g.state_of((0, 0))] == G.SOUTH                                                  # oberhalb der Zielreihe: nach Süden Richtung Zielreihe
