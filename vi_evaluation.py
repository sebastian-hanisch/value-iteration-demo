"""Auswertung: Value Iteration und Policy Iteration auf demselben Modell, dazu drei Experimente (Sweeps über Rastergröße und Diskontfaktor, Wirkung der
Rutsch-Wahrscheinlichkeit auf die Policy, Wirkung des Diskontfaktors auf die Policy gegen die Konvergenzgeschwindigkeit). Das Modell ist deterministisch
(kein Zufall in der Erzeugung) - jede Zahl ist exakt, keine Seeds, keine Bänder nötig."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import vi_constants as C
import vi_grid as G
import vi_solve as S


@dataclass(frozen=True)
class Settings:
    rows: int = C.DEFAULT_ROWS
    cols: int = C.DEFAULT_COLS
    slip: float = C.DEFAULT_SLIP
    gamma: float = C.DEFAULT_GAMMA

    @property
    def grid(self):
        return G.Grid(self.rows, self.cols, self.slip, self.gamma)


@lru_cache(maxsize=64)
def _model(key):
    rows, cols, slip, gamma = key
    grid = G.Grid(rows, cols, slip, gamma)
    return grid, G.build_model(grid)


@dataclass
class Analysis:
    settings: Settings
    grid: G.Grid
    P: np.ndarray
    R: np.ndarray
    V_vi: np.ndarray
    policy_vi: np.ndarray
    hist_vi: np.ndarray
    sweeps_vi: int
    V_pi: np.ndarray
    policy_pi: np.ndarray
    outer_pi: int
    eval_sweeps_pi: int
    hist_pi: list


def analyse(s):
    grid, (P, R) = _model((s.rows, s.cols, s.slip, s.gamma))
    V_vi, policy_vi, hist_vi, sweeps_vi = S.value_iteration(P, R, s.gamma)
    V_pi, policy_pi, outer_pi, eval_sweeps_pi, hist_pi = S.policy_iteration(P, R, s.gamma)
    return Analysis(s, grid, P, R, V_vi, policy_vi, hist_vi, sweeps_vi, V_pi, policy_pi, outer_pi, eval_sweeps_pi, hist_pi)


# --- Experiment 1: Sweeps über Rastergröße und Diskontfaktor ---------------------------------------------------------------------------------------

def sweeps_experiment(sizes=None, slip=None, gamma=None):
    sizes = C.EXP_SIZES if sizes is None else sizes
    slip = C.DEFAULT_SLIP if slip is None else slip
    gamma = C.DEFAULT_GAMMA if gamma is None else gamma
    rows = []
    for rr, cc in sizes:
        grid, (P, R) = _model((rr, cc, slip, gamma))
        _, _, _, sweeps_vi = S.value_iteration(P, R, gamma)
        _, _, outer, eval_sweeps, _ = S.policy_iteration(P, R, gamma)
        rows.append({"rows": rr, "cols": cc, "n_states": grid.n_states, "vi_sweeps": sweeps_vi, "pi_outer": outer, "pi_eval_sweeps": eval_sweeps, "pi_total": outer + eval_sweeps})
    return {"sizes": tuple(sizes), "rows": rows}


# --- Experiment 2: Rutsch-Wahrscheinlichkeit gegen die Policy --------------------------------------------------------------------------------------

def slip_experiment(levels=None, rows=None, cols=None, gamma=None):
    levels = C.EXP_SLIP_LEVELS if levels is None else levels
    rows = C.DEFAULT_ROWS if rows is None else rows
    cols = C.DEFAULT_COLS if cols is None else cols
    gamma = C.DEFAULT_GAMMA if gamma is None else gamma
    out = []
    for slip in levels:
        grid, (P, R) = _model((rows, cols, slip, gamma))
        V, policy, _, _ = S.value_iteration(P, R, gamma)
        naive = G.naive_policy(grid)
        V_naive, _ = S.policy_evaluation(P, R, naive, gamma)
        s0 = grid.state_of(grid.start)
        row_idx = rows - 2
        east_count = sum(1 for c in range(cols - 1) if policy[grid.state_of((row_idx, c))] == G.EAST)
        out.append({"slip": slip, "v_star": float(V[s0]), "v_naive": float(V_naive[s0]), "gap": float(V[s0] - V_naive[s0]), "east_cells": east_count, "east_total": cols - 1})
    return {"levels": tuple(levels), "rows": out}


# --- Experiment 3: Diskontfaktor - Policy gegen Konvergenzgeschwindigkeit ---------------------------------------------------------------------------

def gamma_experiment(levels=None, rows=None, cols=None, slip=None):
    levels = C.EXP_GAMMA_LEVELS if levels is None else levels
    rows = C.DEFAULT_ROWS if rows is None else rows
    cols = C.DEFAULT_COLS if cols is None else cols
    slip = C.DEFAULT_SLIP if slip is None else slip
    out = []
    prev_policy = None
    for gamma in levels:
        grid, (P, R) = _model((rows, cols, slip, gamma))
        V, policy, _, sweeps_vi = S.value_iteration(P, R, gamma)
        _, _, outer, eval_sweeps, _ = S.policy_iteration(P, R, gamma)
        diff = int(np.sum(policy != prev_policy)) if prev_policy is not None else 0
        out.append({"gamma": gamma, "vi_sweeps": sweeps_vi, "pi_eval_sweeps": eval_sweeps, "policy_diff_vs_previous": diff})
        prev_policy = policy
    return {"levels": tuple(levels), "rows": out}
