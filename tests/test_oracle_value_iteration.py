"""Orakel-Tests mit anderem Rechenweg: Modell per Hand-Neuimplementierung, V* als lineares Programm (linprog), Politikbewertung als lineares
Gleichungssystem, Konvergenzfaktor gamma^k, und die Zahl der äußeren Policy-Iterations-Schritte gegen eine Policy Iteration mit exakter Bewertung
(sie darf nicht vom Rauschen der iterativen Bewertung abhängen - früher entschied dieses Rauschen über Gleichstände der Startbewertung)."""

import numpy as np
import pytest

import vi_grid as G
import vi_solve as S

linprog = pytest.importorskip("scipy.optimize").linprog

_DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}


def _oracle_model(rows, cols, slip):
    goal, start = (rows - 1, cols - 1), (rows - 1, 0)
    cliff = {(rows - 1, c) for c in range(1, cols - 1)}
    P = np.zeros((rows * cols, 4, rows * cols))
    R = np.zeros((rows * cols, 4))
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            for a in range(4):
                if (r, c) == goal:
                    P[s, a, s] = 1.0
                    continue
                perp = (2, 3) if a in (0, 1) else (0, 1)
                for p, d in ((1 - slip, a), (slip / 2, perp[0]), (slip / 2, perp[1])):
                    nr, nc = r + _DIRS[d][0], c + _DIRS[d][1]
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        nr, nc = r, c
                    n = (nr, nc)
                    t, rew = (goal, 10.0) if n == goal else (start, -100.0) if n in cliff else (n, -1.0)
                    P[s, a, t[0] * cols + t[1]] += p
                    R[s, a] += p * rew
    return P, R


def _exact_policy_iteration(P, R, gamma, tol=1e-5):
    n = P.shape[0]
    idx = np.arange(n)
    pol = np.zeros(n, dtype=int)
    outer = 0
    while True:
        outer += 1
        V = np.linalg.solve(np.eye(n) - gamma * P[idx, pol], R[idx, pol])
        Q = R + gamma * P @ V
        ok = Q >= Q.max(axis=1, keepdims=True) - tol
        new = np.where(ok[idx, pol], pol, ok.argmax(axis=1))
        if np.array_equal(new, pol):
            return V, pol, outer
        pol = new


CASES = [(3, 4, 0.1, 0.95), (4, 8, 0.1, 0.95), (4, 12, 0.1, 0.95), (6, 12, 0.1, 0.95), (4, 8, 0.0, 0.95), (4, 8, 0.3, 0.95), (4, 8, 0.1, 0.80),
         (4, 8, 0.1, 0.90), (2, 5, 0.2, 0.9), (1, 4, 0.1, 0.9)]


@pytest.mark.parametrize("rows,cols,slip,gamma", CASES)
def test_model_and_optimum_against_independent_oracles(rows, cols, slip, gamma):
    P, R = G.build_model(G.Grid(rows, cols, slip, gamma))
    P2, R2 = _oracle_model(rows, cols, slip)
    assert np.allclose(P, P2) and np.allclose(R, R2)
    n = rows * cols
    V, pol, hist, sweeps = S.value_iteration(P, R, gamma)
    A = np.array([gamma * P2[s, a] - np.eye(n)[s] for s in range(n) for a in range(4)])
    b = np.array([-R2[s, a] for s in range(n) for a in range(4)])
    lp = linprog(np.ones(n), A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
    assert lp.status == 0
    assert np.allclose(V, lp.x, atol=2e-6)                                                  # V* aus dem LP (kleinste Obermenge der Bellman-Ungleichungen)
    idx = np.arange(n)
    Vp = np.linalg.solve(np.eye(n) - gamma * P2[idx, pol], R2[idx, pol])
    assert np.allclose(Vp, lp.x, atol=2e-6)                                                 # die gierige Policy ist exakt bewertet optimal
    Vk = np.zeros(n)
    for k in range(1, min(len(hist), 30)):
        Vk = (R2 + gamma * P2 @ Vk).max(axis=1)
        assert np.allclose(hist[k], Vk)                                                     # Verlauf = k-fache Bellman-Anwendung, kein Versatz um einen Sweep
        assert np.abs(Vk - lp.x).max() <= gamma ** k * np.abs(lp.x).max() + 1e-9           # Konvergenzfaktor gamma^k
    assert len(hist) == sweeps + 1


@pytest.mark.parametrize("rows,cols,slip,gamma", CASES)
def test_outer_steps_do_not_depend_on_evaluation_noise(rows, cols, slip, gamma):
    P, R = G.build_model(G.Grid(rows, cols, slip, gamma))
    V_ex, pol_ex, outer_ex = _exact_policy_iteration(P, R, gamma)
    for tol in (1e-7, 1e-8, 1e-10):
        V, pol, outer, _, _ = S.policy_iteration(P, R, gamma, eval_tol=tol)
        assert outer == outer_ex, (tol, outer, outer_ex)
        assert np.array_equal(pol, pol_ex)
        assert np.allclose(V, V_ex, atol=1e-4)
