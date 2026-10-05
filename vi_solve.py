"""Value Iteration und Policy Iteration auf dem bekannten Modell (P, R) - Bellman (1957), Howard (1960). Beide liefern dieselbe optimale Value Function und
Policy (bis auf Gleichstände); die Policy Evaluation von Policy Iteration läuft selbst iterativ (nicht als exakte Lösung eines linearen Gleichungssystems),
damit sich der Aufwand beider Verfahren in derselben Einheit (Sweeps: eine vollständige Runde über alle States) vergleichen lässt."""

import numpy as np

import vi_constants as C


def q_values(P, R, V, gamma):
    """Q(s, a) = R(s, a) + gamma * sum_s' P(s, a, s') V(s')."""
    return R + gamma * np.einsum("sat,t->sa", P, V)


def greedy_policy(Q, current=None, tol=C.TIE_TOL):
    """Gierige Policy mit festem Gleichstand: unter allen Actions, deren Q höchstens `tol` unter dem Maximum liegt, gilt die bisherige Action (falls
    `current` gegeben und darunter), sonst die mit dem kleinsten Index. Ohne die Toleranz entscheidet das Rauschen der iterativen Bewertung (~1e-8)
    über Gleichstände - etwa die konstante Startbewertung der Policy 'immer Norden' - und damit über die Zahl der äußeren Schritte."""
    ok = Q >= Q.max(axis=1, keepdims=True) - tol
    policy = ok.argmax(axis=1)
    if current is not None:
        idx = np.arange(Q.shape[0])
        policy = np.where(ok[idx, current], current, policy)
    return policy


def value_iteration(P, R, gamma, tol=C.TOL, max_iter=C.MAX_ITER):
    """Rückgabe: V*, Policy (gierig bezüglich V*), Verlauf von V (eine Zeile je Sweep, Zeile 0 = Start bei 0), Zahl der Sweeps."""
    S = P.shape[0]
    V = np.zeros(S)
    history = [V.copy()]
    for it in range(1, max_iter + 1):
        Q = q_values(P, R, V, gamma)
        newV = Q.max(axis=1)
        history.append(newV.copy())
        delta = float(np.abs(newV - V).max())
        V = newV
        if delta < tol:
            break
    policy = greedy_policy(q_values(P, R, V, gamma))
    return V, policy, np.array(history), it


def policy_evaluation(P, R, policy, gamma, tol=C.TOL, max_iter=C.MAX_ITER):
    """Iterative Policy Evaluation (kein exaktes Gleichungssystem): V_pi als Grenzwert von V <- R_pi + gamma * P_pi @ V. Rückgabe: V_pi, Zahl der Sweeps."""
    S = P.shape[0]
    idx = np.arange(S)
    P_pi = P[idx, policy]
    R_pi = R[idx, policy]
    V = np.zeros(S)
    for it in range(1, max_iter + 1):
        newV = R_pi + gamma * (P_pi @ V)
        delta = float(np.abs(newV - V).max())
        V = newV
        if delta < tol:
            break
    return V, it


def policy_iteration(P, R, gamma, eval_tol=C.TOL, max_outer=C.MAX_ITER, max_eval_iter=C.MAX_ITER):
    """Rückgabe: V_pi der letzten Policy, Policy, Zahl der äußeren Verbesserungsschritte, Gesamtzahl der Bewertungs-Sweeps, Verlauf der Policies."""
    S = P.shape[0]
    policy = np.zeros(S, dtype=int)
    history = [policy.copy()]
    total_eval_sweeps = 0
    V = np.zeros(S)
    for outer in range(1, max_outer + 1):
        V, sweeps = policy_evaluation(P, R, policy, gamma, eval_tol, max_eval_iter)
        total_eval_sweeps += sweeps
        new_policy = greedy_policy(q_values(P, R, V, gamma), current=policy)
        history.append(new_policy.copy())
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy, outer, total_eval_sweeps, history
