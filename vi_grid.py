"""Das Vehikel: ein Lagerroboter auf einem Raster (die Cliff-Walking-Vorlage aus Sutton & Barto 2018, Beispiel 6.6). Start unten links, Packstation (Ziel)
unten rechts, dazwischen in der untersten Zeile ein schmaler Gang entlang der Regalkante ("Klippe"). Vier Richtungen; mit der Rutsch-Wahrscheinlichkeit
`slip` rutscht der Roboter statt der gewählten Richtung mit gleicher Wahrscheinlichkeit auf eine der beiden dazu senkrechten Richtungen. Ein Schritt an
den Rand oder gegen die Wand hält auf der Stelle. Die Packstation ist ein aufnehmender (absorbierender) Zustand: von dort führt jede Aktion zurück
zu ihr selbst, ohne weitere Kosten oder Erträge."""

from dataclasses import dataclass

import numpy as np

import vi_constants as C

NORTH, SOUTH, EAST, WEST = range(4)
ACTIONS = (NORTH, SOUTH, EAST, WEST)
ACTION_NAMES = {NORTH: "Norden", SOUTH: "Süden", EAST: "Osten", WEST: "Westen"}
ACTION_ARROWS = {NORTH: "↑", SOUTH: "↓", EAST: "→", WEST: "←"}
_DELTA = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1)}
_ORTHOGONAL = {NORTH: (EAST, WEST), SOUTH: (EAST, WEST), EAST: (NORTH, SOUTH), WEST: (NORTH, SOUTH)}


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    slip: float
    gamma: float

    @property
    def start(self):
        return (self.rows - 1, 0)

    @property
    def goal(self):
        return (self.rows - 1, self.cols - 1)

    @property
    def cliff(self):
        return frozenset((self.rows - 1, c) for c in range(1, self.cols - 1))

    @property
    def n_states(self):
        return self.rows * self.cols

    def state_of(self, rc):
        r, c = rc
        return r * self.cols + c

    def rc_of(self, s):
        return divmod(s, self.cols)

    def cell_kind(self, rc):
        if rc == self.goal:
            return "goal"
        if rc == self.start:
            return "start"
        if rc in self.cliff:
            return "cliff"
        return "free"


def naive_policy(grid):
    """Vergleichspolitik ohne Rücksicht auf das Rutschen: von jeder Zelle direkt zur Reihe über der Klippe, dort nach Osten bis zur letzten Spalte,
    dann nach Süden zur Packstation - der kürzeste Weg, wenn man die Übergänge für sicher hält."""
    S = grid.n_states
    pol = np.zeros(S, dtype=int)
    target_row = grid.rows - 2
    for s in range(S):
        r, c = grid.rc_of(s)
        if (r, c) == grid.goal:
            pol[s] = NORTH
        elif r < target_row:
            pol[s] = SOUTH
        elif r > target_row:
            pol[s] = NORTH
        else:
            pol[s] = EAST if c < grid.cols - 1 else SOUTH
    return pol


def build_model(grid):
    """Modell (P, R): P[s, a, s'] Übergangswahrscheinlichkeit, R[s, a] erwartete Belohnung (Erwartungswert über die drei möglichen Ausgänge der Aktion)."""
    S, A = grid.n_states, len(ACTIONS)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    goal_s = grid.state_of(grid.goal)
    start_s = grid.state_of(grid.start)
    cliff_states = {grid.state_of(rc) for rc in grid.cliff}
    for s in range(S):
        r, c = grid.rc_of(s)
        if s == goal_s:
            for a in ACTIONS:
                P[s, a, s] = 1.0
            continue
        for a in ACTIONS:
            outcomes = [(1.0 - grid.slip, a)] + [(grid.slip / 2.0, oa) for oa in _ORTHOGONAL[a]]
            for prob, direction in outcomes:
                dr, dc = _DELTA[direction]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                    nr, nc = r, c
                ns = grid.state_of((nr, nc))
                if ns == goal_s:
                    reward, target = C.GOAL_REWARD, goal_s
                elif ns in cliff_states:
                    reward, target = C.CLIFF_PENALTY, start_s
                else:
                    reward, target = C.STEP_COST, ns
                P[s, a, target] += prob
                R[s, a] += prob * reward
    return P, R
