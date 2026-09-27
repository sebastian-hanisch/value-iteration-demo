"""Konstanten der Demo "Value Iteration und Policy Iteration" (Stück 2, Wurzel B der Reinforcement-Learning-Linie): das Raster, feste Rewards, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

# --- Das Raster (Cliff-Walking-Vorlage, Sutton & Barto 2018) ---------------------------------------------------------------------------------------
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
GOAL_REWARD = 10.0

ROWS_MIN, ROWS_MAX, DEFAULT_ROWS = 3, 6, 4
COLS_MIN, COLS_MAX, DEFAULT_COLS = 4, 12, 8
SLIP_MIN, SLIP_MAX, SLIP_STEP, DEFAULT_SLIP = 0.0, 0.30, 0.02, 0.10
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.80, 0.99, 0.01, 0.95

TOL = 1e-8
MAX_ITER = 5000

# --- Experimente (feste Konfigurationen) -----------------------------------------------------------------------------------------------------------
EXP_SIZES = ((3, 4), (4, 8), (4, 12), (6, 12))
EXP_SLIP_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.30)
EXP_GAMMA_LEVELS = (0.80, 0.90, 0.95, 0.99)
