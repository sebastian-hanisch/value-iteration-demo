"""Value Iteration und Policy Iteration - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück (Wurzel B) der Reinforcement-Learning-Linie der "Konzepte"-Reihe: ein Lagerroboter auf einem Raster mit bekanntem Modell (Rutsch-Wahrscheinlichkeit,
Rewards) - Value Iteration und Policy Iteration berechnen die optimale Policy exakt über die Bellman-Gleichung, ganz ohne Ausprobieren.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import vi_constants as C
import vi_grid as G
from vi_evaluation import Settings, analyse, gamma_experiment, slip_experiment, sweeps_experiment
from vi_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, sync_query_params
from vi_visualization import build_convergence, build_gamma, build_grid, build_slip, build_sweeps

st.set_page_config(page_title="Value Iteration und Policy Iteration – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _sweeps(sizes):
    return sweeps_experiment(sizes=sizes)


@st.cache_data(show_spinner=False)
def _slip(levels):
    return slip_experiment(levels=levels)


@st.cache_data(show_spinner=False)
def _gamma(levels):
    return gamma_experiment(levels=levels)


st.title("🗺️ Value Iteration und Policy Iteration")
st.markdown(
    """
Ein Lagerroboter steht auf einem Raster - Regale, ein schmaler Gang entlang einer Regalkante ("Klippe") und eine Packstation (Ziel). Anders als beim **Bandit** (Stück 1) gibt es jetzt einen **State**: der Roboter bewegt
sich, und mit einer **Rutsch-Wahrscheinlichkeit** landet er nicht immer dort, wo er hinwollte. Diesmal kennen wir das Modell vollständig - **Value Iteration** und **Policy Iteration** (Bellman 1957, Howard 1960) berechnen die
optimale Policy direkt aus der Bellman-Gleichung, ganz ohne einen einzigen Testlauf. Beide Verfahren finden dieselbe Lösung, aber auf unterschiedlichem Weg - und mit unterschiedlichem Aufwand. Alle Daten sind erzeugt;
die Rechnung ist in numpy geschrieben.
"""
)
st.caption(
    "Zweites Stück (Wurzel B) der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe: der Gegenpart zum Bandit - ein State statt keinem, ein bekanntes statt eines unbekannten Modells. **Bezug zu OR:** Anders als "
    "Kürzeste Wege oder die Dynamische Programmierung der Exakten-Suche-Linie sind die Übergänge hier **zufällig** - die Policy maximiert den erwarteten diskontierten Ertrag, nicht die billigste Route."
)

with st.expander("So funktionieren Value Iteration und Policy Iteration", expanded=True):
    st.markdown(
        r"""
1. **Das Modell.** Ein Raster mit Start, Ziel und einer Klippe entlang des kürzesten Wegs. Vier Richtungen; mit der Rutsch-Wahrscheinlichkeit $p$ geht die Bewegung stattdessen in eine der beiden Richtungen quer zur Absicht. Jeder Schritt kostet $-1$, die Klippe kostet $-100$ **und** setzt an den Start zurück, das Ziel gibt $+10$ und ist danach ein absorbierender State.
2. **Die Bellman-Gleichung.** Der Wert eines States ist der beste erreichbare erwartete, mit $\gamma$ diskontierte Ertrag: $V^*(s) = \max_a \sum_{s'} P(s'|s,a)\,\big(R(s,a,s') + \gamma\,V^*(s')\big)$.
3. **Value Iteration** wendet diese Gleichung immer wieder auf alle States gleichzeitig an, bis sich $V$ kaum noch ändert (ein *Sweep* = eine Runde über alle States).
4. **Policy Iteration** wechselt zwischen zwei Schritten: eine **feste** Policy bis zur Konvergenz bewerten (auch mehrere Sweeps), dann die Policy an der bewerteten $V$ **verbessern** (gierig). Das wiederholt sich, bis sich die Policy nicht mehr ändert - meist in **wenigen** äußeren Schritten, aber jeder davon kostet mehr.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Raster**")
    rows = st.slider("Zeilen", *bounds("rows_slider"), key="rows_slider", help="Höhe des Rasters (die unterste Zeile trägt die Klippe).")
    cols = st.slider("Spalten", *bounds("cols_slider"), key="cols_slider", help="Breite des Rasters (Start unten links, Ziel unten rechts).")
    st.markdown("**Die Übergänge**")
    slip = st.slider("Rutsch-Wahrscheinlichkeit", *bounds("slip_slider"), key="slip_slider", step=C.SLIP_STEP, format="%.2f", help="Wahrscheinlichkeit, quer zur Absicht zu rutschen (je Seite die Hälfte davon).")
    gamma = st.slider("Diskontfaktor γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.2f", help="Wie stark künftige Rewards gegenüber sofortigen abgewertet werden.")

sync_query_params({"rows_slider": int(rows), "cols_slider": int(cols), "slip_slider": round(float(slip), 3), "gamma_slider": round(float(gamma), 3)})

settings = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3))
with st.spinner("Value Iteration und Policy Iteration lösen das Modell exakt ..."):
    a = analyse(settings)
grid = a.grid
s0 = grid.state_of(grid.start)

# --- Wachsendes Beispiel -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Sweep für Sweep zur optimalen Policy")
if "vi_step" not in st.session_state or st.session_state.get("vi_step_owner") != settings:
    st.session_state["vi_step"] = a.sweeps_vi
    st.session_state["vi_step_owner"] = settings
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.slider("Sweep der Value Iteration", 0, a.sweeps_vi, key="vi_step", help="0 = V ist überall 0 (der Startwert).")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _render(s):
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        head = "V = 0 überall (Start)" if s == 0 else f"Nach Sweep {s} von {a.sweeps_vi}"
        show_policy = s == a.sweeps_vi
        c1.markdown(f"**{head}**" + (" - Policy erreicht" if show_policy else ""))
        c1.plotly_chart(build_grid(grid, a.hist_vi[s], a.policy_vi if show_policy else None), width="stretch", key=f"vi_grid_{s}")
        c2.markdown("**V(Start) über die Sweeps**")
        c2.plotly_chart(build_convergence(a.hist_vi[: s + 1], s0), width="stretch", key=f"vi_conv_{s}")


def _frames(n):
    return sorted({int(round(x)) for x in np.linspace(0, n, min(n + 1, 40))})


if auto_play:
    import time as _time
    for f in _frames(a.sweeps_vi):
        _render(f)
        _time.sleep(0.12)
else:
    _render(step)

st.markdown("---")

# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Finden beide Verfahren dieselbe Policy?")
mcols = st.columns(4)
mcols[0].metric("Value Iteration", f"{a.sweeps_vi} Sweeps", help="Zahl der Sweeps, bis sich V kaum noch ändert.")
mcols[1].metric("Policy Iteration", f"{a.outer_pi} äußere Schritte", help="Zahl der Policy-Verbesserungsschritte.")
mcols[2].metric("... davon Sweeps insgesamt", f"{a.outer_pi + a.eval_sweeps_pi}", delta=f"{a.eval_sweeps_pi} für die Policy Evaluation", delta_color="off", help="Jeder äußere Schritt bewertet die aktuelle Policy iterativ bis zur Konvergenz - das kostet die meisten Sweeps.")
mcols[3].metric("V(Start)", de(a.V_vi[s0], 2), help="Erwarteter, diskontierter Ertrag ab dem Start unter der optimalen Policy.")
match = bool(np.allclose(a.V_vi, a.V_pi, atol=1e-3))
reachable = [s for s in range(grid.n_states) if grid.rc_of(s) not in grid.cliff]
policy_match = bool(np.array_equal(a.policy_vi[reachable], a.policy_pi[reachable]))
if match and policy_match:
    st.success(f"✅ Beide Verfahren finden dieselbe Value Function (Abweichung < 0,001) und dieselbe Policy. Policy Iteration braucht dafür nur {a.outer_pi} äußere Schritte gegen {a.sweeps_vi} Sweeps der Value Iteration - **aber insgesamt {a.outer_pi + a.eval_sweeps_pi} Sweeps**, weil jede Policy Evaluation selbst viele Sweeps braucht.")
else:
    st.warning("⚠️ Die beiden Lösungen weichen voneinander ab - das sollte bei korrekter Rechnung nicht passieren (bitte prüfen).")
g1, g2 = st.columns(2)
with g1:
    st.markdown("##### Optimale Policy (Value Iteration)")
    st.plotly_chart(build_grid(grid, a.V_vi, a.policy_vi), width="stretch", key="vi_final_grid")
with g2:
    st.markdown("##### V(Start) über die Sweeps")
    st.plotly_chart(build_convergence(a.hist_vi, s0), width="stretch", key="vi_final_conv")
st.caption("Farbe: V(s), der erwartete diskontierte Ertrag ab dieser Zelle. Pfeile: die optimale Richtung. Klippenzellen (dunkel) werden unter der optimalen Policy nie besucht.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel Rechenaufwand braucht jedes Verfahren?")
st.caption(f"Vier Rastergrößen, Rutsch-Wahrscheinlichkeit {de(C.DEFAULT_SLIP,2)}, Diskontfaktor {de(C.DEFAULT_GAMMA,2)}. Gezeigt: Sweeps der Value Iteration, äußere Schritte der Policy Iteration, und deren Sweeps insgesamt (logarithmische Achse). Dauer wenige Sekunden.")
if st.button("Rastergrößen durchrechnen", key="sweeps_start"):
    st.session_state["sweeps_on"] = True
if st.session_state.get("sweeps_on"):
    rs = _sweeps(C.EXP_SIZES)
    st.plotly_chart(build_sweeps(rs), width="stretch", key="sweeps_chart")
    r0, r1 = rs["rows"][0], rs["rows"][-1]
    st.warning(
        f"**Befund:** Policy Iteration braucht bei jeder Größe deutlich weniger äußere Schritte als Value Iteration Sweeps ({r0['pi_outer']} gegen {r0['vi_sweeps']} beim {r0['rows']}×{r0['cols']}-Raster, {r1['pi_outer']} gegen {r1['vi_sweeps']} beim {r1['rows']}×{r1['cols']}-Raster) - "
        f"**aber insgesamt** (mit der Policy Evaluation mitgezählt) braucht sie ein Vielfaches an Sweeps ({r0['pi_total']} bzw. {r1['pi_total']}). Value Iteration bleibt über alle vier Rastergrößen fast gleich teuer (Sweeps wachsen kaum mit der Zahl der States) - "
        "Howards Vorteil gilt für die äußeren Schritte, nicht für die Gesamtarbeit."
    )

st.markdown("---")

st.subheader("🔬 Wie verändert Rutschen die optimale Policy?")
st.caption(f"Standardraster ({C.DEFAULT_ROWS}×{C.DEFAULT_COLS}), Rutsch-Wahrscheinlichkeit {', '.join(de(x,2) for x in C.EXP_SLIP_LEVELS)}. Gezeigt: wie viel teurer die naive Policy (immer der Reihe direkt über der Klippe entlang) gegenüber der optimalen ist, und wie viele Zellen dieser Reihe noch \"Osten\" statt \"Norden\" wählen. Dauer wenige Sekunden.")
if st.button("Rutschraten durchrechnen", key="slip_start"):
    st.session_state["slip_on"] = True
if st.session_state.get("slip_on"):
    rsl = _slip(C.EXP_SLIP_LEVELS)
    st.plotly_chart(build_slip(rsl), width="stretch", key="slip_chart")
    r0, r1 = rsl["rows"][0], rsl["rows"][-1]
    st.warning(
        f"**Befund:** Ohne Rutschen ({de(r0['slip'],2)}) ist die naive Policy (immer die Reihe direkt über der Klippe) bereits optimal (Mehrkosten {de(r0['gap'],2)}, {r0['east_cells']} von {r0['east_total']} Zellen \"Osten\"). Mit wachsendem Rutschen steigen die Mehrkosten der naiven Policy dramatisch "
        f"(auf {de(r1['gap'],2)} bei {de(r1['slip'],2)}), während die optimale Policy zunehmend weiter von der Klippe abrückt (nur noch {r1['east_cells']} von {r1['east_total']} Zellen bleiben direkt an der Klippe) - **die Policy kennt das Risiko, die naive Abkürzung nicht.**"
    )

st.markdown("---")

st.subheader("🔬 Ändert der Diskontfaktor die Policy oder nur die Geschwindigkeit?")
st.caption(f"Standardraster, Rutsch-Wahrscheinlichkeit {de(C.DEFAULT_SLIP,2)}, Diskontfaktor {', '.join(de(x,2) for x in C.EXP_GAMMA_LEVELS)}. Gezeigt: Sweeps beider Verfahren und wie viele States ihre optimale Action gegenüber der vorherigen Gamma-Stufe wechseln. Dauer wenige Sekunden.")
if st.button("Diskontfaktoren durchrechnen", key="gamma_start"):
    st.session_state["gamma_on"] = True
if st.session_state.get("gamma_on"):
    rg = _gamma(C.EXP_GAMMA_LEVELS)
    st.plotly_chart(build_gamma(rg), width="stretch", key="gamma_chart")
    total_diff = sum(r["policy_diff_vs_previous"] for r in rg["rows"])
    last = rg["rows"][-1]
    st.warning(
        f"**Befund:** Über alle {len(rg['rows'])} Diskontfaktor-Stufen ändert sich die optimale Policy insgesamt nur an {total_diff} von {grid.n_states} States - **der Diskontfaktor bestimmt hier fast nur die Geschwindigkeit, nicht das Ergebnis.** Policy Iteration reagiert dabei viel empfindlicher: ihre "
        f"Bewertungs-Sweeps wachsen von {rg['rows'][0]['pi_eval_sweeps']} bei γ={de(rg['rows'][0]['gamma'],2)} auf {last['pi_eval_sweeps']} bei γ={de(last['gamma'],2)} (die Policy Evaluation konvergiert mit der Rate γ selbst), während Value Iteration deutlich stabiler bleibt."
    )

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Modell ist vollständig bekannt** | Ohne bekannte Übergangswahrscheinlichkeiten und Rewards lässt sich die Bellman-Gleichung nicht direkt auswerten. | Q-Learning (Stück 3) |
| **Endlich viele States und Actions** | Ein sehr großes oder stetiges Raster macht dichte Tabellen für V und die Policy unhandlich. | Funktionsapproximation / DQN (Stück 6) |
| **Der Diskontfaktor macht die Bellman-Operation zu einer Kontraktion** | Ohne $\\gamma < 1$ (oder eine andere Garantie) muss Value Iteration nicht konvergieren. | - |
| **Policy Evaluation bis zur Konvergenz kostet nichts** | Gemessen: Policy Iteration braucht insgesamt ein Vielfaches der Sweeps von Value Iteration - "weniger äußere Schritte" bedeutet nicht "weniger Arbeit". | Policy Evaluation mit fester Schrittzahl statt bis zur Konvergenz |
| **Deterministisches, exakt bekanntes Raster** | Reale Lagerhallen ändern sich, Sensorrauschen ist selten exakt bekannt. | Q-Learning, Dyna-Q (Stück 3, 5) |
| **Eine einzige Klippenzone, eine einzige Policy** | Mehrere Gefahrenzonen oder mehrere Roboter gleichzeitig sind hier nicht abgebildet. | - |
"""
)
st.caption("Die Linie: Bandit (Stück 1) → **Value Iteration und Policy Iteration** (dieses Stück) → Q-Learning → SARSA / Dyna-Q / Funktionsapproximation (DQN) / Policy Gradient → Actor-Critic.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Modell.** States $s$ = Rasterzellen (die Packstation ist ein absorbierender State: $P(s|s,a)=1$, $R(s,a)=0$ für alle $a$). Action $a \in \{$Nord, Süd, Ost, West$\}$; mit Wahrscheinlichkeit $1-p$ die beabsichtigte Richtung, mit $p/2$ je eine der beiden dazu senkrechten. Ein Schritt an den Rand oder gegen eine Wand bleibt auf der Zelle stehen. Landet der Schritt auf der Packstation: Reward $+10$. Landet er auf der Klippe: Reward $-100$ **und** die nächste Zelle ist der Start (nicht die Klippe selbst). Sonst: Reward $-1$.

**Bellman-Optimalitätsgleichung.** $V^*(s) = \max_a Q^*(s,a)$, $Q^*(s,a) = \sum_{s'} P(s'|s,a)\big(R(s,a,s') + \gamma V^*(s')\big)$, Policy $\pi^*(s) = \arg\max_a Q^*(s,a)$.

**Value Iteration.** $V_0 = 0$; $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)(R(s,a,s') + \gamma V_k(s'))$ bis $\max_s |V_{k+1}(s) - V_k(s)| < 10^{-8}$; Policy aus dem letzten $V$.

**Policy Iteration** (Howard 1960). Start $\pi_0$ beliebig (hier: immer Norden). **Bewertung:** $V_{\pi}$ als Grenzwert von $V \leftarrow R_\pi + \gamma P_\pi V$ (iterativ, nicht als exaktes Gleichungssystem gelöst - so bleibt der Aufwand in derselben Einheit wie bei der Value Iteration vergleichbar: Sweeps). **Verbesserung:** $\pi'(s) = \arg\max_a Q_\pi(s,a)$. Wiederholen, bis $\pi' = \pi$.

Implementiert in `vi_grid.py` (das Vehikel, das Modell), `vi_solve.py` (Value Iteration und Policy Iteration), `vi_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Reinforcement Learning: Bandit bis Actor-Critic](https://sebastianhanisch.net/konzepte-reinforcement-learning.html)."
)
