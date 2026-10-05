# 🗺️ Value Iteration und Policy Iteration

**[→ Demo live ausprobieren](https://sebastianhanisch-value-iteration-demo.streamlit.app/)**

Zweites Stück (Wurzel B) der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Ein Lagerroboter bewegt sich auf einem Raster mit einer Klippe entlang des kürzesten Wegs; anders als beim [Bandit](https://github.com/sebastian-hanisch/bandit-demo) (Stück 1) gibt es jetzt einen State – und das Modell ist **vollständig bekannt**. **Value Iteration** und **Policy Iteration** (Bellman 1957, Howard 1960) berechnen die optimale Policy direkt aus der Bellman-Gleichung, ganz ohne einen einzigen Testlauf.

## Kernfrage

**Finden beide Verfahren dieselbe Policy – und zu welchem Preis?** Policy Iteration braucht deutlich weniger äußere Verbesserungsschritte als Value Iteration Sweeps – heißt das auch weniger Gesamtarbeit? Und wie verändert **Unsicherheit** (eine Rutsch-Wahrscheinlichkeit) eine Policy, die exakt berechnet, nicht gelernt wird?

## Modell

- **Vehikel** (`vi_grid.py`): die Cliff-Walking-Vorlage aus Sutton & Barto (2018, Beispiel 6.6). Start unten links, Packstation (Ziel) unten rechts, dazwischen in der untersten Zeile eine Klippe. Vier Richtungen; mit der **Rutsch-Wahrscheinlichkeit** $p$ rutscht die Bewegung mit gleicher Wahrscheinlichkeit auf eine der beiden Richtungen quer zur Absicht. Ein Schritt an den Rand hält auf der Stelle. Jeder Schritt kostet **−1**, die Klippe kostet **−100** und setzt an den Start zurück, das Ziel gibt **+10** und ist danach ein absorbierender State (Zahlen wörtlich aus der Vorlage übernommen).
- **Bellman-Optimalitätsgleichung** (`vi_solve.py`): $V^*(s) = \max_a \sum_{s'} P(s'|s,a)(R(s,a,s') + \gamma V^*(s'))$.
- **Value Iteration:** wiederholte Bellman-Backups über alle States (ein *Sweep* = eine Runde), bis $V$ sich kaum noch ändert.
- **Policy Iteration** (Howard 1960): abwechselnd eine feste Policy **bewerten** (iterativ bis zur Konvergenz – nicht als exaktes Gleichungssystem gelöst, damit der Aufwand in derselben Einheit wie Value Iteration vergleichbar bleibt) und **verbessern** (gierig), bis sich die Policy nicht mehr ändert.

## Methodik

Beide Verfahren laufen auf demselben Modell; verglichen werden die resultierende Value Function und Policy sowie der Aufwand in **Sweeps** (eine vollständige Runde über alle States). Drei Experimente: der Aufwand über vier Rastergrößen, die Wirkung der Rutsch-Wahrscheinlichkeit auf die optimale Route (gegen eine naive "immer den kürzesten Weg"-Policy), und ob der Diskontfaktor die Policy selbst oder nur die Konvergenzgeschwindigkeit verändert. Das Modell ist deterministisch erzeugt – jede Zahl ist exakt, keine Seeds, keine Streuung.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Finden beide Verfahren dieselbe Lösung?** (4×8-Raster, Rutschen 0,10) | Ja: Value Functions stimmen bis auf < 0,001 überein, die Policy ist auf allen erreichbaren States identisch. | `test_value_and_policy_iteration_agree_on_the_optimum` |
| **Braucht Policy Iteration weniger Aufwand?** (vier Rastergrößen) | Policy Iteration braucht bei jeder Größe deutlich weniger **äußere Schritte** (5-8) als Value Iteration **Sweeps** (38-49) – aber **insgesamt** (mit der Policy Evaluation mitgezählt) ein Vielfaches an Sweeps (1192-1963). Value Iteration bleibt über alle vier Rastergrößen (12 bis 72 States) fast gleich teuer. | `test_sweeps_by_size` |
| **Wie verändert Rutschen die Policy?** (4×8-Raster, Rutschen 0 bis 0,30) | Ohne Rutschen ist die naive Policy (immer die Reihe direkt über der Klippe) bereits optimal (Mehrkosten 0,00, 7 von 7 Zellen "Osten"). Bei Rutschen 0,30 kosten die naiven Mehrkosten **141,79** (gegenüber 0,00 anfangs), während die optimale Policy ganz von der Klippe abrückt (0 von 7 Zellen bleiben "Osten"). | `test_slip_moves_the_policy_away_from_the_cliff` |
| **Ändert der Diskontfaktor die Policy?** (γ = 0,80 bis 0,99) | Über alle vier Stufen ändert sich die optimale Policy nur an **1 von 32** States – der Diskontfaktor bestimmt fast nur die Geschwindigkeit. Policy Iterations Bewertungs-Sweeps wachsen dabei stark (458 bei γ=0,80 auf 5974 bei γ=0,99), Value Iteration bleibt stabiler (68 bis 47 Sweeps). | `test_gamma_barely_changes_the_policy_but_changes_convergence_speed` |
| Standardfall (Preset, 4×8, Rutschen 0,10, γ=0,95) | V(Start) −9,23; Value Iteration 44 Sweeps, Policy Iteration 5 äußere Schritte, 1314 Bewertungs-Sweeps (1319 insgesamt). | `test_standard_preset` |
| Kein Rutschen (Preset, deterministisch) | V(Start) −0,10; Policy Iteration braucht hier ungewöhnlich **viele** äußere Schritte (11) – viele gleich gute Wege ohne Risiko erzeugen Gleichstände, die erst nach und nach aufgelöst werden. | `test_no_slip_preset` |
| Starkes Rutschen (Preset, 0,30) | V(Start) −10,12 – kaum schlechter als bei 0,10 (−9,23), die Policy hat sich da schon ganz zurückgezogen; Value Iteration braucht dafür mehr Sweeps (102). | `test_strong_slip_preset` |

Die Preset-Zeilen sind **Einzelläufe** auf dem jeweils exakten Modell – hier gibt es keine Seeds, jede Zahl ist deterministisch.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Modell ist vollständig bekannt** | Ohne bekannte Übergangswahrscheinlichkeiten und Rewards lässt sich die Bellman-Gleichung nicht direkt auswerten. | Q-Learning (Stück 3) |
| **Endlich viele States und Actions** | Ein sehr großes oder stetiges Raster macht dichte Tabellen unhandlich. | Funktionsapproximation / DQN (Stück 6) |
| **Policy Evaluation bis zur Konvergenz kostet nichts** | Gemessen: Policy Iteration braucht insgesamt ein Vielfaches der Sweeps von Value Iteration – "weniger äußere Schritte" heißt nicht "weniger Arbeit". | Policy Evaluation mit fester Schrittzahl statt bis zur Konvergenz |
| **Der Diskontfaktor macht die Bellman-Operation zu einer Kontraktion** | Ohne $\gamma<1$ (oder eine andere Garantie) muss Value Iteration nicht konvergieren. | – |
| **Deterministisches, exakt bekanntes Raster** | Reale Lagerhallen ändern sich, Sensorrauschen ist selten exakt bekannt. | Q-Learning, Dyna-Q (Stück 3, 5) |

## Tests

`tests/` prüft das Vehikel (`vi_grid.py`: States, Zellenarten, jede Übergangswahrscheinlichkeit und erwarteter Reward von Hand nachgerechnet – Rand-Abprall, Rutschen, Klippen-Rücksetzung, Ziel-Reward), die Lösungsverfahren (`vi_solve.py`: die Bellman-Formel von Hand, ein entarteter Ein-Zeilen-Fall mit geschlossener Lösung [$V^* = -1/(1-\gamma)$, weil der einzige Weg zwangsläufig durch die Klippe führt], Value Iteration gegen Policy Iteration gegeneinander UND beide gegen Vollaufzählung auf einem winzigen Raster), die Auswertung und drei Experimente, die Presets und Permalinks, die App (AppTest: Standard, jedes Preset, Sweep-Slider, Permalink-Klemmen/-Einrasten, Extremwerte, drei Experimente auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`). Alle Zahlen sind exakt (kein Zufall im Modell); Laufzeit unter einer Minute. Die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Sweep-für-Sweep-Ansicht, Kernfrage, drei Experimente auf Abruf, Grenzen, Formeln |
| `vi_constants.py` | Regler-Grenzen, feste Rewards, Experimentkonstanten |
| `vi_grid.py` | Das Vehikel: Raster, Zellenarten, Übergangsmodell, naive Vergleichspolicy |
| `vi_solve.py` | Value Iteration und Policy Iteration |
| `vi_evaluation.py` | Analyse, drei Experimente |
| `vi_visualization.py` | Plotly-Abbildungen (Raster als Heatmap mit Policy-Pfeilen, alle Achsen gesperrt) |
| `vi_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Unbekanntes Modell** – das ist der nächste Schritt dieser Linie (Q-Learning, Stück 3).
- **Mehrere Klippenzonen, mehrere Roboter gleichzeitig, stetiger State-Raum.**
- **Exakte Policy Evaluation als lineares Gleichungssystem** – bewusst iterativ gelassen, um den Aufwand mit Value Iteration in derselben Einheit (Sweeps) vergleichbar zu halten.

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Reinforcement Learning: Bandit bis Actor-Critic](https://sebastianhanisch.net/konzepte-reinforcement-learning.html).
