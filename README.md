# 🗺️ Wert- und Politikiteration

Zweites Stück (Wurzel B) der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Ein Lagerroboter bewegt sich auf einem Raster mit einer Klippe entlang des kürzesten Wegs; anders als beim [Bandit](https://github.com/sebastian-hanisch/bandit-demo) (Stück 1) gibt es jetzt einen Zustand – und das Modell ist **vollständig bekannt**. **Wert-** und **Politikiteration** (Bellman 1957, Howard 1960) berechnen die optimale Politik direkt aus der Bellman-Gleichung, ganz ohne einen einzigen Testlauf.

## Kernfrage

**Finden beide Verfahren dieselbe Politik – und zu welchem Preis?** Politikiteration braucht deutlich weniger äußere Verbesserungsschritte als Wertiteration Sweeps – heißt das auch weniger Gesamtarbeit? Und wie verändert **Unsicherheit** (eine Rutsch-Wahrscheinlichkeit) eine Politik, die exakt berechnet, nicht gelernt wird?

## Modell

- **Vehikel** (`vi_grid.py`): die Cliff-Walking-Vorlage aus Sutton & Barto (2018, Beispiel 6.6). Start unten links, Packstation (Ziel) unten rechts, dazwischen in der untersten Zeile eine Klippe. Vier Richtungen; mit der **Rutsch-Wahrscheinlichkeit** $p$ rutscht die Bewegung mit gleicher Wahrscheinlichkeit auf eine der beiden Richtungen quer zur Absicht. Ein Schritt an den Rand hält auf der Stelle. Jeder Schritt kostet **−1**, die Klippe kostet **−100** und setzt an den Start zurück, das Ziel gibt **+10** und ist danach ein Ruhezustand (Zahlen wörtlich aus der Vorlage übernommen).
- **Bellman-Optimalitätsgleichung** (`vi_solve.py`): $V^*(s) = \max_a \sum_{s'} P(s'|s,a)(R(s,a,s') + \gamma V^*(s'))$.
- **Wertiteration:** wiederholte Bellman-Backups über alle Zustände (ein *Sweep* = eine Runde), bis $V$ sich kaum noch ändert.
- **Politikiteration** (Howard 1960): abwechselnd eine feste Politik **bewerten** (iterativ bis zur Konvergenz – nicht als exaktes Gleichungssystem gelöst, damit der Aufwand in derselben Einheit wie Wertiteration vergleichbar bleibt) und **verbessern** (gierig), bis sich die Politik nicht mehr ändert.

## Methodik

Beide Verfahren laufen auf demselben Modell; verglichen werden die resultierende Wertfunktion und Politik sowie der Aufwand in **Sweeps** (eine vollständige Runde über alle Zustände). Drei Experimente: der Aufwand über vier Rastergrößen, die Wirkung der Rutsch-Wahrscheinlichkeit auf die optimale Route (gegen eine naive "immer den kürzesten Weg"-Politik), und ob der Diskontfaktor die Politik selbst oder nur die Konvergenzgeschwindigkeit verändert. Das Modell ist deterministisch erzeugt – jede Zahl ist exakt, keine Seeds, keine Streuung.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Finden beide Verfahren dieselbe Lösung?** (4×8-Raster, Rutschen 0,10) | Ja: Wertfunktionen stimmen bis auf < 0,001 überein, die Politik ist auf allen erreichbaren Zuständen identisch. | `test_value_and_policy_iteration_agree_on_the_optimum` |
| **Braucht Politikiteration weniger Aufwand?** (vier Rastergrößen) | Politikiteration braucht bei jeder Größe deutlich weniger **äußere Schritte** (4-5) als Wertiteration **Sweeps** (38-49) – aber **insgesamt** (mit der Politikbewertung mitgezählt) ein Vielfaches an Sweeps (936-1163). Wertiteration bleibt über alle vier Rastergrößen (12 bis 72 Zustände) fast gleich teuer. | `test_sweeps_by_size` |
| **Wie verändert Rutschen die Politik?** (4×8-Raster, Rutschen 0 bis 0,30) | Ohne Rutschen ist die naive Politik (immer die Reihe direkt über der Klippe) bereits optimal (Mehrkosten 0,00, 7 von 7 Zellen "Osten"). Bei Rutschen 0,30 kosten die naiven Mehrkosten **141,79** (gegenüber 0,00 anfangs), während die optimale Politik ganz von der Klippe abrückt (0 von 7 Zellen bleiben "Osten"). | `test_slip_moves_the_policy_away_from_the_cliff` |
| **Ändert der Diskontfaktor die Politik?** (γ = 0,80 bis 0,99) | Über alle vier Stufen ändert sich die optimale Politik nur an **1 von 32** Zuständen – der Diskontfaktor bestimmt fast nur die Geschwindigkeit. Politikiterations Bewertungs-Sweeps wachsen dabei stark (389 bei γ=0,80 auf 4029 bei γ=0,99), Wertiteration bleibt stabiler (68 bis 47 Sweeps). | `test_gamma_barely_changes_the_policy_but_changes_convergence_speed` |
| Standardfall (Preset, 4×8, Rutschen 0,10, γ=0,95) | V(Start) −9,23; Wertiteration 44 Sweeps, Politikiteration 5 äußere Schritte, 1158 Bewertungs-Sweeps (1163 insgesamt). | `test_standard_preset` |
| Kein Rutschen (Preset, deterministisch) | V(Start) −0,10; Politikiteration braucht hier ungewöhnlich **viele** äußere Schritte (11) – viele gleich gute Wege ohne Risiko erzeugen Gleichstände, die erst nach und nach aufgelöst werden. | `test_no_slip_preset` |
| Starkes Rutschen (Preset, 0,30) | V(Start) −10,12 – kaum schlechter als bei 0,10 (−9,23), die Politik hat sich da schon ganz zurückgezogen; Wertiteration braucht dafür mehr Sweeps (102). | `test_strong_slip_preset` |

Die Preset-Zeilen sind **Einzelläufe** auf dem jeweils exakten Modell – hier gibt es keine Seeds, jede Zahl ist deterministisch.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Modell ist vollständig bekannt** | Ohne bekannte Übergangswahrscheinlichkeiten und Belohnungen lässt sich die Bellman-Gleichung nicht direkt auswerten. | Q-Learning (Stück 3) |
| **Endlich viele Zustände und Aktionen** | Ein sehr großes oder stetiges Raster macht dichte Tabellen unhandlich. | Funktionsapproximation / DQN (Stück 6) |
| **Politikbewertung bis zur Konvergenz kostet nichts** | Gemessen: Politikiteration braucht insgesamt ein Vielfaches der Sweeps von Wertiteration – "weniger äußere Schritte" heißt nicht "weniger Arbeit". | Politikbewertung mit fester Schrittzahl statt bis zur Konvergenz |
| **Der Diskontfaktor macht die Bellman-Operation zu einer Kontraktion** | Ohne $\gamma<1$ (oder eine andere Garantie) muss Wertiteration nicht konvergieren. | – |
| **Deterministisches, exakt bekanntes Raster** | Reale Lagerhallen ändern sich, Sensorrauschen ist selten exakt bekannt. | Q-Learning, Dyna-Q (Stück 3, 5) |

## Tests

`tests/` prüft das Vehikel (`vi_grid.py`: Zustände, Zellenarten, jede Übergangswahrscheinlichkeit und erwartete Belohnung von Hand nachgerechnet – Rand-Abprall, Rutschen, Klippen-Rücksetzung, Zielbelohnung), die Lösungsverfahren (`vi_solve.py`: die Bellman-Formel von Hand, ein entarteter Ein-Zeilen-Fall mit geschlossener Lösung [$V^* = -1/(1-\gamma)$, weil der einzige Weg zwangsläufig durch die Klippe führt], Wert- gegen Politikiteration gegeneinander UND beide gegen Vollaufzählung auf einem winzigen Raster), die Auswertung und drei Experimente, die Presets und Permalinks, die App (AppTest: Standard, jedes Preset, Sweep-Slider, Permalink-Klemmen/-Einrasten, Extremwerte, drei Experimente auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`). Alle Zahlen sind exakt (kein Zufall im Modell); Laufzeit unter einer Minute. Die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Sweep-für-Sweep-Ansicht, Kernfrage, drei Experimente auf Abruf, Grenzen, Formeln |
| `vi_constants.py` | Regler-Grenzen, feste Belohnungen, Experimentkonstanten |
| `vi_grid.py` | Das Vehikel: Raster, Zellenarten, Übergangsmodell, naive Vergleichspolitik |
| `vi_solve.py` | Wert- und Politikiteration |
| `vi_evaluation.py` | Analyse, drei Experimente |
| `vi_visualization.py` | Plotly-Abbildungen (Raster als Heatmap mit Politik-Pfeilen, alle Achsen gesperrt) |
| `vi_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Unbekanntes Modell** – das ist der nächste Schritt dieser Linie (Q-Learning, Stück 3).
- **Mehrere Klippenzonen, mehrere Roboter gleichzeitig, stetiger Zustandsraum.**
- **Exakte Politikbewertung als lineares Gleichungssystem** – bewusst iterativ gelassen, um den Aufwand mit Wertiteration in derselben Einheit (Sweeps) vergleichbar zu halten.

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.
