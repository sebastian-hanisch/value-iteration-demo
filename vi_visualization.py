"""Plotly-Abbildungen der Demo "Value Iteration und Policy Iteration". Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import vi_grid as G

CLIFF_COLOR = "#3a3a3a"
GOAL_COLOR = "#2e7d32"
START_COLOR = "#8c6bb1"
TEXT_LIGHT = "#ffffff"
TEXT_DARK = "#14233B"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_grid(grid, V, policy=None, title_values=True):
    """Raster als Heatmap über V(s), mit Pfeilen der Policy (falls gegeben); Klippe dunkel, Ziel grün, Start violett umrandet."""
    R, Cc = grid.rows, grid.cols
    z = np.full((R, Cc), np.nan)
    text = [["" for _ in range(Cc)] for _ in range(R)]
    for r in range(R):
        for c in range(Cc):
            s = grid.state_of((r, c))
            kind = grid.cell_kind((r, c))
            if kind == "cliff":
                z[r, c] = np.nan
            else:
                z[r, c] = V[s]
            if kind == "goal":
                text[r][c] = "Ziel"
            elif kind == "cliff":
                text[r][c] = ""
            elif policy is not None:
                text[r][c] = G.ACTION_ARROWS[policy[s]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, colorscale="RdYlGn", zmid=0, showscale=title_values, text=[[de(v, 1) if not np.isnan(v) else "" for v in row] for row in z],
                              hovertemplate="Zeile %{y}, Spalte %{x}: V=%{z:.2f}<extra></extra>", colorbar=dict(title="V(s)", thickness=14)))
    # Klippenzellen dunkel einfärben (über der Heatmap, da NaN durchsichtig bliebe)
    cliff_x = [c for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    cliff_y = [r for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    if cliff_x:
        fig.add_trace(go.Scatter(x=cliff_x, y=cliff_y, mode="markers", marker=dict(symbol="square", size=34, color=CLIFF_COLOR), showlegend=False, hovertemplate="Klippe<extra></extra>"))
    for r in range(R):
        for c in range(Cc):
            kind = grid.cell_kind((r, c))
            if text[r][c]:
                color = TEXT_LIGHT if kind == "goal" else TEXT_DARK
                fig.add_annotation(x=c, y=r, text=text[r][c], showarrow=False, font=dict(size=18, color=color))
    sr, sc = grid.start
    fig.add_shape(type="rect", x0=sc - 0.45, x1=sc + 0.45, y0=sr - 0.45, y1=sr + 0.45, line=dict(color=START_COLOR, width=3))
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 90 * grid.rows + 60)


def build_convergence(hist_vi, start_state):
    """V(Start) über die Sweeps der Value Iteration."""
    y = hist_vi[:, start_state]
    x = np.arange(len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#1f77b4", width=2)))
    fig.update_xaxes(title_text="Sweep")
    fig.update_yaxes(title_text="V(Start)")
    return _base(fig, 260)


def build_sweeps(exp):
    labels = [f"{r['rows']}×{r['cols']}" for r in exp["rows"]]
    vi = [r["vi_sweeps"] for r in exp["rows"]]
    pi_outer = [r["pi_outer"] for r in exp["rows"]]
    pi_total = [r["pi_total"] for r in exp["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=vi, name="Value Iteration (Sweeps)", marker=dict(color="#1f77b4")))
    fig.add_trace(go.Bar(x=labels, y=pi_outer, name="Policy Iteration (äußere Schritte)", marker=dict(color="#8c6bb1")))
    fig.add_trace(go.Bar(x=labels, y=pi_total, name="Policy Iteration (Sweeps insgesamt)", marker=dict(color="#d62728")))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Sweeps", type="log")
    fig.update_xaxes(title_text="Raster")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_slip(exp):
    levels = [r["slip"] for r in exp["rows"]]
    gap = [r["gap"] for r in exp["rows"]]
    east = [100 * r["east_cells"] / r["east_total"] for r in exp["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[de(x, 2) for x in levels], y=gap, name="Mehrkosten der naiven Policy", marker=dict(color="#d62728"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=[de(x, 2) for x in levels], y=east, name="Anteil \"an der Klippe entlang\"", mode="lines+markers", line=dict(color="#1f77b4", width=2), marker=dict(size=7), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="V*(Start) − V_naiv(Start)", rangemode="tozero"), yaxis2=dict(title="Anteil Zellen \"Osten\" (%)", overlaying="y", side="right", range=[0, 100], showgrid=False))
    fig.update_xaxes(title_text="Rutsch-Wahrscheinlichkeit", type="category")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_gamma(exp):
    levels = [r["gamma"] for r in exp["rows"]]
    vi = [r["vi_sweeps"] for r in exp["rows"]]
    pi = [r["pi_eval_sweeps"] for r in exp["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[de(x, 2) for x in levels], y=vi, name="Value Iteration (Sweeps)", mode="lines+markers", line=dict(color="#1f77b4", width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=[de(x, 2) for x in levels], y=pi, name="Policy Iteration (Bewertungs-Sweeps)", mode="lines+markers", line=dict(color="#d62728", width=2), marker=dict(size=7)))
    fig.update_xaxes(title_text="Diskontfaktor γ", type="category")
    fig.update_yaxes(title_text="Sweeps", type="log")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))
