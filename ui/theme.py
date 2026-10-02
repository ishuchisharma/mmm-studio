"""Visual system: palette, type, Plotly template, number formatting, shared HTML pieces."""
from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK = "#16213A"
MUTED = "#5A6478"
PAPER = "#F3F5F8"
SURFACE = "#FFFFFF"
LINE = "#DCE1E8"
PRIMARY = "#1F4E79"
GAIN = "#2F7D4F"
LOSS = "#B5452F"

CHANNEL_COLORS = ["#1F4E79", "#9A5B7C", "#C98B2B", "#3F9A8A", "#6574C4", "#A0522D", "#5C6B73", "#B8A23A"]


def channel_colors(channels) -> dict[str, str]:
    return {ch: CHANNEL_COLORS[i % len(CHANNEL_COLORS)] for i, ch in enumerate(channels)}


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,400..800&family=Source+Sans+3:wght@400;500;600;700&display=swap');

h1, h2, h3, h4, .hero-line {{
  font-family: 'Archivo', 'Arial Narrow', system-ui, sans-serif !important;
  font-stretch: 87.5%;
  color: {INK};
  letter-spacing: -0.01em;
}}
h1 {{ font-weight: 750 !important; font-size: 2.1rem !important; line-height: 1.12 !important; }}
h2 {{ font-weight: 700 !important; font-size: 1.45rem !important; }}
h3 {{ font-weight: 650 !important; font-size: 1.15rem !important; }}
[data-testid="stMetricValue"] {{
  font-family: 'Archivo', system-ui, sans-serif; font-stretch: 87.5%;
  font-variant-numeric: tabular-nums; font-weight: 650;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.9rem; }}
[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 3rem; max-width: 1180px; }}

.lede {{ color: {MUTED}; font-size: 1.08rem; line-height: 1.55; max-width: 68ch; margin: -0.3rem 0 1.4rem; }}

.hero-line {{
  font-size: clamp(1.6rem, 2.9vw, 2.35rem); font-weight: 760; line-height: 1.14;
  max-width: 32ch; margin: 0.4rem 0 1.1rem;
}}
.hero-line .num {{ font-variant-numeric: tabular-nums; }}
.hero-sub {{ color: {MUTED}; font-size: 1.05rem; max-width: 64ch; line-height: 1.55; margin-bottom: 1.6rem; }}

/* the budget strip: one bar per allocation, segments sized by share of spend */
.strip {{ margin: 0.2rem 0 1.4rem; }}
.strip-row {{ display: grid; grid-template-columns: 9.5rem 1fr 6.5rem; gap: 0.9rem; align-items: center; margin-bottom: 0.55rem; }}
.strip-label {{ font-weight: 600; color: {INK}; font-size: 0.95rem; }}
.strip-total {{ text-align: right; font-variant-numeric: tabular-nums; color: {MUTED}; font-size: 0.95rem; }}
.strip-bar {{ display: flex; height: 2.35rem; border-radius: 3px; overflow: hidden; background: {LINE}; }}
.strip-seg {{
  display: flex; align-items: center; padding: 0 0.5rem; color: #fff; font-size: 0.82rem;
  font-weight: 600; white-space: nowrap; overflow: hidden; min-width: 0;
  transition: flex-grow 600ms cubic-bezier(.2,.7,.2,1);
}}
.strip-seg + .strip-seg {{ border-left: 2px solid {SURFACE}; }}
.strip-legend {{ display: flex; flex-wrap: wrap; gap: 0.35rem 1.1rem; margin-top: 0.3rem; color: {MUTED}; font-size: 0.88rem; }}
.strip-legend span::before {{ content: ""; display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 2px; margin-right: 0.4rem; vertical-align: -1px; background: var(--c); }}
@media (max-width: 640px) {{
  .strip-row {{ grid-template-columns: 1fr; gap: 0.3rem; }}
  .strip-total {{ text-align: left; }}
}}
@media (prefers-reduced-motion: reduce) {{ .strip-seg {{ transition: none; }} }}

.note {{ border-left: 3px solid {PRIMARY}; padding: 0.55rem 0.9rem; background: {SURFACE};
        color: {INK}; margin: 0.4rem 0 1rem; border-radius: 0 3px 3px 0; line-height: 1.5; }}
.note.warn {{ border-left-color: {LOSS}; }}
.findings {{ margin: 0.3rem 0 1.2rem; padding-left: 1.1rem; line-height: 1.6; max-width: 76ch; }}
.findings li {{ margin-bottom: 0.35rem; }}
.gain {{ color: {GAIN}; font-weight: 600; }}
.loss {{ color: {LOSS}; font-weight: 600; }}
a:focus-visible, button:focus-visible {{ outline: 2px solid {PRIMARY}; outline-offset: 2px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def register_plotly_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Source Sans Pro, Source Sans 3, sans-serif", size=13, color=INK),
        title=dict(font=dict(family="Archivo, sans-serif", size=16, color=INK), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        colorway=CHANNEL_COLORS,
        xaxis=dict(gridcolor="#EEF1F5", linecolor=LINE, zeroline=False, ticks="outside", tickcolor=LINE),
        yaxis=dict(gridcolor="#EEF1F5", linecolor=LINE, zeroline=False),
        legend=dict(orientation="h", y=-0.18, x=0, title=None),
        margin=dict(l=10, r=10, t=48, b=10),
        hoverlabel=dict(font=dict(family="Source Sans 3, sans-serif")),
    )
    pio.templates["mmm"] = t
    pio.templates.default = "mmm"


# ------------------------------------------------------------ number formatting


def money(x: float, cur: str | None = None, style: str | None = None, signed: bool = False) -> str:
    cur = cur if cur is not None else st.session_state.get("cfg").currency
    style = style or st.session_state.get("number_style", "K/M")
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    sign = "−" if x < 0 else ("+" if signed and x > 0 else "")
    a = abs(x)
    if style == "Lakh/Crore":
        if a >= 1e7:
            s = f"{a/1e7:.2f} Cr"
        elif a >= 1e5:
            s = f"{a/1e5:.2f} L"
        elif a >= 1e3:
            s = f"{a/1e3:.1f}K"
        else:
            s = f"{a:,.0f}"
    else:
        if a >= 1e9:
            s = f"{a/1e9:.2f}B"
        elif a >= 1e6:
            s = f"{a/1e6:.2f}M"
        elif a >= 1e3:
            s = f"{a/1e3:.1f}K"
        else:
            s = f"{a:,.0f}"
    return f"{sign}{cur}{s}"


def pct(x: float, signed: bool = False, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{abs(x)*100:.{digits}f}%"
    if x < 0:
        return "−" + s
    return ("+" if signed and x > 0 else "") + s


def ratio(x: float) -> str:
    cur = st.session_state.get("cfg").currency
    if x is None or np.isnan(x):
        return "–"
    return f"{cur}{x:.2f}"


# ------------------------------------------------------------ components


def lede(text: str):
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def note(text: str, warn: bool = False):
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def findings(items: list[str]):
    st.markdown('<ul class="findings">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>", unsafe_allow_html=True)


def budget_strip(rows: list[tuple[str, "pd.Series"]], colors: dict[str, str]):
    """Stacked share bars, one row per allocation, all on the same channel order.

    Each row's bar spans the full width so shares are comparable; the total is printed at the end.
    """
    parts = ['<div class="strip">']
    for label, alloc in rows:
        total = float(alloc.sum())
        parts.append('<div class="strip-row">')
        parts.append(f'<div class="strip-label">{html.escape(label)}</div><div class="strip-bar">')
        for ch, v in alloc.items():
            share = v / total if total > 0 else 0
            text = (f"{html.escape(ch)} {share*100:.0f}%" if share >= 0.15
                    else f"{share*100:.0f}%" if share >= 0.05 else "")
            parts.append(
                f'<div class="strip-seg" style="flex-grow:{max(share, 0):.5f};flex-basis:0;background:{colors[ch]}" '
                f'title="{html.escape(ch)}: {share*100:.1f}%">{text}</div>'
            )
        parts.append(f'</div><div class="strip-total">{money(total)}/wk</div></div>')
    parts.append('<div class="strip-legend">')
    for ch in rows[0][1].index:
        parts.append(f'<span style="--c:{colors[ch]}">{html.escape(ch)}</span>')
    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
