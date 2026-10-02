import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm.transforms import decay_weights, half_life
from ui import state
from ui.theme import INK, MUTED, channel_colors, findings, lede, money, pct, ratio

res, ens, prep = state.get_models()
cfg = st.session_state.cfg
colors = channel_colors(res.channels)
ct = ens.channel_table()
cur = cfg.currency
n_models = len(ens.models)

st.title("Channel returns")
lede("Two numbers matter for each channel: what it returned on average, and what the next "
     f"{cur}1 would return at today's spend. Budget decisions should follow the second one.")

# ------------------------------------------------------------------ ROI vs mROI
order = ct.sort_values("mroi", ascending=False).index.tolist()
fig = go.Figure()
fig.add_bar(x=order, y=ct.roi[order], name=f"Average return per {cur}1",
            marker_color=[colors[c] for c in order], opacity=0.4,
            error_y=dict(type="data", symmetric=False, color=MUTED, thickness=1.2,
                         array=(ct.roi_hi - ct.roi).clip(lower=0)[order],
                         arrayminus=(ct.roi - ct.roi_lo).clip(lower=0)[order]))
fig.add_bar(x=order, y=ct.mroi[order], name=f"Return on the next {cur}1",
            marker_color=[colors[c] for c in order],
            error_y=dict(type="data", symmetric=False, color=INK, thickness=1.2,
                         array=(ct.mroi_hi - ct.mroi).clip(lower=0)[order],
                         arrayminus=(ct.mroi - ct.mroi_lo).clip(lower=0)[order]))
fig.add_hline(y=1, line_dash="dot", line_color=INK, annotation_text="Break-even", annotation_position="top right")
fig.update_layout(barmode="group", height=420, yaxis_title=f"Revenue per {cur}1",
                  title=f"Average return versus return on the next {cur}1")
st.plotly_chart(fig, width="stretch")
st.caption(f"Bars are the average across {n_models} near-best models; whiskers span the 10th to 90th "
           "percentile of those models. Wide whiskers mean the data cannot pin the channel down.")

best, worst = order[0], order[-1]
saturated = (ct.mroi / ct.roi).idxmin()
items = [
    f"<b>{best}</b> has the most headroom: the next {cur}1 returns {ratio(ct.loc[best, 'mroi'])} "
    f"(models range from {ratio(ct.loc[best, 'mroi_lo'])} to {ratio(ct.loc[best, 'mroi_hi'])}).",
    f"<b>{worst}</b> has the least: the next {cur}1 returns {ratio(ct.loc[worst, 'mroi'])}."
    + (" That is below break-even." if ct.loc[worst, "mroi"] < 1 else ""),
    f"<b>{saturated}</b> is the most saturated relative to its own average: its return on the next unit of "
    f"spend is {pct(ct.loc[saturated, 'mroi'] / ct.loc[saturated, 'roi'])} of its average return.",
]
wide = ct[(ct.mroi_hi - ct.mroi_lo) > ct.mroi].index.tolist()
if wide:
    items.append(f"Treat {', '.join(wide)} with caution. Plausible models disagree by more than the estimate "
                 "itself, usually because spend was small or moved together with another channel.")
findings(items)

# ------------------------------------------------------------------ share of spend vs share of effect
st.subheader("Share of spend versus share of revenue driven")
fig = go.Figure()
fig.add_bar(y=ct.index, x=ct.spend_share, orientation="h", name="Share of media spend", marker_color="#B9C2D0")
fig.add_bar(y=ct.index, x=ct.effect_share, orientation="h", name="Share of media-driven revenue",
            marker_color=[colors[c] for c in ct.index])
fig.update_layout(barmode="group", height=360, xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("A channel whose coloured bar is longer than its grey bar earns more than its share of the budget.")

# ------------------------------------------------------------------ response curves
st.subheader("Response curves")
lede("Each curve shows weekly revenue if spend were held at that level week after week. The bold line is "
     "the average; faint lines are the individual models behind it. The dot is today.")
ref = ens.reference_spend()
cols = st.columns(min(3, len(res.channels)))
for i, ch in enumerate(res.channels):
    hi = max(ref[ch] * 3, res.spend[ch].max() * 1.1, 1)
    xs = np.linspace(0, hi, 160)
    f = go.Figure()
    for m in ens.models:
        f.add_scatter(x=xs, y=m.weekly_response(ch, xs), line=dict(color=colors[ch], width=0.6),
                      opacity=0.18, showlegend=False, hoverinfo="skip")
    f.add_scatter(x=xs, y=ens.weekly_response(ch, xs), line=dict(color=colors[ch], width=2.8), showlegend=False,
                  hovertemplate="Spend %{x:,.0f}<br>Revenue %{y:,.0f}<extra></extra>")
    f.add_scatter(x=[ref[ch]], y=[float(ens.weekly_response(ch, ref[ch]))], mode="markers", showlegend=False,
                  marker=dict(size=11, color=colors[ch], line=dict(color="#fff", width=2)),
                  hovertemplate="Today<extra></extra>")
    f.add_vline(x=res.spend[ch].max(), line_dash="dot", line_color=MUTED, line_width=1)
    f.update_layout(height=270, title=ch, margin=dict(t=40, b=30, l=10, r=10),
                    xaxis_title="Weekly spend", yaxis_title="Weekly revenue")
    with cols[i % len(cols)]:
        st.plotly_chart(f, width="stretch")
        st.caption(f"Today {money(ref[ch])}/wk; next {cur}1 returns {ratio(ct.loc[ch, 'mroi'])}. "
                   "Right of the dotted line is beyond any week in the data.")

# ------------------------------------------------------------------ carryover
st.subheader("How long each channel keeps working")
weeks = 10
thetas = pd.DataFrame([{ch: m.params[ch].theta for ch in res.channels} for m in ens.models])
f = go.Figure()
for ch in res.channels:
    w = decay_weights(float(thetas[ch].median()), weeks)
    f.add_scatter(x=np.arange(weeks), y=w, name=ch, mode="lines+markers",
                  line=dict(color=colors[ch], width=2), marker=dict(size=5))
f.update_layout(height=340, xaxis_title="Weeks after spend", yaxis_title="Effect still active",
                yaxis_tickformat=".0%")
st.plotly_chart(f, width="stretch")
hl = []
for ch in res.channels:
    lo, mid, hi_ = (half_life(thetas[ch].quantile(q)) for q in (0.1, 0.5, 0.9))
    hl.append(f"{ch} {mid:.1f} ({lo:.1f} to {hi_:.1f})")
st.caption("Median half-life in weeks, with the range across models: " + "; ".join(hl) + ".")

# ------------------------------------------------------------------ table
with st.expander("Full channel table"):
    out = pd.DataFrame({
        "Spend": ct.spend.map(money),
        "Revenue driven": ct.contribution.map(money),
        "Avg return": ct.roi.map(ratio),
        "Avg return range": ct.roi_lo.map(ratio) + " to " + ct.roi_hi.map(ratio),
        f"Next {cur}1 returns": ct.mroi.map(ratio),
        "Next-unit range": ct.mroi_lo.map(ratio) + " to " + ct.mroi_hi.map(ratio),
        "Today per week": ct.ref_weekly_spend.map(money),
    })
    st.dataframe(out, width="stretch")
