import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm.optimiser import frontier, optimise, robustness
from ui import state
from ui.theme import GAIN, INK, MUTED, PRIMARY, budget_strip, channel_colors, lede, money, note, pct, ratio

ss = st.session_state
res, ens, prep = state.get_models()
cfg = ss.cfg
colors = channel_colors(res.channels)
cur = ens.reference_spend()

st.title("Budget optimiser")
lede("Choose a weekly budget and how far each channel may move from today. The optimiser finds the "
     "split that maximises media-driven revenue within those limits.")

# ------------------------------------------------------------------ inputs
c1, c2, c3 = st.columns([2, 1, 1])
change = c1.slider("Total weekly budget versus today", -50, 100, 0, step=5, format="%+d%%")
budget = cur.sum() * (1 + change / 100)
c1.caption(f"{money(budget)} a week (today {money(cur.sum())})")
floor = c2.number_input("Default floor (% of today)", 0, 100, 50, step=5,
                        help="No channel drops below this share of its current spend.")
ceil = c3.number_input("Default ceiling (% of today)", 100, 500, 200, step=10,
                       help="No channel rises above this share of its current spend.")

with st.expander("Set limits for individual channels"):
    st.caption("Set floor and ceiling to the same value to lock a channel, for example a contracted TV buy.")
    limits = pd.DataFrame({"Channel": res.channels, "Today": [round(cur[c]) for c in res.channels],
                           "Floor %": floor, "Ceiling %": ceil})
    edited = st.data_editor(
        limits, hide_index=True, width="stretch", disabled=["Channel", "Today"],
        column_config={
            "Floor %": st.column_config.NumberColumn(min_value=0, max_value=500, step=5),
            "Ceiling %": st.column_config.NumberColumn(min_value=0, max_value=1000, step=5),
        },
        key=f"limits_{floor}_{ceil}",
    )
lower = pd.Series(edited["Floor %"].values / 100, index=res.channels) * cur
upper = pd.Series(edited["Ceiling %"].values / 100, index=res.channels) * cur
bad = [c for c in res.channels if lower[c] > upper[c]]
if bad:
    st.error(f"Floor is above ceiling for {', '.join(bad)}. Fix the limits to run the optimiser.")
    st.stop()

try:
    alloc = optimise(ens, budget, cur, lower, upper)
except ValueError as e:
    st.error(str(e))
    st.stop()
ss.allocation = alloc
tbl = alloc.table(ens)
rob = robustness(res, budget, cur, lower, upper)

# baseline for comparison: today's mix scaled to the new budget
same_mix = cur * budget / cur.sum()
rev_same = float(sum(ens.weekly_response(c, same_mix[c]) for c in res.channels))
gain_vs_mix = alloc.optimal_revenue - rev_same

# ------------------------------------------------------------------ headline
st.subheader("Recommended split")
if change == 0:
    note(f"Same budget, <b>{money(alloc.uplift)}</b> more weekly revenue from media "
         f"({pct(alloc.uplift / alloc.current_revenue, signed=True, digits=1)}).")
else:
    note(f"At {money(budget)} a week the best split earns <b>{money(alloc.optimal_revenue)}</b> from media: "
         f"{money(gain_vs_mix, signed=True)} compared with keeping today's mix at that budget, and "
         f"{money(alloc.uplift, signed=True)} compared with today.")
if not alloc.success:
    note(f"The solver reported: {alloc.message}. Results may not be fully optimal.", warn=True)

budget_strip([("Today", cur), ("Recommended", alloc.optimal)], colors)

def agree(ch):
    n = int(rob.loc[ch, "models"])
    v = tbl.loc[ch, "change_pct"]
    if v > 0.05:
        return f"{int(rob.loc[ch, 'up'])} of {n} raise it"
    if v < -0.05:
        return f"{int(rob.loc[ch, 'down'])} of {n} cut it"
    return f"{n - int(rob.loc[ch, 'up']) - int(rob.loc[ch, 'down'])} of {n} hold it"


show = pd.DataFrame({
    "Today": tbl.current.map(money),
    "Recommended": tbl.optimal.map(money),
    "Change": tbl.change_pct.map(lambda v: pct(v, signed=True)),
    "Revenue today": tbl.revenue_current.map(money),
    "Revenue recommended": tbl.revenue_optimal.map(money),
    f"Next {cfg.currency}1 returns": tbl.mroi_optimal.map(ratio),
    "Limit hit": tbl.at_bound.replace({"floor": "At floor", "ceiling": "At ceiling", "": ""}),
    "Models that agree": [agree(c) for c in tbl.index],
})
st.dataframe(show, width="stretch")
st.caption(f"The recommendation averages {int(rob.models.iloc[0])} near-best model variants. The last column "
           "re-runs the optimiser on each variant alone and counts how many move the channel the same way.")
shaky = [c for c in tbl.index if abs(tbl.loc[c, "change_pct"]) > 0.05 and
         max(rob.loc[c, "up"], rob.loc[c, "down"]) / rob.loc[c, "models"] < 0.6]
if shaky:
    note(f"The models split on {', '.join(shaky)}. Treat those moves as a hypothesis to test, not a decision.",
         warn=True)

# ------------------------------------------------------------------ why
st.subheader("Why this split")
lede("At the best split, the next unit of spend earns the same in every channel that is free to move. "
     "If one channel earned more, shifting money into it would raise revenue. Channels at a limit "
     "are the exception: the limit, not the curve, is holding them.")
fig = go.Figure()
fig.add_bar(x=tbl.index, y=tbl.mroi_current, name="Today", marker_color="#B9C2D0")
fig.add_bar(x=tbl.index, y=tbl.mroi_optimal, name="Recommended", marker_color=[colors[c] for c in tbl.index])
fig.add_hline(y=1, line_dash="dot", line_color=INK, annotation_text="Break-even")
fig.update_layout(barmode="group", height=360, yaxis_title=f"Revenue from the next {cfg.currency}1",
                  title=f"Return on the next {cfg.currency}1, before and after")
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ frontier
st.subheader("What each budget level can buy")
fr = frontier(ens, cur, floor / 100, ceil / 100)
fig = go.Figure()
fig.add_scatter(x=fr.budget, y=fr.same_mix_revenue, name="Today's mix, scaled",
                line=dict(color=MUTED, dash="dot", width=2))
fig.add_scatter(x=fr.budget, y=fr.optimal_revenue, name="Best split at each budget",
                line=dict(color=PRIMARY, width=2.6))
fig.add_scatter(x=[budget], y=[alloc.optimal_revenue], mode="markers", name="Your selection",
                marker=dict(size=12, color=GAIN, line=dict(color="#fff", width=2)))
fig.update_layout(height=400, xaxis_title="Weekly media budget", yaxis_title="Weekly revenue from media")
st.plotly_chart(fig, width="stretch")

fr = fr.dropna()
if len(fr) > 2:
    marg = np.diff(fr.optimal_revenue) / np.diff(fr.budget)
    below = np.where(marg < 1)[0]
    if len(below):
        b = fr.budget.iloc[below[0]]
        st.caption(f"Beyond about {money(b)} a week, each extra {cfg.currency}1 of budget returns less than "
                   f"{cfg.currency}1 in revenue even when allocated well. Where the line ends, the ceilings "
                   "stop the optimiser from spending more.")

# ------------------------------------------------------------------ export
export = tbl[["current", "optimal", "change", "revenue_current", "revenue_optimal", "mroi_optimal", "at_bound"]]
st.download_button("Download plan (CSV)", export.reset_index().to_csv(index=False).encode(),
                   "mmm_budget_plan.csv", "text/csv", type="primary")
note("These are steady-state answers: they assume the plan runs long enough for carryover to build up. "
     "Before moving large sums, test the biggest shift with a geo or holdout experiment.")
