import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm.optimiser import total_response
from ui import state
from ui.theme import GAIN, INK, LOSS, budget_strip, channel_colors, lede, money, note, pct, ratio

ss = st.session_state
res, ens, prep = state.get_models()
cfg = ss.cfg
colors = channel_colors(res.channels)
cur = ens.reference_spend()

st.title("Scenario planner")
lede("Set a weekly spend level for each channel and see what the model expects. "
     "Each level is treated as sustained week after week, so carryover is fully built up.")

if st.button("Reset to today's spend"):
    for ch in res.channels:
        ss[f"sc_{ch}"] = 100
    st.rerun()

plan = {}
cols = st.columns(len(res.channels))
for i, ch in enumerate(res.channels):
    with cols[i]:
        ss.setdefault(f"sc_{ch}", 100)
        v = st.slider(ch, 0, 300, step=5, format="%d%%", key=f"sc_{ch}",
                      help=f"Percent of today's {money(cur[ch])} a week.")
        plan[ch] = cur[ch] * v / 100
        st.caption(money(plan[ch]) + "/wk")
plan = pd.Series(plan)

rev_now = total_response(ens, cur)
rev_plan = total_response(ens, plan)
d_spend = plan.sum() - cur.sum()
d_rev = rev_plan - rev_now

c1, c2, c3 = st.columns(3)
c1.metric("Weekly media spend", money(plan.sum()), money(d_spend, signed=True).replace("−", "-"), delta_color="off")
c2.metric("Weekly revenue from media", money(rev_plan), money(d_rev, signed=True).replace("−", "-"))
if abs(d_spend) > 1:
    inc = d_rev / d_spend
    c3.metric("Return on the change", ratio(inc),
              help="Extra revenue divided by extra spend. Below 1 means the added spend loses money "
                   "before margins; above 1 means it covers its cost in revenue.")
else:
    c3.metric("Return on the change", "–", help="Change total spend to see the return on the difference.")

if abs(d_spend) <= 1 and d_rev > 0:
    note(f"Same budget, {money(d_rev)} more revenue a week. A pure reallocation like this is the cheapest gain available.")
elif d_spend > 1 and d_rev / d_spend < 1:
    note(f"Every extra {cfg.currency}1 here brings back {ratio(d_rev / d_spend)}. Unless your gross margin "
         "is unusually high, this plan spends more than it earns.", warn=True)

budget_strip([("Today", cur), ("This scenario", plan)], colors)

# ------------------------------------------------------------------ waterfall
st.subheader("Where the change comes from")
per = pd.Series({ch: float(ens.weekly_response(ch, plan[ch]) - ens.weekly_response(ch, cur[ch]))
                 for ch in res.channels})
fig = go.Figure(go.Waterfall(
    x=["Today", *per.index, "Scenario"],
    measure=["absolute", *["relative"] * len(per), "total"],
    y=[rev_now, *per.values, rev_plan],
    increasing=dict(marker_color=GAIN), decreasing=dict(marker_color=LOSS),
    totals=dict(marker_color=INK), connector=dict(line=dict(color="#C5CCD6", width=1)),
    text=[money(rev_now), *[money(v, signed=True) for v in per.values], money(rev_plan)],
    textposition="outside",
))
lo = min(rev_now, rev_plan) + min(0, per.min())
fig.update_layout(height=380, yaxis=dict(range=[max(0, lo * 0.9), max(rev_now, rev_plan) * 1.08],
                                         title="Weekly revenue from media"), showlegend=False)
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ saved scenarios
st.subheader("Compare scenarios")
c1, c2 = st.columns([3, 1])
name = c1.text_input("Scenario name", f"Scenario {len(ss.scenarios) + 1}", label_visibility="collapsed")
if c2.button("Save scenario", width="stretch"):
    ss.scenarios.append(dict(name=name, spend=plan.copy(), revenue=rev_plan))
    st.toast(f"Saved {name}")

if ss.scenarios:
    rows = [dict(Scenario="Today", **{ch: money(cur[ch]) for ch in res.channels},
                 Spend=money(cur.sum()), Revenue=money(rev_now), Change="–")]
    for s in ss.scenarios:
        rows.append(dict(Scenario=s["name"], **{ch: money(s["spend"][ch]) for ch in res.channels},
                         Spend=money(s["spend"].sum()), Revenue=money(s["revenue"]),
                         Change=pct(s["revenue"] / rev_now - 1, signed=True, digits=1)))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    budget_strip([("Today", cur)] + [(s["name"], s["spend"]) for s in ss.scenarios], colors)
    if st.button("Clear saved scenarios"):
        ss.scenarios = []
        st.rerun()
else:
    st.caption("Save a scenario to line it up against today's plan and others you try.")
