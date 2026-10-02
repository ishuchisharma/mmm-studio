import streamlit as st

from mmm.optimiser import optimise
from ui import state
from ui.theme import budget_strip, channel_colors, findings, money, note, pct, ratio

res, ens, prep = state.get_models()
cfg = st.session_state.cfg
colors = channel_colors(res.channels)
cur = ens.reference_spend()
alloc = optimise(ens, cur.sum(), cur, cur * 0.5, cur * 2.0)
tbl = alloc.table(ens)

gainers = tbl[tbl.change > cur.sum() * 0.01].sort_values("change", ascending=False)
losers = tbl[tbl.change < -cur.sum() * 0.01].sort_values("change")
moved = gainers.change.sum()


def names(idx):
    idx = list(idx)
    return idx[0] if len(idx) == 1 else ", ".join(idx[:-1]) + " and " + idx[-1]


st.caption("Media Mix Studio")
if moved > 0 and alloc.uplift > 0:
    headline = (
        f'Moving <span class="num">{money(moved)}</span> a week into {names(gainers.index)} '
        f'and out of {names(losers.index)} could add <span class="num">{money(alloc.uplift)}</span> '
        f"in weekly revenue, with the same budget."
    )
else:
    headline = "Today's split is already close to the best this model can find."
st.markdown(f'<div class="hero-line">{headline}</div>', unsafe_allow_html=True)

lift = alloc.uplift / alloc.current_revenue if alloc.current_revenue else 0
st.markdown(
    f'<p class="hero-sub">That is {pct(lift, digits=1)} more revenue from media. Today means the average '
    f"week over the last {res.ref_weeks} weeks ({money(cur.sum())} of spend). The recommendation keeps every "
    f"channel between half and double its current spend. Change those limits in the Budget optimiser.</p>",
    unsafe_allow_html=True,
)

budget_strip([("Today", cur), ("Recommended", alloc.optimal)], colors)

ct = ens.channel_table()
best = ct.mroi.idxmax()
worst = ct.mroi.idxmin()
media_share = ct.contribution.sum() / res.y.sum()
top_roi = ct.roi.idxmax()

st.subheader("What the model sees")
findings([
    f"The next {cfg.currency}1 in <b>{best}</b> returns about {ratio(ct.loc[best, 'mroi'])} in revenue; "
    f"in <b>{worst}</b> it returns {ratio(ct.loc[worst, 'mroi'])}. That gap is why the plan above moves money.",
    f"Over the whole period, <b>{top_roi}</b> had the best average return at {ratio(ct.loc[top_roi, 'roi'])} "
    f"per {cfg.currency}1. Average and next-unit returns differ because each channel saturates.",
    f"Media drives about {pct(media_share)} of revenue. The rest is baseline demand, competitor activity, "
    f"seasonality and other non-media factors, so even a perfect media plan moves a limited slice.",
])

m = res.metrics
trust = (
    f"The model explains {pct(m['r2_full'])} of week-to-week revenue variation. On the last "
    f"{res.holdout_weeks} weeks, which it never saw while training, its forecasts were off by "
    f"{pct(m['mape_holdout'], digits=1)} on average. Plans on every page average the "
    f"{len(ens.models)} best-scoring model variants, so they do not hinge on one lucky fit."
)
note(trust)
for n in res.notes:
    note(n, warn=True)

st.subheader("Where to go next")
c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("views/model.py", label="Check the model fit", icon=":material/insights:")
    st.caption("See how well the model tracks actual revenue and where it struggles.")
with c2:
    st.page_link("views/scenario.py", label="Try your own plan", icon=":material/tune:")
    st.caption("Move each channel's spend and see the predicted effect on revenue.")
with c3:
    st.page_link("views/optimiser.py", label="Set a budget and optimise", icon=":material/balance:")
    st.caption("Pick a total budget and limits per channel; get the best split.")
