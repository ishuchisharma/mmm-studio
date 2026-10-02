import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import state
from ui.theme import INK, LOSS, MUTED, PRIMARY, channel_colors, lede, money, note, pct

ss = st.session_state
st.title("Model fit")
lede("Before trusting any channel number, check that the model tracks real revenue, "
     "including weeks it was not trained on.")

with st.expander("Fit settings"):
    s = dict(ss.settings)
    c1, c2, c3 = st.columns(3)
    s["n_trials"] = c1.slider("Search trials", 300, 5000, s["n_trials"], step=100,
                              help="Each trial tries one set of carryover and saturation settings for every channel. "
                                   "More trials find better fits but take longer.")
    s["holdout"] = c2.slider("Weeks held out for testing", 0, 52, s["holdout"], step=2,
                             help="The most recent weeks are hidden during the search and used to score each trial.")
    s["rssd_weight"] = c3.slider("Penalty on implausible splits", 0.0, 1.0, s["rssd_weight"], step=0.05,
                                 help="Pushes the search away from fits that hand nearly all credit to one small channel. "
                                      "Borrowed from Meta's Robyn (DECOMP.RSSD). Zero means fit accuracy only.")
    c4, c5, c6 = st.columns(3)
    s["ref_weeks"] = c4.slider("Weeks that define 'today'", 4, 104, s["ref_weeks"], step=4,
                               help="Current weekly spend is the average over this many recent weeks.")
    s["n_alternatives"] = c5.slider("Models averaged for planning", 5, 60, s["n_alternatives"], step=5,
                                    help="The best-scoring random draws from the search. Their response curves are "
                                         "averaged for every plan, which keeps recommendations stable.")
    s["seed"] = int(c6.number_input("Random seed", 0, 10_000, s["seed"]))
    if st.button("Refit model", type="primary"):
        ss.settings = s
        state.reset_downstream()
        st.rerun()

res, prep = state.get_fit()
m = res.metrics
st.caption("Diagnostics below describe the single best-scoring model. Planning pages average it with "
           f"{len(res.alternatives)} near-best variants.")
colors = channel_colors(res.channels)

c1, c2, c3, c4 = st.columns(4)
c1.metric("R² on all weeks", f"{m['r2_full']:.2f}",
          help="Share of week-to-week revenue variation the model explains.")
c2.metric(f"R² on last {res.holdout_weeks} weeks", f"{m['r2_holdout']:.2f}",
          help="Scored with a model trained without these weeks: an honest out-of-sample test.")
c3.metric("Average forecast error", pct(m["mape_holdout"], digits=1),
          help="Mean absolute percentage error on the held-out weeks.")
c4.metric("Durbin–Watson", f"{m['durbin_watson']:.2f}",
          help="Checks whether errors follow each other. Close to 2 is good; below 1.5 means the model "
                "misses something persistent, like a trend or a slow-moving factor.")
for n in res.notes:
    note(n, warn=True)

# ------------------------------------------------------------------ actual vs predicted
st.subheader("Actual versus predicted revenue")
fig = go.Figure()
ntr = len(res.y) - res.holdout_weeks
if res.holdout_weeks:
    fig.add_vrect(x0=res.dates.iloc[ntr], x1=res.dates.iloc[-1], fillcolor="#E9EDF3", line_width=0,
                  annotation_text="Held out", annotation_position="top left",
                  annotation_font_color=MUTED)
fig.add_scatter(x=res.dates, y=res.y, name="Actual", line=dict(color=INK, width=1.8))
fig.add_scatter(x=res.dates, y=res.fitted, name="Model (all weeks)", line=dict(color=PRIMARY, width=1.6))
if res.holdout_weeks:
    fig.add_scatter(x=res.dates.iloc[ntr:], y=res.holdout_pred[ntr:], name="Forecast without these weeks",
                    line=dict(color=LOSS, width=1.6, dash="dot"))
fig.update_layout(height=400, yaxis_title="Revenue")
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ decomposition
st.subheader("What drove revenue each week")
media_total = res.contrib[res.channels].sum(axis=1)
share = media_total.sum() / res.y.sum()
lede(f"Media accounts for {pct(share)} of revenue across the period. The chart shows only the media "
     "slice so channel patterns stay readable; the baseline sits underneath it.")
fig = go.Figure()
for ch in res.channels:
    fig.add_scatter(x=res.dates, y=res.contrib[ch], name=ch, stackgroup="m",
                    line=dict(width=0.5, color=colors[ch]))
fig.update_layout(height=380, yaxis_title="Revenue from media")
st.plotly_chart(fig, width="stretch")

with st.expander("Baseline breakdown"):
    parts = res.contrib_controls.sum()
    bt = pd.DataFrame({"Component": parts.index, "Total contribution": [money(v) for v in parts.values],
                       "Share of revenue": [pct(v / res.y.sum(), digits=1) for v in parts.values]})
    st.dataframe(bt, hide_index=True, width="stretch")
    st.caption("Large positive and negative baseline pieces can offset each other; that is normal in a "
               "regression with an intercept. Read them together, not one by one.")

# ------------------------------------------------------------------ diagnostics
st.subheader("Diagnostics")
resid = res.y - res.fitted
t1, t2, t3 = st.tabs(["Residuals over time", "Autocorrelation", "Search progress"])
with t1:
    f = go.Figure(go.Bar(x=res.dates, y=resid, marker_color=np.where(resid >= 0, PRIMARY, LOSS)))
    f.update_layout(height=320, yaxis_title="Actual minus predicted")
    st.plotly_chart(f, width="stretch")
    st.caption("Look for runs of same-coloured bars or a shape over time. Random scatter around zero is what you want.")
with t2:
    r = resid - resid.mean()
    lags = np.arange(1, 14)
    acf = [np.sum(r[k:] * r[:-k]) / np.sum(r**2) for k in lags]
    band = 1.96 / np.sqrt(len(r))
    f = go.Figure(go.Bar(x=lags, y=acf, marker_color=[LOSS if abs(a) > band else PRIMARY for a in acf]))
    f.add_hline(y=band, line_dash="dot", line_color=MUTED)
    f.add_hline(y=-band, line_dash="dot", line_color=MUTED)
    f.update_layout(height=320, xaxis_title="Lag (weeks)", yaxis_title="Correlation")
    st.plotly_chart(f, width="stretch")
    st.caption("Bars past the dotted lines mean errors in one week predict errors some weeks later.")
with t3:
    log = res.search_log.copy()
    log["best"] = log.objective.cummin()
    f = go.Figure()
    f.add_scatter(x=log.trial, y=log.objective, mode="markers", name="Trial",
                  marker=dict(size=3, color="#B9C2D0"))
    f.add_scatter(x=log.trial, y=log.best, name="Best so far", line=dict(color=PRIMARY, width=2))
    f.update_layout(height=320, xaxis_title="Trial", yaxis_title="Holdout error + split penalty",
                    yaxis=dict(range=[log.objective.min() * 0.95, log.objective.quantile(0.9)]))
    st.plotly_chart(f, width="stretch")
    st.caption("The first 70% of trials explore the full range; the rest refine around the best one found.")

# ------------------------------------------------------------------ export
summary = res.channel_table().reset_index()
st.download_button("Download channel results (CSV)", summary.to_csv(index=False).encode(),
                   "mmm_channel_results.csv", "text/csv")
