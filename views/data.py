import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm.data import DataConfig, load_default, prepare, validate, vif
from ui import state
from ui.theme import LOSS, MUTED, PRIMARY, channel_colors, lede, money, note

ss = st.session_state
st.title("Data")
lede("The model needs weekly revenue, weekly spend for each paid channel, and any non-media factors "
     "that also move revenue, such as competitor activity, promotions or holidays.")

# ------------------------------------------------------------------ source
with st.expander("Use your own data", expanded=False):
    st.markdown(
        "Upload a CSV with one row per week. Spend columns must be in the same currency as revenue. "
        "Two years of history (104 weeks) gives the model enough variation to work with."
    )
    up = st.file_uploader("Weekly data (CSV)", type=["csv"])
    if st.button("Go back to the sample dataset", type="secondary"):
        ss.df, ss.cfg = load_default(), DataConfig()
        state.reset_downstream()
        st.rerun()

    if up is not None:
        raw = pd.read_csv(up)
        st.dataframe(raw.head(8), width="stretch", hide_index=True)
        cols = list(raw.columns)
        num = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
        with st.form("mapping"):
            c1, c2 = st.columns(2)
            date_col = c1.selectbox("Week column", cols)
            target = c2.selectbox("Revenue column", num)
            media_cols = st.multiselect("Spend columns, one per channel", [c for c in num if c != target])
            control_cols = st.multiselect(
                "Non-media factors (optional)",
                [c for c in num if c != target],
                help="Anything outside your media plan that moves revenue: competitor spend, price index, "
                     "distribution, weather. Leaving these out makes media look more powerful than it is.",
            )
            event_col = st.selectbox("Event or holiday column (optional)", ["None", *[c for c in cols if c not in num]])
            st.markdown("**How long does each channel's effect last?** Offline media is usually remembered for "
                        "weeks; search and social clicks mostly act within days.")
            decay = {}
            for c in media_cols:
                decay[c] = st.radio(c, ["Weeks (offline)", "Days (digital)"], horizontal=True, key=f"decay_{c}")
            c3, c4 = st.columns(2)
            currency = c3.text_input("Currency symbol", "₹", max_chars=3)
            style = c4.radio("Number style", ["K/M", "Lakh/Crore"], horizontal=True)
            submitted = st.form_submit_button("Use this data", type="primary")
        if submitted:
            new_cfg = DataConfig(
                date_col=date_col, target=target,
                media={c: c for c in media_cols},
                decay={c: ("slow" if decay[c].startswith("Weeks") else "fast") for c in media_cols},
                controls=[c for c in control_cols if c not in media_cols],
                event_col=None if event_col == "None" else event_col,
                event_baseline="",
                currency=currency, dataset_name=up.name,
            )
            problems = validate(raw, new_cfg)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                ss.df, ss.cfg, ss.number_style = raw, new_cfg, style
                state.reset_downstream()
                st.success("Data loaded. The model will refit when you open any analysis page.")
                st.rerun()

with st.sidebar:
    style = st.radio("Number style", ["K/M", "Lakh/Crore"],
                     index=["K/M", "Lakh/Crore"].index(ss.number_style), horizontal=True)
    ss.number_style = style

cfg = ss.cfg
prep = prepare(ss.df, cfg)
colors = channel_colors(prep.channels)

# ------------------------------------------------------------------ summary
total_spend = prep.spend.sum().sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Weeks", f"{len(prep.y)}")
c2.metric("Period", f"{prep.dates.min():%b %Y} to {prep.dates.max():%b %Y}")
c3.metric("Total revenue", money(prep.y.sum()))
c4.metric("Total media spend", money(total_spend), help="Across all channels in the model.")
if cfg.dataset_name.startswith("Robyn"):
    note("This is the simulated dataset Meta publishes with its open-source Robyn MMM package. "
         "Units are not real currency; the currency symbol is only for readability.")

# ------------------------------------------------------------------ spend and revenue over time
st.subheader("Spend and revenue by week")
fig = go.Figure()
for ch in prep.channels:
    fig.add_bar(x=prep.dates, y=prep.spend[ch], name=ch, marker_color=colors[ch])
fig.add_scatter(x=prep.dates, y=prep.y, name="Revenue", yaxis="y2",
                line=dict(color="#16213A", width=1.6))
fig.update_layout(barmode="stack", height=420, bargap=0.05,
                  yaxis=dict(title="Media spend"),
                  yaxis2=dict(title="Revenue", overlaying="y", side="right", showgrid=False))
st.plotly_chart(fig, width="stretch")

zero_share = (prep.spend == 0).mean()
flighted = zero_share[zero_share > 0.3]
if len(flighted):
    st.caption(
        "Weeks with no spend: " + ", ".join(f"{ch} {v*100:.0f}%" for ch, v in flighted.items())
        + ". On-and-off spending is useful here: it gives the model weeks to compare with and without each channel."
    )

# ------------------------------------------------------------------ collinearity
st.subheader("Do channels move together?")
lede("If two channels always rise and fall together, no model can tell which one drove the sales. "
     "This is the most common reason media mix results look implausible.")

left, right = st.columns([3, 2])
corr = pd.concat([prep.spend, pd.Series(prep.y, name="Revenue")], axis=1).corr()
heat = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
    colorscale=[[0, LOSS], [0.5, "#FFFFFF"], [1, PRIMARY]],
    text=np.round(corr.values, 2), texttemplate="%{text}", showscale=False,
))
heat.update_layout(height=380, title="Correlation between weekly series", yaxis=dict(autorange="reversed"))
left.plotly_chart(heat, width="stretch")

v = vif(prep.spend).sort_values(ascending=False)
vt = pd.DataFrame({"Channel": v.index, "VIF": v.values.round(2)})
vt["Read as"] = np.where(vt.VIF > 10, "Severe overlap", np.where(vt.VIF > 5, "Some overlap", "Separable"))
right.markdown("**Variance inflation factor**")
right.dataframe(vt, hide_index=True, width="stretch")
right.caption("VIF measures how well a channel's spend can be predicted from the others. "
              "Above 5 deserves attention; above 10 means the channel's own effect is poorly identified.")

with st.expander("Raw data"):
    st.dataframe(ss.df, width="stretch", hide_index=True)
