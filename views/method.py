import streamlit as st

from ui import state
from ui.theme import lede

res, prep = state.get_fit()

st.title("How the model works")
lede("A media mix model explains weekly revenue as a baseline plus the effect of each channel, "
     "after accounting for two facts about advertising: its effect lingers, and it wears out.")

st.subheader("1. Carryover")
st.markdown("An ad seen this week still nudges purchases next week. Each channel's spend is turned into an "
            "effective stock that decays by a fixed share every week (geometric adstock):")
st.latex(r"A_t = x_t + \theta \, A_{t-1}, \qquad 0 \le \theta < 1")
st.markdown("Offline channels search θ between 0.1 and 0.8; digital channels between 0 and 0.4.")

st.subheader("2. Diminishing returns")
st.markdown("Doubling spend does not double sales. The stock passes through a Hill curve that flattens as "
            "spend grows. α sets the shape (above 1 gives an S-curve with a threshold), γ the point where "
            "half the maximum effect is reached:")
st.latex(r"S_t = \frac{A_t^{\alpha}}{A_t^{\alpha} + \gamma^{\alpha}}")

st.subheader("3. Regression")
st.markdown("Revenue is regressed on the transformed channels plus non-media controls: trend, two Fourier "
            "pairs for yearly seasonality, events and any other factors supplied. Ridge regularisation "
            "stabilises estimates when channels overlap, and media coefficients are bounded at zero or "
            "above, since a channel cannot reduce sales by being bought.")
st.latex(r"y_t = \beta_0 + \sum_{c} \beta_c S_{c,t} + \sum_{k} \delta_k z_{k,t} + \varepsilon_t")

st.subheader("4. Choosing θ, α, γ and the penalty")
st.markdown(
    f"These cannot be estimated by the regression itself, so they are searched. The engine ran "
    f"{res.metrics['trials']:,} trials: 70% random draws across the allowed ranges, then 30% local "
    f"refinement around the best. Each trial is fitted on all but the last {res.holdout_weeks} weeks and "
    "scored on those held-out weeks, plus a penalty when the split of credit across channels is far from "
    "the split of spend (Robyn's DECOMP.RSSD). The penalty guards against fits that are accurate but "
    "attribute almost everything to one small channel."
)

st.subheader("5. From model to plan")
st.markdown("Holding weekly spend constant at s, carryover settles at s / (1 − θ), so steady-state weekly "
            "revenue from a channel is:")
st.latex(r"R_c(s) = \beta_c \cdot \mathrm{Hill}\!\left(\frac{s}{1-\theta_c}\right)")
st.markdown("The optimiser maximises the sum of these curves for a fixed total budget with SLSQP, "
            "restarting from several points because S-shaped curves can trap a single run. At the "
            "optimum, the derivative of every unconstrained channel's curve is equal.")

st.subheader("Limits worth knowing")
st.markdown("""
- **Correlation, not proof.** The model finds patterns between spend and revenue. If spend rose because demand was already rising, for example search budgets following seasonal interest, the model credits media for demand it did not create. Geo-lift or holdout experiments are the way to calibrate this.
- **Uncertainty is understated.** Bootstrap intervals hold θ, α and γ fixed at the chosen values. A full Bayesian model, such as Google Meridian or PyMC-Marketing, would carry that uncertainty through.
- **Steady state, not flighting.** Plans assume spend is held level week after week. Bursting a channel in some weeks and pausing in others can work differently, especially for S-shaped channels.
- **Only the past range is reliable.** Response curves beyond the highest spend in the data are extrapolations.
- **Revenue, not profit.** A return above 1 covers media cost in revenue but not cost of goods. Compare next-unit returns with 1 ÷ gross margin to judge profitability.
""")

st.subheader("References")
st.markdown("""
- Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.* Google Inc.
- Meta Open Source. *Robyn: semi-automated marketing mix modeling.* The sample dataset used here ships with it.
- Chan, D., & Perry, M. (2017). *Challenges and Opportunities in Media Mix Modeling.* Google Inc.
""")
