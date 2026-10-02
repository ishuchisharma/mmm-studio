import streamlit as st

from ui import state
from ui.theme import inject_css, register_plotly_template

st.set_page_config(page_title="Media Mix Studio", page_icon="📊", layout="wide")
inject_css()
register_plotly_template()
state.init()

pages = {
    "": [st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True)],
    "Understand": [
        st.Page("views/data.py", title="Data", icon=":material/table_chart:"),
        st.Page("views/model.py", title="Model fit", icon=":material/insights:"),
        st.Page("views/channels.py", title="Channel returns", icon=":material/stacked_line_chart:"),
    ],
    "Plan": [
        st.Page("views/scenario.py", title="Scenario planner", icon=":material/tune:"),
        st.Page("views/optimiser.py", title="Budget optimiser", icon=":material/balance:"),
    ],
    "Reference": [st.Page("views/method.py", title="How the model works", icon=":material/menu_book:")],
}
nav = st.navigation(pages)

with st.sidebar:
    cfg = st.session_state.cfg
    st.caption(f"Data: {cfg.dataset_name}")
    st.caption(f"{len(st.session_state.df)} weeks of data across {len(cfg.media)} channels")

nav.run()
