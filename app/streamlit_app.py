"""Streamlit app of the search algorithm recommender.

Start from the repository root (so ``.streamlit/config.toml`` with the
colour theme is found):

    streamlit run app/streamlit_app.py

The five work steps follow the workflow of the tool: data -> access pattern
-> analysis -> recommendation -> execution. The learning mode is a separate
section. Each page lives in its own module in ``app/views/``; shared state and
layout helpers are in ``app/views/common.py``.
"""

import streamlit as st

from views import analysis_page, data_page, execution_page, learning_page, recommendation_page, workload_page
from views.common import PAGES, STEPS, init_state, sidebar_summary

st.set_page_config(page_title="Suchalgorithmus-Empfehlungssystem", page_icon="🔎", layout="wide")
init_state()

RENDERERS = {
    1: (data_page.render, "daten"),
    2: (workload_page.render, "zugriffsmuster"),
    3: (analysis_page.render, "analyse"),
    4: (recommendation_page.render, "empfehlung"),
    5: (execution_page.render, "ausfuehrung"),
    6: (learning_page.render, "lernmodus"),
}
for number, (render, url_path) in RENDERERS.items():
    title, icon, _ = STEPS[number]
    PAGES[number] = st.Page(render, title=f"{number} · {title}", icon=icon, url_path=url_path, default=number == 1)

navigation = st.navigation({"Werkzeug": [PAGES[n] for n in range(1, 6)], "Lernen": [PAGES[6]]})
with st.sidebar:
    st.markdown("### 🔎 Suchalgorithmus-Empfehlungssystem")
    st.caption("DHBW Ravensburg · Theoretische Informatik")
sidebar_summary()
navigation.run()
