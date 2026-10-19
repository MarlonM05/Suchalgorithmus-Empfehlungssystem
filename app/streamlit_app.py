"""Streamlit app of the search algorithm recommender.

Start with:  streamlit run app/streamlit_app.py

The six pages follow the workflow of the tool: data -> access pattern ->
analysis -> recommendation -> execution, plus a learning mode. Each page lives
in its own module in ``app/views/``.
"""

import streamlit as st

from views import analysis_page, data_page, execution_page, learning_page, recommendation_page, workload_page
from views.common import init_state, sidebar_summary

st.set_page_config(page_title="Suchalgorithmus-Empfehlungssystem", page_icon="🔎", layout="wide")
init_state()

pages = [
    st.Page(data_page.render, title="1 · Daten", url_path="daten", default=True),
    st.Page(workload_page.render, title="2 · Zugriffsmuster", url_path="zugriffsmuster"),
    st.Page(analysis_page.render, title="3 · Analyse", url_path="analyse"),
    st.Page(recommendation_page.render, title="4 · Empfehlung", url_path="empfehlung"),
    st.Page(execution_page.render, title="5 · Ausführung", url_path="ausfuehrung"),
    st.Page(learning_page.render, title="6 · Lernmodus", url_path="lernmodus"),
]
navigation = st.navigation(pages)
with st.sidebar:
    st.markdown("### Suchalgorithmus-Empfehlungssystem")
    st.caption("DHBW Ravensburg · Theoretische Informatik")
sidebar_summary()
navigation.run()
