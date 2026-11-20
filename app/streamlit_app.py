"""Streamlit app of the search algorithm recommender.

Start from the repository root (so ``.streamlit/config.toml`` with the
colour theme is found):

    streamlit run app/streamlit_app.py

Three sections:

* Präsentation: the big picture with the key results (start page) and six
  everyday stories that each lead to a different recommendation,
* Werkzeug: the five work steps data -> access pattern -> analysis ->
  recommendation -> execution, for exploring freely,
* Lernen: the learning mode. Each page lives in its own module in ``app/views/``; shared state and
layout helpers are in ``app/views/common.py``.
"""

import streamlit as st

from views import analysis_page, data_page, execution_page, learning_page, recommendation_page, start_page, story_page, workload_page
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
    PAGES[number] = st.Page(render, title=f"{number} · {title}", icon=icon, url_path=url_path)
PAGES["start"] = st.Page(start_page.render, title="Überblick", icon=":material/home:", url_path="ueberblick", default=True)
PAGES["story"] = st.Page(story_page.render, title="Geschichten", icon=":material/auto_stories:", url_path="geschichten")

navigation = st.navigation(
    {
        "Präsentation": [PAGES["start"], PAGES["story"]],
        "Werkzeug": [PAGES[n] for n in range(1, 6)],
        "Lernen": [PAGES[6]],
    }
)
with st.sidebar:
    st.markdown("### 🔎 Suchalgorithmus-Empfehlungssystem")
    st.caption("DHBW Ravensburg · Theoretische Informatik")
# The summary of the tool's current data only makes sense on the tool pages;
# the stories have their own data and would be confused with it.
if navigation in [PAGES[n] for n in range(1, 7)]:
    sidebar_summary()
navigation.run()
