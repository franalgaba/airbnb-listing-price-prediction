from src import home, eda, price_predict, training

import streamlit as st
import awesome_streamlit as ast

ast.core.services.other.set_logging_format()

PAGES = {
    ":house: Home": home,
    ":bar_chart: EDA": eda,
    ":star: Model Training": training,
    ":rocket: Model Deployment": price_predict,
}

st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Jump to:", list(PAGES.keys()))

page = PAGES[selection]

with st.spinner(f"Loading {selection} ..."):
    ast.shared.components.write_page(page)
