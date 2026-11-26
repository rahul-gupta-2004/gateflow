import streamlit as st
from src.db_utils import get_table_rows, init_db

import sys
from pathlib import Path

# ensure project root is on sys.path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# set page title
st.set_page_config(page_title="gateway - automatic car parking handling agent", layout="wide")
init_db()

st.title("gateway - automatic car parking handling agent")
st.subheader("view db records")

# dropdowns for each of the 3 tables
vehicles_rows = get_table_rows("vehicles")
rates_rows = get_table_rows("rates")
sessions_rows = get_table_rows("sessions")

with st.expander("vehicles table dropdown", expanded=True):
    st.selectbox("select vehicle record", options=vehicles_rows if vehicles_rows else ["no records"])
    st.dataframe(vehicles_rows, use_container_width=True)

with st.expander("rates table dropdown", expanded=False):
    st.selectbox("select rate record", options=rates_rows if rates_rows else ["no records"])
    st.dataframe(rates_rows, use_container_width=True)

with st.expander("sessions table dropdown", expanded=False):
    st.selectbox("select session record", options=sessions_rows if sessions_rows else ["no records"])
    st.dataframe(sessions_rows, use_container_width=True)