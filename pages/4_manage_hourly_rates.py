import streamlit as st
from src.db_utils import add_rate, get_current_rate, get_table_rows, init_db

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
st.subheader("manage hourly rates")

# display active rate
current_rate = get_current_rate()
st.metric("current hourly rate", f"{current_rate:.2f}")

# form to add new hourly rate
with st.form("rate_form"):
    new_rate = st.number_input("new hourly rate", min_value=0.0, value=current_rate, step=5.0)
    submitted = st.form_submit_button("update hourly rate")

    if submitted:
        add_rate(new_rate)
        st.success(f"updated hourly rate to {new_rate:.2f}")
        st.rerun()

# show rate history
st.write("rate history")
st.dataframe(get_table_rows("rates"), use_container_width=True)