import streamlit as st
from src.db_utils import delete_vehicle, get_table_rows, init_db, upsert_vehicle

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
st.subheader("manage car details (vehicles crud)")

# form to add or update vehicle
with st.form("vehicle_form"):
    plate_number = st.text_input("plate number").strip().upper()
    owner_name = st.text_input("owner name").strip()
    phone_number = st.text_input("phone number").strip()
    email = st.text_input("email").strip()
    submitted = st.form_submit_button("save vehicle")

    if submitted:
        if not plate_number:
            st.error("plate number is required")
        else:
            upsert_vehicle(plate_number, owner_name, phone_number, email)
            st.success(f"saved vehicle {plate_number}")
            st.rerun()

# display all vehicles
vehicles = get_table_rows("vehicles")
st.dataframe(vehicles, use_container_width=True)

# delete vehicle by id
if vehicles:
    vehicle_ids = [v["vehicle_id"] for v in vehicles]
    selected_id = st.selectbox("select vehicle_id to delete", options=vehicle_ids)
    if st.button("delete vehicle"):
        delete_vehicle(selected_id)
        st.success(f"deleted vehicle_id {selected_id}")
        st.rerun()