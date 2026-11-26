import json
import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
from paddleocr import TextRecognition
from ultralytics import YOLO
from db_utils import (
    MODEL_PATH,
    OUTPUT_PATH,
    add_rate,
    complete_exit,
    create_entry,
    delete_vehicle,
    get_active_session,
    get_current_rate,
    get_table_rows,
    init_db,
    upsert_vehicle,
)

# set cpu thread count
torch.set_num_threads(2)

# set page config
st.set_page_config(
    page_title="Gateflow - Automatic Car Parking Handling Agent",
    layout="wide",
)

# initialize database tables
init_db()


# load yolo and paddleocr models once
@st.cache_resource
def load_models():
    yolo_model = YOLO(str(MODEL_PATH))
    ocr_model = TextRecognition(model_name="en_PP-OCRv5_mobile_rec", device="cpu")
    return yolo_model, ocr_model


# preprocess cropped plate image
def preprocess_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    enlarged = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enlarged)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    return {
        "original": plate,
        "enhanced": cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR),
    }


# run paddleocr on preprocessed variants
def read_plate_text(plate, recognizer):
    OUTPUT_PATH.mkdir(exist_ok=True)
    variants = preprocess_plate(plate)
    best_text = ""
    best_conf = 0.0

    for name, img in variants.items():
        temp_file = OUTPUT_PATH / f"temp_{name}.jpg"
        cv2.imwrite(str(temp_file), img)
        preds = recognizer.predict(input=str(temp_file))

        for pred in preds:
            data = pred.json
            if isinstance(data, str):
                data = json.loads(data)
            res = data.get("res", {})
            raw = res.get("rec_text", "")
            conf = float(res.get("rec_score", 0.0))
            clean = re.sub(r"[^A-Z0-9]", "", raw.upper())

            # pick highest confidence reading
            if clean and conf >= best_conf:
                best_conf = conf
                best_text = clean

    return best_text, best_conf


# detect plate using yolo model
def detect_plate(image, model, recognizer):
    preds = model.predict(source=image, imgsz=640, conf=0.25, device="cpu", verbose=False)
    annotated = image.copy()
    detected_plates = []

    h, w = image.shape[:2]
    for pred in preds:
        if pred.boxes is None:
            continue
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            plate_text, ocr_conf = read_plate_text(crop, recognizer)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                plate_text or "Unknown",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            detected_plates.append({
                "plate_number": plate_text,
                "confidence": ocr_conf,
                "crop": crop,
            })

    return annotated, detected_plates

# render dashboard overview page with vibrant multicolor charts
def dashboard_page():
    st.title("Gateway - Automatic Car Parking Handling Agent")
    st.write("Gateflow Parking Entry, Exit, Billing, Vehicle, And Rate Manager")

    sessions_rows = get_table_rows("sessions")
    vehicles_rows = get_table_rows("vehicles")
    current_rate = get_current_rate()

    active_sessions = [r for r in sessions_rows if r["status"] == "active"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Cars Inside", len(active_sessions))
    col2.metric("Registered Vehicles", len(vehicles_rows))
    col3.metric("Current Hourly Rate", f"{current_rate:.2f}")

    st.info("Use The Left Navigation Menu To Open Handle Entry, View Db Records, Manage Car Details, Or Manage Hourly Rates.")

    st.subheader("Dashboard Charts")
    chart_col1, chart_col2 = st.columns(2)

    # chart 1: sessions by status with vivid status colors
    with chart_col1:
        st.caption("Sessions By Status")
        if sessions_rows:
            status_df = pd.DataFrame(sessions_rows)
            status_df["status"] = status_df["status"].str.title()
            status_counts = status_df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            fig1 = px.bar(
                status_counts,
                x="Status",
                y="Count",
                color="Status",
                text_auto=True,
                color_discrete_map={
                    "Completed": "#00CC96",
                    "Active": "#EF553B",
                },
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig1.update_traces(textposition="outside")
            fig1.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.write("No Session Records Yet")

    # chart 2: revenue by plate number with vivid multicolor palette
    with chart_col2:
        st.caption("Revenue By Plate Number")
        completed_rows = [
            r for r in sessions_rows
            if r["status"] == "completed" and r["total_amount"] is not None
        ]
        if completed_rows:
            rev_df = pd.DataFrame(completed_rows)
            rev_by_plate = (
                rev_df.groupby("plate_number", as_index=False)["total_amount"]
                .sum()
                .rename(columns={"plate_number": "Plate Number", "total_amount": "Total Amount"})
            )

            fig2 = px.bar(
                rev_by_plate,
                x="Plate Number",
                y="Total Amount",
                color="Plate Number",
                text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("No Completed Sessions Yet")

# render handle entry and exit page
def handle_entry_page():
    st.title("Gateflow - Automatic Car Parking Handling Agent")
    st.subheader("Handle Entry / Exit")

    model, recognizer = load_models()
    uploaded = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could Not Read Image")
        else:
            annotated_img, plates = detect_plate(image, model, recognizer)
            st.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                caption="Detected Plate",
                use_container_width=True,
            )

            if not plates:
                st.warning("No Number Plate Detected")
            else:
                detected_num = plates[0]["plate_number"]
                st.write(f"Detected Plate Number: **{detected_num or 'None'}**")

                # allow manual plate edit if ocr needs correction
                final_plate = st.text_input("Confirm Or Enter Plate Number", value=detected_num).strip().upper()

                if st.button("Process Entry / Exit"):
                    if not final_plate:
                        st.error("Plate Number Is Required")
                    else:
                        # check if car is currently active inside
                        active_session = get_active_session(final_plate)

                        if active_session is None:
                            # vehicle is not inside -> create new entry
                            entry_time, rate = create_entry(final_plate)
                            st.success(f"Entry Recorded For {final_plate} At {entry_time} (Rate: {rate:.2f}/Hr)")
                        else:
                            # vehicle is inside -> mark exit and calculate total
                            exit_time, duration, rate, total = complete_exit(active_session)
                            st.success(
                                f"Exit Recorded For {final_plate} At {exit_time} | "
                                f"Duration: {duration} Mins | Rate: {rate:.2f}/Hr | Total: {total:.2f}"
                            )


# render view db records page with 3 dropdowns
def view_db_records_page():
    st.title("Gateflow - Automatic Car Parking Handling Agent")
    st.subheader("View Db Records")

    vehicles_rows = get_table_rows("vehicles")
    rates_rows = get_table_rows("rates")
    sessions_rows = get_table_rows("sessions")

    with st.expander("Vehicles Table Dropdown", expanded=True):
        vehicle_labels = [
            f"{r['vehicle_id']} - {r['plate_number']} ({r['owner_name'] or 'No Owner'})"
            for r in vehicles_rows
        ]
        st.selectbox("Select Vehicle Record", options=vehicle_labels if vehicle_labels else ["No Records"])
        st.dataframe(vehicles_rows, use_container_width=True)

    with st.expander("Rates Table Dropdown", expanded=False):
        rate_labels = [
            f"{r['rate_id']} - {r['hourly_rate']:.2f}/Hr ({r['updated_at']})"
            for r in rates_rows
        ]
        st.selectbox("Select Rate Record", options=rate_labels if rate_labels else ["No Records"])
        st.dataframe(rates_rows, use_container_width=True)

    with st.expander("Sessions Table Dropdown", expanded=False):
        session_labels = [
            f"{r['session_id']} - {r['plate_number']} [{r['status'].title()}]"
            for r in sessions_rows
        ]
        st.selectbox("Select Session Record", options=session_labels if session_labels else ["No Records"])
        st.dataframe(sessions_rows, use_container_width=True)


# render manage car details page
def manage_car_details_page():
    st.title("Gateflow - Automatic Car Parking Handling Agent")
    st.subheader("Manage Car Details")

    with st.form("vehicle_form"):
        plate_number = st.text_input("Plate Number").strip().upper()
        owner_name = st.text_input("Owner Name").strip().title()
        phone_number = st.text_input("Phone Number").strip()
        email = st.text_input("Email").strip()
        submitted = st.form_submit_button("Save Vehicle")

        if submitted:
            if not plate_number:
                st.error("Plate Number Is Required")
            else:
                upsert_vehicle(plate_number, owner_name, phone_number, email)
                st.success(f"Saved Vehicle {plate_number}")
                st.rerun()

    vehicles = get_table_rows("vehicles")
    st.dataframe(vehicles, use_container_width=True)

    if vehicles:
        vehicle_ids = [v["vehicle_id"] for v in vehicles]
        selected_id = st.selectbox("Select Vehicle Id To Delete", options=vehicle_ids)
        if st.button("Delete Vehicle"):
            delete_vehicle(selected_id)
            st.success(f"Deleted Vehicle Id {selected_id}")
            st.rerun()


# render manage hourly rates page
def manage_hourly_rates_page():
    st.title("Gateflow - Automatic Car Parking Handling Agent")
    st.subheader("Manage Hourly Rates")

    current_rate = get_current_rate()
    st.metric("Current Hourly Rate", f"{current_rate:.2f}")

    with st.form("rate_form"):
        new_rate = st.number_input("New Hourly Rate", min_value=0.0, value=current_rate, step=5.0)
        submitted = st.form_submit_button("Update Hourly Rate")

        if submitted:
            add_rate(new_rate)
            st.success(f"Updated Hourly Rate To {new_rate:.2f}")
            st.rerun()

    st.write("Rate History")
    st.dataframe(get_table_rows("rates"), use_container_width=True)


# configure sidebar navigation
pages = [
    st.Page(dashboard_page, title="Dashboard", default=True),
    st.Page(handle_entry_page, title="Handle Entry"),
    st.Page(view_db_records_page, title="View Db Records"),
    st.Page(manage_car_details_page, title="Manage Car Details"),
    st.Page(manage_hourly_rates_page, title="Manage Hourly Rates"),
]

selected_page = st.navigation(pages)
selected_page.run()