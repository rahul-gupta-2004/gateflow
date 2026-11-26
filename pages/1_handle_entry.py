import json
import re
import cv2
import numpy as np
import streamlit as st
import torch
from paddleocr import TextRecognition
from ultralytics import YOLO
from src.db_utils import (
    OUTPUT_PATH,
    MODEL_PATH,
    complete_exit,
    create_entry,
    get_active_session,
    init_db,
)
import sys
from pathlib import Path

# ensure project root is on sys.path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# set cpu thread count
torch.set_num_threads(2)

# set page title
st.set_page_config(page_title="gateway - automatic car parking handling agent", layout="wide")
init_db()

st.title("gateway - automatic car parking handling agent")
st.subheader("handle entry / exit")


# load yolo and paddleocr models once
@st.cache_resource
def load_models():
    yolo_model = YOLO(str(MODEL_PATH))
    ocr_model = TextRecognition(model_name="en_PP-OCRv5_mobile_rec", device="cpu")
    return yolo_model, ocr_model


model, recognizer = load_models()


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
def read_plate_text(plate):
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
def detect_plate(image):
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

            plate_text, ocr_conf = read_plate_text(crop)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                plate_text or "unknown",
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


# upload vehicle image
uploaded = st.file_uploader("upload vehicle image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("could not read image")
    else:
        annotated_img, plates = detect_plate(image)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="detected plate", use_container_width=True)

        if not plates:
            st.warning("no number plate detected")
        else:
            detected_num = plates[0]["plate_number"]
            st.write(f"detected plate number: **{detected_num or 'none'}**")

            # allow manual plate edit if ocr needs correction
            final_plate = st.text_input("confirm or enter plate number", value=detected_num).strip().upper()

            if st.button("process entry / exit"):
                if not final_plate:
                    st.error("plate number is required")
                else:
                    # check if car is currently active inside
                    active_session = get_active_session(final_plate)

                    if active_session is None:
                        # vehicle is not inside -> create new entry
                        entry_time, rate = create_entry(final_plate)
                        st.success(f"entry recorded for {final_plate} at {entry_time} (rate: {rate:.2f}/hr)")
                    else:
                        # vehicle is inside -> mark exit and calculate total
                        exit_time, duration, rate, total = complete_exit(active_session)
                        st.success(
                            f"exit recorded for {final_plate} at {exit_time} | "
                            f"duration: {duration} mins | rate: {rate:.2f}/hr | total: {total:.2f}"
                        )