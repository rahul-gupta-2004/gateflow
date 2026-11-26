# Gateway - Automatic Car Parking Handling Agent

GateFlow is a Streamlit and SQLite3 application for automatic number plate detection, vehicle entry/exit tracking, hourly billing, vehicle records, and rate management.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Database:** SQLite3 (`db/gateflow.db`)
- **Vision & OCR:** YOLOv8 (`model/best.pt`), OpenCV, PaddleOCR

## Features
- **Dashboard:** View active cars inside, registered vehicles, current hourly rate, and summary charts.
- **Handle Entry:** Upload a vehicle image to detect and read the number plate, record a new entry (`active`), or complete an exit (`completed`) with duration and hourly billing.
- **View Db Records:** Browse `vehicles`, `rates`, and `sessions` tables.
- **Manage Car Details:** Add, update, and delete vehicle records.
- **Manage Hourly Rates:** View and update the hourly parking rate.
