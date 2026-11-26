import sqlite3
from datetime import datetime
from pathlib import Path

# get project root path
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "gateflow.db"
MODEL_PATH = BASE_DIR / "model" / "best.pt"
OUTPUT_PATH = BASE_DIR / "output"


# open sqlite connection
def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# create tables if not exists
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # create vehicles table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            owner_name TEXT,
            phone_number TEXT,
            email TEXT,
            updated_at TEXT
        )
    """)

    # create rates table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rates (
            rate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hourly_rate REAL NOT NULL,
            updated_at TEXT
        )
    """)

    # create sessions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            status TEXT NOT NULL,
            entry_time TEXT,
            exit_time TEXT,
            duration_minutes INTEGER,
            hourly_rate REAL,
            total_amount REAL
        )
    """)

    # insert default rate if table is empty
    cur.execute("SELECT COUNT(*) AS cnt FROM rates")
    if cur.fetchone()["cnt"] == 0:
        cur.execute(
            "INSERT INTO rates (hourly_rate, updated_at) VALUES (?, ?)",
            (50.0, now_str()),
        )

    conn.commit()
    conn.close()


# return current timestamp string
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# get current hourly rate
def get_current_rate():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT hourly_rate FROM rates ORDER BY rate_id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return float(row["hourly_rate"]) if row else 50.0


# check if vehicle has an active session
def get_active_session(plate_number):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM sessions WHERE plate_number = ? AND status = 'active' ORDER BY session_id DESC LIMIT 1",
        (plate_number,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# start new parking session
def create_entry(plate_number):
    entry_time = now_str()
    rate = get_current_rate()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions (plate_number, status, entry_time, hourly_rate)
        VALUES (?, 'active', ?, ?)
        """,
        (plate_number, entry_time, rate),
    )
    conn.commit()
    conn.close()
    return entry_time, rate


# complete active parking session and calculate billing
def complete_exit(session):
    exit_dt = datetime.now()
    exit_time = exit_dt.strftime("%Y-%m-%d %H:%M:%S")
    entry_dt = datetime.strptime(session["entry_time"], "%Y-%m-%d %H:%M:%S")

    # calculate duration in minutes (minimum 1 minute)
    diff_minutes = int((exit_dt - entry_dt).total_seconds() // 60)
    duration_minutes = max(1, diff_minutes)

    # calculate hours billed (round up to next hour)
    billed_hours = max(1, (duration_minutes + 59) // 60)
    rate = float(session["hourly_rate"] if session["hourly_rate"] is not None else get_current_rate())
    total_amount = round(billed_hours * rate, 2)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE sessions
        SET status = 'completed',
            exit_time = ?,
            duration_minutes = ?,
            hourly_rate = ?,
            total_amount = ?
        WHERE session_id = ?
        """,
        (exit_time, duration_minutes, rate, total_amount, session["session_id"]),
    )
    conn.commit()
    conn.close()

    return exit_time, duration_minutes, rate, total_amount


# fetch all rows from a given table
def get_table_rows(table_name):
    allowed = {"vehicles", "rates", "sessions"}
    if table_name not in allowed:
        return []
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# add or update vehicle record
def upsert_vehicle(plate_number, owner_name, phone_number, email):
    plate_clean = plate_number.strip().upper()
    updated = now_str()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO vehicles (plate_number, owner_name, phone_number, email, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(plate_number) DO UPDATE SET
            owner_name = excluded.owner_name,
            phone_number = excluded.phone_number,
            email = excluded.email,
            updated_at = excluded.updated_at
        """,
        (plate_clean, owner_name.strip(), phone_number.strip(), email.strip(), updated),
    )
    conn.commit()
    conn.close()


# delete vehicle record by id
def delete_vehicle(vehicle_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM vehicles WHERE vehicle_id = ?", (vehicle_id,))
    conn.commit()
    conn.close()


# add new hourly rate record
def add_rate(hourly_rate):
    updated = now_str()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO rates (hourly_rate, updated_at) VALUES (?, ?)",
        (float(hourly_rate), updated),
    )
    conn.commit()
    conn.close()