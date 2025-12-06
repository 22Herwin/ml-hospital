import os
import logging
import sqlite3
from dotenv import load_dotenv
import datetime

load_dotenv()
logger = logging.getLogger(__name__)

# Path to SQLite database file
DB_PATH = os.getenv("SQLITE_DB", "hospital.db")

# ---------------------------
# Database Helpers
# ---------------------------
def get_db():
    """Return a SQLite connection (auto-creates db file)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # return dict-like rows
    return conn


def ensure_tables():
    """Create admissions table if it doesn't exist, and migrate schema if needed."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            admit_time TEXT,
            planned_admission_date TEXT,
            planned_admission_time TEXT,
            estimated_discharge_date TEXT,
            length_of_stay_days INTEGER,
            ward_type TEXT,
            estimated_days INTEGER,
            med_used TEXT,
            qty INTEGER,
            diagnosis_code TEXT,
            severity_score INTEGER,
            created_at TEXT,
            diagnosis_name TEXT,
            hospital_name TEXT,
            hospital_id TEXT
        );
    """)
    
    # Migrate existing table by adding missing columns
    try:
        # Check existing columns
        cursor.execute("PRAGMA table_info(admissions)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Add missing columns if needed
        required_columns = {
            'planned_admission_date': 'TEXT',
            'planned_admission_time': 'TEXT',
            'estimated_discharge_date': 'TEXT',
            'length_of_stay_days': 'INTEGER',
            'diagnosis_name': 'TEXT',
            'hospital_name': 'TEXT',
            'hospital_id': 'TEXT'
        }
        
        for col_name, col_type in required_columns.items():
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE admissions ADD COLUMN {col_name} {col_type}")
                print(f"✅ Added missing column: {col_name}")
        
    except Exception as e:
        print(f"Schema migration note: {str(e)}")
    
    conn.commit()
    conn.close()


# Run table creation at import time
ensure_tables()

# DEBUG: Check if table has data
def debug_check_data():
    """Debug function to check if data exists."""
    try:
        conn = get_db()
        cursor = conn.execute("SELECT COUNT(*) as count FROM admissions")
        row = cursor.fetchone()
        count = row['count'] if row else 0
        conn.close()
        print(f"DEBUG: Total admissions in database = {count}")
        return count
    except Exception as e:
        print(f"DEBUG: Error checking data - {str(e)}")
        return 0

debug_check_data()


# ---------------------------
# CRUD Operations
# ---------------------------
def insert_admission(admission_data: dict) -> bool:
    """Insert a new admission into SQLite."""
    try:
        conn = get_db()
        cursor = conn.cursor()

        record = {
            "patient_id": admission_data.get("patient_id"),
            "admit_time": admission_data.get("admit_time"),
            "planned_admission_date": admission_data.get("planned_admission_date"),
            "planned_admission_time": admission_data.get("planned_admission_time"),
            "estimated_discharge_date": admission_data.get("estimated_discharge_date"),
            "length_of_stay_days": admission_data.get("length_of_stay_days"),
            "ward_type": admission_data.get("ward_type"),
            "estimated_days": admission_data.get("estimated_days"),
            "med_used": admission_data.get("med_used"),
            "qty": admission_data.get("qty"),
            "diagnosis_code": admission_data.get("diagnosis_code"),
            "diagnosis_name": admission_data.get("diagnosis_name"),
            "severity_score": admission_data.get("severity_score"),
            "hospital_id": admission_data.get("hospital_id"),
            "hospital_name": admission_data.get("hospital_name"),
            "created_at": datetime.datetime.now().isoformat(),
        }

        # Remove None values to allow SQLite to use defaults
        keys = [k for k, v in record.items() if v is not None]
        values = [record[k] for k in keys]

        query = f"INSERT INTO admissions ({', '.join(keys)}) VALUES ({','.join('?' for _ in keys)})"

        cursor.execute(query, values)
        conn.commit()
        conn.close()

        logger.info(f"Inserted admission for patient {record.get('patient_id')}")
        return True

    except Exception as e:
        logger.error(f"SQLite insert error: {str(e)}", exc_info=True)
        return False


def get_all_admissions() -> list:
    """Fetch all admissions sorted by newest admit_time."""
    try:
        conn = get_db()
        cursor = conn.execute(
            "SELECT * FROM admissions ORDER BY admit_time DESC"
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        logger.info(f"Retrieved {len(rows)} admissions from database")
        return rows

    except Exception as e:
        logger.error(f"SQLite fetch error: {str(e)}")
        return []


def get_patient_admissions(patient_id: str) -> list:
    """Fetch admissions for a specific patient."""
    try:
        conn = get_db()
        cursor = conn.execute(
            "SELECT * FROM admissions WHERE patient_id = ? ORDER BY admit_time DESC",
            (patient_id,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    except Exception as e:
        logger.error(f"SQLite patient fetch error: {str(e)}")
        return []


def update_admission(admission_id: int, updates: dict) -> bool:
    """Update an admission record."""
    try:
        conn = get_db()
        cursor = conn.cursor()

        keys = [f"{k} = ?" for k in updates.keys()]
        values = list(updates.values())

        query = f"UPDATE admissions SET {', '.join(keys)} WHERE id = ?"
        values.append(admission_id)

        cursor.execute(query, values)
        conn.commit()
        conn.close()

        logger.info(f"Updated admission {admission_id}")
        return True

    except Exception as e:
        logger.error(f"SQLite update error: {str(e)}")
        return False


def get_admissions_by_hospital(hospital_id: str) -> list:
    """Fetch admissions for a specific hospital."""
    try:
        conn = get_db()
        cursor = conn.execute(
            "SELECT * FROM admissions WHERE hospital_id = ? ORDER BY admit_time DESC",
            (hospital_id,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    except Exception as e:
        logger.error(f"SQLite hospital fetch error: {str(e)}")
        return []


def get_admissions_by_date_range(start_date: str, end_date: str) -> list:
    """Fetch admissions within a date range."""
    try:
        conn = get_db()
        cursor = conn.execute(
            "SELECT * FROM admissions WHERE DATE(admit_time) BETWEEN ? AND ? ORDER BY admit_time DESC",
            (start_date, end_date)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    except Exception as e:
        logger.error(f"SQLite date range fetch error: {str(e)}")
        return []
