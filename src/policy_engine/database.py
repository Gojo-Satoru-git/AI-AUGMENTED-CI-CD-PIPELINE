import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "trust_tiers.db")

def init_db():
    """Initializes the SQLite database with the developer_trust table and seed data."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS developer_trust (
            developer_id TEXT PRIMARY KEY,
            trust_tier INTEGER NOT NULL
        )
    """)
    
    # Insert seed data if table is empty
    cursor.execute("SELECT COUNT(*) FROM developer_trust")
    if cursor.fetchone()[0] == 0:
        seed_data = [
            ("dev_junior_1", 1), # Junior Developer (Tier 1)
            ("dev_senior_3", 3), # Senior Developer (Tier 3)
            ("dev_mid_2", 2)     # Mid Developer (Tier 2)
        ]
        cursor.executemany(
            "INSERT INTO developer_trust (developer_id, trust_tier) VALUES (?, ?)", 
            seed_data
        )
        print("Database initialized with seed data.")
    else:
        print("Database already initialized.")
        
    conn.commit()
    conn.close()

def get_developer_tier(developer_id: str) -> int:
    """Retrieves the trust tier for a given developer_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT trust_tier FROM developer_trust WHERE developer_id = ?", 
        (developer_id,)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    return 1  # Default to tier 1 (least trust) if not found

if __name__ == "__main__":
    init_db()
