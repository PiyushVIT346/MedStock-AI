import pandas as pd
from sqlalchemy import create_engine
import os

# 1. Connect to the database
# Ensure the filename matches exactly what is in your app.py
DB_FILE = "hospital_data.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

if not os.path.exists(DB_FILE):
    print(f"❌ Error: Database file '{DB_FILE}' not found.")
    exit()

engine = create_engine(DATABASE_URL)

def view_database():
    print(f"📊 Reading data from: {DB_FILE}\n")

    # --- VIEW USERS TABLE ---
    print("="*50)
    print(f"👤 TABLE: users")
    print("="*50)
    try:
        df_users = pd.read_sql("SELECT * FROM users", engine)
        if df_users.empty:
            print("⚠️  Table is empty.")
        else:
            # Drop hashed_password for security when viewing
            if 'hashed_password' in df_users.columns:
                display_users = df_users.drop(columns=['hashed_password'])
            else:
                display_users = df_users
            print(display_users.to_string(index=False))
    except Exception as e:
        print(f"❌ Error reading 'users' table: {e}")

    print("\n" + "="*50)
    print(f"📦 TABLE: inventory_data")
    print("="*50)
    
    # --- VIEW INVENTORY TABLE ---
    try:
        df_inventory = pd.read_sql("SELECT * FROM inventory_data", engine)
        if df_inventory.empty:
            print("⚠️  Table is empty.")
        else:
            print(df_inventory.to_string(index=False))
    except Exception as e:
        print(f"❌ Error reading 'inventory_data' table: {e}")

if __name__ == "__main__":
    view_database()