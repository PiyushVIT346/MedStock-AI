import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the Database URL and the Inventory model from your app3.py file
try:
    from app3 import Inventory, SQLALCHEMY_DATABASE_URL
except ImportError:
    print("Error: Could not find app3.py. Ensure this script is in the same folder.")
    exit()

# --- 1. DATABASE CONNECTION ---
# We create a new engine instance specifically for this script
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def view_inventory(org_code: str):
    """
    Reads and displays the current inventory table data.
    """
    db = SessionLocal()
    try:
        print(f"\n--- Current Inventory for {org_code} ---")
        items = db.query(Inventory).filter(Inventory.org_code == org_code).all()
        
        if not items:
            print("No data found in the inventory table.")
            return
        
        # Header
        print(f"{'ID':<5} | {'Item Name':<20} | {'Stock':<10} | {'Date':<15}")
        print("-" * 55)
        
        for item in items:
            print(f"{item.id:<5} | {item.item_name:<20} | {item.current_stock:<10} | {item.date:<15}")
            
    except Exception as e:
        print(f"Error reading data: {e}")
    finally:
        db.close()

def delete_example_items(org_code: str):
    """
    Removes all rows where item_name is 'Example Item'.
    """
    db = SessionLocal()
    target_name = 'Example Item'
    try:
        print(f"\nSearching for '{target_name}' to remove...")
        
        # Perform the deletion
        query = db.query(Inventory).filter(
            Inventory.org_code == org_code,
            Inventory.item_name == target_name
        )
        
        count = query.count()
        if count == 0:
            print(f"No items named '{target_name}' found for organization {org_code}.")
            return

        query.delete(synchronize_session=False)
        db.commit()
        print(f"SUCCESS: Successfully deleted {count} instances of '{target_name}'.")
        
    except Exception as e:
        db.rollback()
        print(f"FAILED: An error occurred during deletion: {e}")
    finally:
        db.close()

# --- 2. EXECUTION ---
if __name__ == "__main__":
    # You can change this to match the org_code you used in your app
    TARGET_ORG = "DEFAULT" 

    print("Step 1: Reading initial data...")
    view_inventory(TARGET_ORG)

    print("\nStep 2: Cleaning up data...")
    delete_example_items(TARGET_ORG)

    print("\nStep 3: Verifying changes...")
    view_inventory(TARGET_ORG)