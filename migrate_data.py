import sqlalchemy
from app3 import Base, engine, SessionLocal, User, Inventory, Staff

# 1. Initialize Neon Schema (Creates the tables in Neon)
print("Creating tables in Neon...")
Base.metadata.create_all(bind=engine)

# 2. Connect to local SQLite
sqlite_engine = sqlalchemy.create_engine("sqlite:///./hospital_data.db")
SqliteSession = sqlalchemy.orm.sessionmaker(bind=sqlite_engine)
sqlite_db = SqliteSession()

# 3. Connect to Neon
neon_db = SessionLocal()

def migrate_table(model):
    print(f"Migrating {model.__tablename__}...")
    items = sqlite_db.query(model).all()
    for item in items:
        # We clear the state to prevent SQLAlchemy from thinking it's still attached to SQLite
        sqlalchemy.orm.make_transient(item) 
        neon_db.add(item)
    neon_db.commit()

try:
    migrate_table(User)
    migrate_table(Staff)
    migrate_table(Inventory)
    print("✅ Migration Successful!")
except Exception as e:
    print(f"❌ Error: {e}")
    neon_db.rollback()
finally:
    sqlite_db.close()
    neon_db.close()