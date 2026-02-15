import os
import glob
import joblib
import uvicorn
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import groupby

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Body
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model

# --- 1. SETUP & DATABASE ---
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SQLALCHEMY_DATABASE_URL = "sqlite:///./hospital_data.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- 2. MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    medical_org = Column(String)
    org_code = Column(String, unique=True, index=True)
    phone = Column(String)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Inventory(Base):
    __tablename__ = "inventory_data"
    id = Column(Integer, primary_key=True, index=True)
    org_code = Column(String, ForeignKey("users.org_code"))
    date = Column(String)
    item_name = Column(String)
    current_stock = Column(Float)
    predicted_min = Column(Float)
    predicted_max = Column(Float)

class Staff(Base):
    __tablename__ = "staff_data"
    id = Column(Integer, primary_key=True, index=True)
    org_code = Column(String, ForeignKey("users.org_code"))
    date = Column(String)
    patient_name = Column(String)
    diagnosis = Column(String)
    procedure = Column(String)
    room_type = Column(String)
    bed_type = Column(String)
    bed_days = Column(Float)
    predicted_staff = Column(String)
    

class AlertResolution(BaseModel):
    item_name: str
    org_code: str
    item_type: str

Base.metadata.create_all(bind=engine)

class PredictionRequest(BaseModel):
    date: str
    itemName: str
    currentStock: float
    orgCode: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_medicine_forecast():
    files = sorted(glob.glob('inv_*.csv'))
    if not files: return {}
    try:
        df_list = [pd.read_csv(f) for f in files]
        full_df = pd.concat(df_list)
        # Ensure Item_ID exists
        if 'Item_ID' not in full_df.columns: return {}
        
        item_ids = full_df['Item_ID'].unique()
        months_x = np.array(range(1, len(files) + 1)).reshape(-1, 1)
        
        forecast_results = {}
        for item in item_ids:
            y_usage = []
            for df in df_list:
                val = df[df['Item_ID'] == item]['Used_This_Month'].values
                y_usage.append(val[0] if len(val) > 0 else 0)
            if y_usage:
                reg_model = LinearRegression().fit(months_x, np.array(y_usage))
                pred = max(0, round(reg_model.predict([[len(files) + 1]])[0]))
                forecast_results[str(item)] = {
                    "min": int(min(y_usage)),
                    "max": int(max(y_usage)),
                    "pred": int(pred)
                }
        return forecast_results
    except: return {}

def get_medicine_alerts():
    global CURRENT_DF, STATS_CACHE
    medicine_alerts = []
    if CURRENT_DF is None or CURRENT_DF.empty: return medicine_alerts

    for _, row in CURRENT_DF.iterrows():
        item_id = str(row['Item_ID'])
        
        # FIX: Look for 'Medicine_Name' specifically based on your CSV columns
        item_name = row.get('Medicine_Name') or row.get('Item_Name') or f"Medicine {item_id}"
        
        current_stock = row['Current_Stock']
        forecast = STATS_CACHE.get(item_id)
        
        if forecast and current_stock < forecast['min']:
            medicine_alerts.append({
                "item_name": item_name,
                "item_id": item_id,
                "current_stock": current_stock,
                "predicted_min": forecast['min'],
                "deficit": round(forecast['min'] - current_stock, 1),
                "date": "Current Status", 
                "type": "medicine" 
            })
    return medicine_alerts

def load_startup_data():
    files = sorted(glob.glob('inv_*.csv'))
    latest_file = files[-1] if files else None
    
    stats_cache = {}
    inventory_df = pd.DataFrame()
    
    if latest_file:
        try:
            inventory_df = pd.read_csv(latest_file)
            # Clean column names (remove spaces like " Current_Stock ")
            inventory_df.columns = inventory_df.columns.str.strip()
            
            # Load stats
            stats_cache = get_medicine_forecast()
            print(f"SUCCESS: Loaded {latest_file} with columns: {list(inventory_df.columns)}")
        except Exception as e:
            print(f"ERROR: Could not read CSV file. {e}")
            
    return stats_cache, inventory_df, latest_file


STATS_CACHE, CURRENT_DF, SOURCE = load_startup_data()

def get_total_alerts_count(db_session: Session, org_code: str):
    subquery = db_session.query(
        Inventory.item_name, func.max(Inventory.id).label('max_id')
    ).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
    sql_alerts = db_session.query(Inventory).join(subquery, Inventory.id == subquery.c.max_id).all()
    sql_count = sum(1 for item in sql_alerts if item.predicted_min and item.current_stock < item.predicted_min)
    return sql_count + len(get_medicine_alerts())

# --- 3. ML MODEL LOADING ---
# Ensure these files are in your project folder!
stock_model = load_model('hospital_stock_model.h5', compile=False)
scaler_feat = joblib.load('feature_scaler.pkl')
scaler_targ = joblib.load('target_scaler.pkl')
le = joblib.load('label_encoder.pkl')

staff_model = load_model('augmented_hospital_model.h5', compile=False)
staff_scaler = joblib.load('scaler_staff.joblib')
staff_feature_cols = joblib.load('feature_columns_staff.joblib')
staff_class_names = joblib.load('classes_staff.joblib')

# --- 4. LOGIC HELPERS ---

def get_staff_prediction(diag, proc, room, bed_days):
    try:
        input_df = pd.DataFrame([{
            'Primary_Diagnosis': diag, 'Procedure_Performed': proc,
            'Room_Type': room, 'Bed_Days': bed_days
        }])
        
        complexity_scores = {'Appendicitis': 5, 'Fracture': 4, 'Pneumonia': 3, 'Diabetes': 2}
        procedure_scores = {'Appendectomy': 5, 'MRI': 4, 'Chest X-ray': 2, 'Blood Test': 1}
        
        diag_comp = complexity_scores.get(diag, 0)
        proc_comp = procedure_scores.get(proc, 0)
        
        input_df['Diag_Complexity'] = diag_comp
        input_df['Proc_Complexity'] = proc_comp
        input_df['Total_Complexity'] = diag_comp + proc_comp
        input_df['Bed_Days_Norm'] = (bed_days) / 30
        input_df['Bed_Days_Log'] = np.log1p(bed_days)
        input_df['Bed_Days_Cat'] = 0
        input_df['Surgery_ICU'] = int(proc == 'Appendectomy' and room == 'ICU')
        input_df['Emergency_Case'] = int(diag in ['Appendicitis', 'Fracture'])
        input_df['ICU_Case'] = int(room == 'ICU')
        input_df['Complex_Case'] = int((diag_comp + proc_comp) > 7)
        input_df['Rule_2_Surgeons'] = 0
        input_df['Rule_Nurse_Doctor'] = 0
        input_df['Rule_Nurse_Only'] = 0
        input_df['Complexity_Bed_Ratio'] = (diag_comp + proc_comp) / (bed_days + 1)
        input_df['Risk_Score'] = (diag_comp * 0.4 + proc_comp * 0.3 + 3 + 2)

        for col in staff_feature_cols:
            if col not in input_df.columns:
                if col.startswith('Diag_') and diag in col: input_df[col] = 1
                elif col.startswith('Proc_') and proc in col: input_df[col] = 1
                elif col.startswith('Room_') and room in col: input_df[col] = 1
                else: input_df[col] = 0
        
        final_scaled = staff_scaler.transform(input_df[staff_feature_cols])
        return staff_class_names[np.argmax(staff_model.predict(final_scaled, verbose=0))]
    except Exception:
        return "General Medical Team"

# --- 5. ROUTES: AUTH ---

@app.get("/")
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.post("/signin")
async def register(medical_org: str = Form(...), org_code: str = Form(...), phone: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = User(medical_org=medical_org, org_code=org_code, phone=phone, email=email, hashed_password=pwd_context.hash(password))
    db.add(user)
    db.commit()
    return RedirectResponse(url="/login", status_code=302)

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(org_code: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email, User.org_code == org_code).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    return RedirectResponse(url=f"/home?org_code={org_code}", status_code=302)

# --- 6. ROUTES: HOME ---

@app.get("/home")
async def home_page(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    # 1. Fetch Organization Info
    user = db.query(User).filter(User.org_code == org_code).first()
    org_name = user.medical_org if user else "Medical Organization"

    # 2. SQL Inventory Data (Database Items)
    subq = db.query(
        Inventory.item_name, 
        func.max(Inventory.id).label('max_id')
    ).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
    
    inv_items = db.query(Inventory).join(subq, Inventory.id == subq.c.max_id).all()
    inv_labels = [item.item_name for item in inv_items]
    inv_stocks = [item.current_stock for item in inv_items]
    
    # 3. Medicine CSV Data (CSV Items)
    med_labels = []
    med_stocks = []
    
    if CURRENT_DF is not None and not CURRENT_DF.empty:
        # FIX: Explicitly use 'Medicine_Name' based on your CSV columns
        if 'Medicine_Name' in CURRENT_DF.columns:
            med_labels = CURRENT_DF['Medicine_Name'].astype(str).tolist()
        elif 'Item_Name' in CURRENT_DF.columns:
            med_labels = CURRENT_DF['Item_Name'].astype(str).tolist()
        else:
            # Fallback if name is missing
            med_labels = [f"Item {x}" for x in CURRENT_DF['Item_ID'].tolist()]
            
        if 'Current_Stock' in CURRENT_DF.columns:
            med_stocks = CURRENT_DF['Current_Stock'].fillna(0).tolist()

    # 4. Calculate Alerts
    combined_alerts = []
    
    # SQL Alerts
    for item in inv_items:
        if item.predicted_min and item.current_stock < item.predicted_min:
            combined_alerts.append(f"Supply Low: {item.item_name}")
            
    # CSV Alerts
    med_alerts_list = get_medicine_alerts()
    for ma in med_alerts_list:
        combined_alerts.append(f"Medicine Alert: {ma['item_name']}")

    context_data = {
        "request": request,
        "active_page": "dashboard",
        "org_code": org_code,
        "org_name": org_name,
        "inv_labels": inv_labels,
        "inv_stocks": inv_stocks,
        "med_labels": med_labels,
        "med_stocks": med_stocks,
        "alerts": combined_alerts,
        "alerts_count": len(combined_alerts),
        "stats": {
            "total_items": len(inv_items) + len(med_labels),
            "low_stock": len(combined_alerts),
            "total_patients": db.query(Staff).filter(Staff.org_code == org_code).count()
        }
    }

    return templates.TemplateResponse("home.html", context_data)
# --- 7. ROUTES: INVENTORY & STAFF ---

@app.get("/inventory")
async def inventory_page(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    subq = db.query(Inventory.item_name, func.max(Inventory.id).label('mid')).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
    data = db.query(Inventory).join(subq, Inventory.id == subq.c.mid).all()
    return templates.TemplateResponse("index.html", {"request": request, "data": data, "org_code": org_code, "active_page": "inventory"})

@app.post("/predict")
async def predict_stock(data: PredictionRequest, db: Session = Depends(get_db)):
    dt = pd.to_datetime(data.date)
    name_enc = le.transform([data.itemName])[0]
    feats = scaler_feat.transform([[name_enc, data.currentStock, np.sin(2*np.pi*dt.month/12), np.cos(2*np.pi*dt.month/12), dt.year]])
    seq = np.repeat(feats[np.newaxis, :], 4, axis=1)
    pred = scaler_targ.inverse_transform(stock_model.predict(seq, verbose=0))
    
    entry = Inventory(org_code=data.orgCode, date=data.date, item_name=data.itemName, current_stock=data.currentStock, predicted_min=round(float(pred[0][0]), 2), predicted_max=round(float(pred[0][1]), 2))
    db.add(entry)
    db.commit()
    return {"status": "success", "min_req": entry.predicted_min, "max_req": entry.predicted_max}

@app.get("/staff")
async def staff_page(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    records = db.query(Staff).filter(Staff.org_code == org_code).all()
    return templates.TemplateResponse("staff.html", {"request": request, "staff_list": records, "org_code": org_code, "active_page": "staff"})

@app.post("/add_patient")
async def add_patient(org_code: str = Form(...), admission_date: str = Form(...), patient_name: str = Form(...), diagnosis: str = Form(...), procedure: str = Form(...), room_type: str = Form(...), bed_type: str = Form(...), bed_days: float = Form(...), db: Session = Depends(get_db)):
    pred = get_staff_prediction(diagnosis, procedure, room_type, bed_days)
    new_p = Staff(org_code=org_code, date=admission_date, patient_name=patient_name, diagnosis=diagnosis, procedure=procedure, room_type=room_type, bed_type=bed_type, bed_days=bed_days, predicted_staff=pred)
    db.add(new_p)
    db.commit()
    return RedirectResponse(url=f"/staff?org_code={org_code}", status_code=303)

@app.get("/medicine")
async def medicine_page(request: Request, org_code: str = "DEFAULT", db_session: Session = Depends(get_db)):
    meds = CURRENT_DF.replace({np.nan: None}).to_dict(orient='records') if not CURRENT_DF.empty else []
    return templates.TemplateResponse("medicine.html", {"request": request, "active_page": "medicine", "org_code": org_code, "medicines": meds, "source": SOURCE, "alerts_count": get_total_alerts_count(db_session, org_code)})

@app.get("/alerts")
async def alerts(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    combined_alerts = []

    # 1. GET SQL ALERTS (Inventory)
    subq = db.query(Inventory.item_name, func.max(Inventory.id).label('mid')).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
    sql_items = db.query(Inventory).join(subq, Inventory.id == subq.c.mid).all()
    
    for item in sql_items:
        if item.predicted_min and item.current_stock < item.predicted_min:
            combined_alerts.append({
                "item_name": item.item_name,
                "current_stock": item.current_stock,
                "predicted_min": item.predicted_min,
                "deficit": round(item.predicted_min - item.current_stock, 1),
                "type": "inventory", # Critical for filtering
                "date": item.date
            })

    # 2. GET CSV ALERTS (Medicine)
    med_alerts = get_medicine_alerts() # This returns a list of dicts
    combined_alerts.extend(med_alerts)

    # 3. RENDER TEMPLATE
    return templates.TemplateResponse("alert.html", {
        "request": request, 
        "alerts": combined_alerts, 
        "alerts_count": len(combined_alerts),
        "org_code": org_code, 
        "active_page": "alerts"
    })

# --- ADD THIS NEW ROUTE FOR THE BUTTONS TO WORK ---
@app.post("/resolve_alert")
async def resolve_alert(data: AlertResolution, db: Session = Depends(get_db)):
    try:
        if data.item_type == "inventory":
            # Find the item and update its stock to the predicted max (Restock)
            # In a real app, you would create a new entry with higher stock
            subq = db.query(Inventory.item_name, func.max(Inventory.id).label('mid')).filter(Inventory.org_code == data.org_code, Inventory.item_name == data.item_name).subquery()
            latest = db.query(Inventory).join(subq, Inventory.id == subq.c.mid).first()
            
            if latest:
                # Restock logic: Add a new entry with safe stock level
                safe_stock = latest.predicted_max if latest.predicted_max > 0 else latest.predicted_min + 50
                new_entry = Inventory(
                    org_code=data.org_code,
                    date="Restocked",
                    item_name=data.item_name,
                    current_stock=safe_stock,
                    predicted_min=latest.predicted_min,
                    predicted_max=latest.predicted_max
                )
                db.add(new_entry)
                db.commit()
                return {"status": "success", "message": f"Restocked {data.item_name}"}

        elif data.item_type == "medicine":
            global CURRENT_DF
            if CURRENT_DF is not None:
                mask = (CURRENT_DF['Medicine_Name'] == data.item_name) | (CURRENT_DF['Item_Name'] == data.item_name)
                if mask.any():

                    CURRENT_DF.loc[mask, 'Current_Stock'] += 100 
            return {"status": "success", "message": "Medicine order placed"}

        return {"status": "error", "message": "Item not found"}
    except Exception as e:
        print(f"Error resolving alert: {e}")
        return {"status": "error", "message": str(e)}
    
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)