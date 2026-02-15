from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Body
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from passlib.context import CryptContext
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import uvicorn
from itertools import groupby
import glob
import os
from sklearn.linear_model import LinearRegression

# --- FASTAPI SETUP ---
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- DATABASE SETUP (SQLAlchemy) ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./hospital_data.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ML MODEL LOADING ---
model = load_model('hospital_stock_model.h5', compile=False)
model.compile(optimizer='adam', loss='mse')
scaler_feat = joblib.load('feature_scaler.pkl')
scaler_targ = joblib.load('target_scaler.pkl')
le = joblib.load('label_encoder.pkl')

staff_model = load_model('augmented_hospital_model.h5')
staff_scaler = joblib.load('scaler_staff.joblib')
staff_feature_cols = joblib.load('feature_columns_staff.joblib')
staff_class_names = joblib.load('classes_staff.joblib')


class PredictionRequest(BaseModel):
    date: str
    itemName: str
    currentStock: float
    orgCode: str

# --- MEDICINE PAGE LOGIC (CSV + Linear Regression) ---

def get_medicine_forecast():
    """Reads inv_*.csv files and performs linear regression for medicine forecasting."""
    files = sorted(glob.glob('inv_*.csv'))
    if not files:
        return {}
    
    df_list = [pd.read_csv(f) for f in files]
    full_df = pd.concat(df_list)
    item_ids = full_df['Item_ID'].unique()
    months_x = np.array(range(1, len(files) + 1)).reshape(-1, 1)
    
    forecast_results = {}

    for item in item_ids:
        y_usage = []
        for df in df_list:
            # Filter safely
            val = df[df['Item_ID'] == item]['Used_This_Month'].values
            y_usage.append(val[0] if len(val) > 0 else 0)
        
        # Linear Regression
        if len(y_usage) > 0:
            reg_model = LinearRegression().fit(months_x, np.array(y_usage))
            pred = max(0, round(reg_model.predict([[len(files) + 1]])[0]))
            
            forecast_results[str(item)] = {
                "min": int(min(y_usage)),
                "max": int(max(y_usage)),
                "pred": int(pred)
            }
    return forecast_results

def load_startup_data():
    """Initializes Global Variables for Medicine Page."""
    files = sorted(glob.glob('inv_*.csv'))
    latest_file = files[-1] if files else None
    
    stats_cache = get_medicine_forecast()
    inventory_df = pd.read_csv(latest_file) if latest_file else pd.DataFrame()
    
    return stats_cache, inventory_df, latest_file

# Initialize Global Data for Medicine Page
STATS_CACHE, CURRENT_DF, SOURCE = load_startup_data()

# --- GENERAL ROUTES ---

@app.get("/")
async def landing_page(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/signin")
async def signin_page(request: Request):
    return templates.TemplateResponse("signin.html", {"request": request})

@app.post("/signin")
async def register_user(
    medical_org: str = Form(...),
    org_code: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db_session: Session = Depends(get_db)
):
    hashed_pwd = pwd_context.hash(password)
    new_user = User(
        medical_org=medical_org,
        org_code=org_code,
        phone=phone,
        email=email,
        hashed_password=hashed_pwd
    )
    db_session.add(new_user)
    db_session.commit()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_user(org_code: str = Form(...), email: str = Form(...), password: str = Form(...), db_session: Session = Depends(get_db)):
    user = db_session.query(User).filter(User.email == email).first()
    if not user or not pwd_context.verify(password, user.hashed_password) or user.org_code != org_code:
        return templates.TemplateResponse("login.html", {"request": {}, "error": "Invalid Credentials"})
    
    return RedirectResponse(url=f"/inventory?org_code={org_code}", status_code=status.HTTP_302_FOUND)

@app.get("/home")
async def home_page(request: Request, org_code: str = "DEFAULT"):
    return templates.TemplateResponse("home.html", {"request": request, "active_page": "home", "org_code": org_code})

# --- STAFF PREDICTION LOGIC ---
def get_staff_prediction(diag, proc, room, bed_days):
    try:
        # Create initial DataFrame for a single row
        input_df = pd.DataFrame([{
            'Primary_Diagnosis': diag,
            'Procedure_Performed': proc,
            'Room_Type': room,
            'Bed_Days': bed_days
        }])

        # 1. Feature Engineering (Using int() for scalar boolean results)
        complexity_scores = {'Appendicitis': 5, 'Fracture': 4, 'Pneumonia': 3, 'Diabetes': 2}
        procedure_scores = {'Appendectomy': 5, 'MRI': 4, 'Chest X-ray': 2, 'Blood Test': 1}
        
        diag_comp = complexity_scores.get(diag, 0)
        proc_comp = procedure_scores.get(proc, 0)
        total_comp = diag_comp + proc_comp
        
        # Assign values to the dataframe
        input_df['Diag_Complexity'] = diag_comp
        input_df['Proc_Complexity'] = proc_comp
        input_df['Total_Complexity'] = total_comp
        input_df['Bed_Days_Norm'] = (bed_days - 0) / (30 - 0)
        input_df['Bed_Days_Log'] = np.log1p(bed_days)
        input_df['Bed_Days_Cat'] = 0
        
        # FIX: Use int() instead of .astype(int) for scalar comparisons
        input_df['Surgery_ICU'] = int(proc == 'Appendectomy' and room == 'ICU')
        input_df['Emergency_Case'] = int(diag in ['Appendicitis', 'Fracture'])
        input_df['ICU_Case'] = int(room == 'ICU')
        input_df['Complex_Case'] = int(total_comp > 7)
        
        input_df['Rule_2_Surgeons'] = 0
        input_df['Rule_Nurse_Doctor'] = 0
        input_df['Rule_Nurse_Only'] = 0
        input_df['Complexity_Bed_Ratio'] = total_comp / (bed_days + 1)
        input_df['Risk_Score'] = (diag_comp * 0.4 + proc_comp * 0.3 + 
                                  input_df['ICU_Case'] * 3 + input_df['Bed_Days_Norm'] * 2)

        # 2. Re-index / One-Hot Encoding Alignment
        for col in staff_feature_cols:
            if col not in input_df.columns:
                if col.startswith('Diag_') and diag in col: input_df[col] = 1
                elif col.startswith('Proc_') and proc in col: input_df[col] = 1
                elif col.startswith('Room_') and room in col: input_df[col] = 1
                else: input_df[col] = 0
        
        # Match training column order
        final_input = input_df[staff_feature_cols]
        final_scaled = staff_scaler.transform(final_input)
        
        # 3. Predict
        prediction_probs = staff_model.predict(final_scaled, verbose=0)
        return staff_class_names[np.argmax(prediction_probs)]
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return "General Staff (Model Error)"
    
# --- VIEW ROUTE ---
# 1. GET DATA ROUTE (For populating the popups)
@app.get("/get_patient_details/{patient_id}")
async def get_patient_details(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Staff).filter(Staff.id == patient_id).first()
    if not patient:
        return JSONResponse(status_code=404, content={"message": "Patient not found"})
    
    # Return raw data as JSON so the Javascript can read it
    return {
        "id": patient.id,
        "patient_name": patient.patient_name,
        "diagnosis": patient.diagnosis,
        "procedure": patient.procedure,
        "room_type": patient.room_type,
        "bed_type": patient.bed_type,
        "bed_days": patient.bed_days,
        "predicted_staff": patient.predicted_staff,
        "admission_date": str(patient.date), # Convert date to string
        "org_code": patient.org_code
    }

# 2. EDIT ROUTE (POST - Updates the DB and refreshes page)
@app.post("/update_patient")
async def update_patient(
    patient_id: int = Form(...), # Hidden field in the popup
    patient_name: str = Form(...),
    diagnosis: str = Form(...),
    procedure: str = Form(...),
    room_type: str = Form(...),
    bed_days: float = Form(...),
    db: Session = Depends(get_db)
):
    patient = db.query(Staff).filter(Staff.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Update logic
    patient.patient_name = patient_name
    patient.diagnosis = diagnosis
    patient.procedure = procedure
    patient.room_type = room_type
    patient.bed_days = bed_days
    # patient.predicted_staff = ... (Recalculate your AI prediction here if needed)

    db.commit()
    return RedirectResponse(url=f"/staff?org_code={patient.org_code}", status_code=303)
# --- DELETE ROUTE ---
@app.post("/delete_patient/{patient_id}")
async def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Staff).filter(Staff.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    org_code = patient.org_code
    db.delete(patient)
    db.commit()
    
    # Redirect back to the main table
    return RedirectResponse(url=f"/staff?org_code={org_code}", status_code=303)
@app.get("/staff")
async def staff_page(request: Request, org_code: str = "DEFAULT", db_session: Session = Depends(get_db)):
    staff_records = db_session.query(Staff).filter(Staff.org_code == org_code).all()
    return templates.TemplateResponse("staff.html", {
        "request": request,
        "active_page": "staff",
        "org_code": org_code,
        "staff_list": staff_records
    })
    
@app.post("/add_patient")
async def add_patient(
    org_code: str = Form(...),
    admission_date: str = Form(...),
    patient_name: str = Form(...),
    diagnosis: str = Form(...),
    procedure: str = Form(...),
    room_type: str = Form(...),
    bed_type: str = Form(...),
    bed_days: float = Form(...),
    db_session: Session = Depends(get_db)
):
    # Calculate Prediction
    prediction = get_staff_prediction(diagnosis, procedure, room_type, bed_days)
    
    new_patient = Staff(
        org_code=org_code,
        date=admission_date,
        patient_name=patient_name,
        diagnosis=diagnosis,
        procedure=procedure,
        room_type=room_type,
        bed_type=bed_type,
        bed_days=bed_days,
        predicted_staff=prediction
    )
    db_session.add(new_patient)
    db_session.commit()
    return RedirectResponse(url=f"/staff?org_code={org_code}", status_code=303)

# --- INVENTORY PAGE ROUTES ---

@app.get("/inventory")
async def inventory_page(request: Request, org_code: str = "DEFAULT", db_session: Session = Depends(get_db)):
    subquery = db_session.query(
                Inventory.item_name,
                func.max(Inventory.id).label('max_id')
            ).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()

    saved_data = db_session.query(Inventory).join(
        subquery, Inventory.id == subquery.c.max_id
    ).all()
    
    if not saved_data:
        display_data = [
            {"date": "2026-01-01", "item_name": "Surgical Masks", "current_stock": 100, "predicted_min": "-", "predicted_max": "-"},
            {"date": "2026-01-01", "item_name": "IV Fluids", "current_stock": 50, "predicted_min": "-", "predicted_max": "-"}
        ]
    else:
        display_data = saved_data

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "data": display_data, 
        "org_code": org_code,
        "active_page": "inventory"
    })

@app.post("/predict")
async def predict(data: PredictionRequest, db_session: Session = Depends(get_db)):
    try:
        dt = pd.to_datetime(data.date)
        month_sin = np.sin(2 * np.pi * dt.month / 12)
        month_cos = np.cos(2 * np.pi * dt.month / 12)
        name_enc = le.transform([data.itemName])[0]
        
        feats = np.array([[name_enc, data.currentStock, month_sin, month_cos, dt.year]])
        feats_scaled = scaler_feat.transform(feats)
        seq = np.repeat(feats_scaled[np.newaxis, :], 4, axis=1)
        
        pred_scaled = model.predict(seq, verbose=0)
        pred = scaler_targ.inverse_transform(pred_scaled)
        
        min_v = round(float(pred[0][0]), 2)
        max_v = round(float(pred[0][1]), 2)

        new_entry = Inventory(
            org_code=data.orgCode,
            date=data.date,
            item_name=data.itemName,
            current_stock=data.currentStock,
            predicted_min=min_v,
            predicted_max=max_v
        )
        db_session.add(new_entry)
        db_session.commit()

        return {"status": "success", "min_req": min_v, "max_req": max_v}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- ALERT PAGE ROUTES ---

@app.get("/alerts")
async def alerts_page(request: Request, org_code: str = "DEFAULT", db_session: Session = Depends(get_db)):
    subquery = db_session.query(
        Inventory.item_name,
        func.max(Inventory.id).label('max_id')
    ).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()

    latest_inventory = db_session.query(Inventory).join(
        subquery, Inventory.id == subquery.c.max_id
    ).all()

    alert_items = []
    for item in latest_inventory:
        if item.predicted_min is not None and item.current_stock < item.predicted_min:
            item.deficit = round(item.predicted_min - item.current_stock, 1)
            alert_items.append(item)
    
    alert_items.sort(key=lambda x: x.date, reverse=True)

    grouped_alerts = {}
    for date, items in groupby(alert_items, key=lambda x: x.date):
        grouped_alerts[date] = list(items)

    return templates.TemplateResponse("alert.html", {
        "request": request, 
        "active_page": "alerts", 
        "org_code": org_code,
        "grouped_alerts": grouped_alerts,
        "total_alerts": len(alert_items),
        "alerts": alert_items
    })

@app.post("/resolve_alert")
async def resolve_alert(
    item_name: str = Body(..., embed=True), 
    org_code: str = Body(..., embed=True),
    db_session: Session = Depends(get_db)
):
    subquery = db_session.query(
        Inventory.item_name,
        func.max(Inventory.id).label('max_id')
    ).filter(Inventory.org_code == org_code, Inventory.item_name == item_name).group_by(Inventory.item_name).subquery()

    item = db_session.query(Inventory).join(
        subquery, Inventory.id == subquery.c.max_id
    ).first()

    if item:
        new_entry = Inventory(
            org_code=item.org_code,
            date=item.date,
            item_name=item.item_name,
            current_stock=item.predicted_min,
            predicted_min=item.predicted_min,
            predicted_max=item.predicted_max
        )
        db_session.add(new_entry)
        db_session.commit()
        return {"status": "success", "message": "Item restocked"}
    
    return {"status": "error", "message": "Item not found"}

# --- MEDICINE PAGE SPECIFIC ROUTES ---

@app.get("/medicine")
async def medicine_page(request: Request, org_code: str = "DEFAULT"):
    """
    Displays the Medicine Page.
    NOTE: The CSV data logic (CURRENT_DF) is separate from SQL logic for now.
    """
    medicines = []
    if not CURRENT_DF.empty:
        # Convert NaN to None for Jinja safety
        medicines = CURRENT_DF.replace({np.nan: None}).to_dict(orient='records')
        
    return templates.TemplateResponse("medicine.html", {
        "request": request,
        "active_page": "medicine", 
        "org_code": org_code,
        "medicines": medicines,
        "source": SOURCE
    })

@app.post("/api/update_stock")
async def update_stock(item_id: str = Form(...), new_stock: int = Form(...)):
    global CURRENT_DF, SOURCE
    try:
        if not SOURCE or not os.path.exists(SOURCE):
             return JSONResponse(status_code=404, content={"message": "No CSV source file found"})

        df = pd.read_csv(SOURCE)
        mask = (df['Item_ID'].astype(str) == str(item_id))
        
        if not mask.any():
            return JSONResponse(status_code=404, content={"message": "Item ID not found in CSV"})
            
        df.loc[mask, 'Current_Stock'] = new_stock
        df.to_csv(SOURCE, index=False)
        CURRENT_DF = df
        
        return {"message": f"Successfully updated {item_id} to {new_stock}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/api/forecast/{item_id}")
async def get_forecast(item_id: str):
    data = STATS_CACHE.get(item_id) or STATS_CACHE.get(str(int(item_id)) if item_id.isdigit() else None)
    if data:
        return data
    return JSONResponse(status_code=404, content={"message": "Forecast data not found"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)