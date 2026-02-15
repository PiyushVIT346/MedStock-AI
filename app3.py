import os
import glob
import joblib
import uvicorn
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import groupby
from datetime import datetime
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import google.generativeai as genai
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
from dotenv import load_dotenv

# --- 1. SETUP & DATABASE ---
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
load_dotenv()

#SQLALCHEMY_DATABASE_URL = "sqlite:///./hospital_data.db"
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA0C-um2whNz-W2Rz1oaoYKScl4W9PkZp8")
genai.configure(api_key=GEMINI_API_KEY)

# ---MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    medical_org = Column(String)
    org_code = Column(String, unique=True, index=True)
    phone = Column(String)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

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

class PredictionRequest(BaseModel):
    date: str
    itemName: str
    currentStock: float
    orgCode: str

class ChatbotRequest(BaseModel):
    question: str
    org_code: str
    
class InventoryAddRequest(BaseModel):
    org_code: str
    date: str
    item_name: str
    current_stock: float

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



app = FastAPI()

def get_latest_outbreaks_data():
    api_url = "https://www.who.int/api/news/diseaseoutbreaknews"
    params = {
        "$top": 5,
        "$orderby": "PublicationDateAndTime desc",
        "$select": "Title,PublicationDateAndTime,ItemDefaultUrl"
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    treatment_map = {
        "Nipah": "No vaccine; supportive care.",
        "Marburg": "No licensed vaccine; rehydration.",
        "Influenza": "Antivirals (e.g. Oseltamivir).",
        "Mpox": "Vaccine (MVA-BN); antivirals.",
        "Cholera": "ORS and vaccines.",
        "Measles": "MMR Vaccine; Vitamin A.",
        "Dengue": "Symptomatic care; avoid Aspirin."
    }

    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        articles = data.get('value', [])
        
        results = []
        for item in articles:
            full_title = item.get('Title', '')
            raw_date = item.get('PublicationDateAndTime', '')
            formatted_date = datetime.fromisoformat(raw_date.replace('Z', '')).strftime('%d %B %Y')
            link = item.get('ItemDefaultUrl', '#')
            if link.startswith('/'): link = f"https://www.who.int{link}"

            if " - " in full_title:
                disease, country = full_title.rsplit(" - ", 1)
            else:
                disease, country = full_title, "Global"
            cure = next((v for k, v in treatment_map.items() if k.lower() in disease.lower()), "Supportive care.")

            results.append({
                "Date": formatted_date,
                "Country": country.strip(),
                "Disease": disease.strip(),
                "Cure": cure,
                "Link": link
            })
            
        return {"status": "success", "outbreaks": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/who_outbreaks")
async def api_outbreaks():
    return get_latest_outbreaks_data()

# --- GEMINI AI CHATBOT ---
def get_database_context(org_code: str, db: Session):
    """
    Gathers comprehensive context from all databases for the AI
    """
    context = {
        'inventory': [],
        'medicines': [],
        'patients': [],
        'staff_predictions': []
    }
    
    try:
        subq = db.query(
            Inventory.item_name, 
            func.max(Inventory.id).label('max_id')
        ).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
        
        inv_items = db.query(Inventory).join(subq, Inventory.id == subq.c.max_id).all()
        context['inventory'] = [
            {
                'name': item.item_name,
                'stock': item.current_stock,
                'min_required': item.predicted_min,
                'max_required': item.predicted_max,
                'status': 'low' if item.current_stock < item.predicted_min else 'optimal'
            }
            for item in inv_items
        ]
        
        global CURRENT_DF
        if CURRENT_DF is not None and not CURRENT_DF.empty:
            for _, row in CURRENT_DF.iterrows():
                med_name = row.get('Medicine_Name') or row.get('Item_Name') or f"Medicine {row['Item_ID']}"
                context['medicines'].append({
                    'name': med_name,
                    'stock': row['Current_Stock'],
                    'vendor': row.get('Vendor_Name', 'Unknown'),
                    'monthly_usage': row.get('Used_This_Month', 0)
                })
        staff_records = db.query(Staff).filter(Staff.org_code == org_code).all()
        context['patients'] = [
            {
                'name': s.patient_name,
                'diagnosis': s.diagnosis,
                'procedure': s.procedure,
                'room_type': s.room_type,
                'bed_days': s.bed_days,
                'staff_needed': s.predicted_staff
            }
            for s in staff_records
        ]
        
    except Exception as e:
        print(f"Error gathering database context: {e}")
    
    return context

def query_gemini_ai(question: str, context: dict):
    """
    Queries Gemini AI with the question and database context
    """
    try:
        prompt = f"""You are a medical AI assistant for a hospital inventory management system. 
You have access to the following data:

INVENTORY ITEMS ({len(context['inventory'])} items):
{chr(10).join([f"- {item['name']}: {item['stock']} units (Status: {item['status']}, Min Required: {item['min_required']})" for item in context['inventory'][:20]])}

MEDICINES ({len(context['medicines'])} items):
{chr(10).join([f"- {med['name']}: {med['stock']} units in stock, {med['monthly_usage']} used monthly, Vendor: {med['vendor']}" for med in context['medicines'][:20]])}

PATIENTS ({len(context['patients'])} active):
{chr(10).join([f"- Patient name: {pat['name']}, Diagnosis: {pat['diagnosis']}, Procedure: {pat['procedure']}, Room: {pat['room_type']}, Staff Needed: {pat['staff_needed']}" for pat in context['patients'][:20]])}

QUESTION: {question}

Provide a detailed, data-driven answer based on the information above. If the question asks about relationships between data (like why a medicine is in high demand), analyze the patient data to find correlations. Be specific with numbers and insights.

Answer:"""

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        print(f"Error querying Gemini AI: {e}")
        return f"I apologize, but I encountered an error processing your question. Error details: {str(e)}"


def get_medicine_alerts():
    global CURRENT_DF, STATS_CACHE
    medicine_alerts = []
    if CURRENT_DF is None or CURRENT_DF.empty: return medicine_alerts

    for _, row in CURRENT_DF.iterrows():
        item_id = str(row['Item_ID'])
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

def get_medicine_forecast():
    files = sorted(glob.glob('inv_*.csv'))
    if not files: return {}
    try:
        df_list = [pd.read_csv(f) for f in files]
        full_df = pd.concat(df_list)
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
    
def load_startup_data():
    files = sorted(glob.glob('inv_*.csv'))
    latest_file = files[-1] if files else None
    
    stats_cache = {}
    inventory_df = pd.DataFrame()
    
    if latest_file:
        try:
            inventory_df = pd.read_csv(latest_file)
            inventory_df.columns = inventory_df.columns.str.strip()
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

# --- 6. ML MODEL LOADING ---
stock_model = load_model('hospital_stock_model.h5', compile=False)
scaler_feat = joblib.load('feature_scaler.pkl')
scaler_targ = joblib.load('target_scaler.pkl')
le = joblib.load('label_encoder.pkl')

staff_model = load_model('augmented_hospital_model.h5', compile=False)
staff_scaler = joblib.load('scaler_staff.joblib')
staff_feature_cols = joblib.load('feature_columns_staff.joblib')
staff_class_names = joblib.load('classes_staff.joblib')

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

# --- ROUTES: AUTH ---
@app.get("/")
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/signin")
async def get_signin_page(request: Request):
    return templates.TemplateResponse("signin.html", {"request": request})
@app.post("/signin")
async def register(request:Request,org_type:str=Form(...), medical_org: str = Form(...), org_code: str = Form(...), phone: str = Form(...), 
                email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = User(medical_org=medical_org, org_code=org_code, phone=phone, email=email, 
                hashed_password=pwd_context.hash(password))
    db.add(user)
    db.flush()
    default_items = [
        "Surgical Masks",
        "IV Fluids",
        "Syringes",
        "Antibiotics"
    ]
    for item_name in default_items:
        new_inv = Inventory(
            org_code=org_code,
            date="2026-01-01",
            item_name=item_name,
            current_stock=100.0,
            predicted_min=50.0,
            predicted_max=110.0
        )
        db.add(new_inv)
    db.commit()
    db.refresh(user)
    
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request:Request, org_code: str = Form(...), email: str = Form(...), password: str = Form(...), 
                db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email, User.org_code == org_code).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    return RedirectResponse(url=f"/home?org_code={org_code}", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/", status_code=303)
    return response

# --- ROUTES: MAIN PAGES ---
@app.get("/home")
async def home_page(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    user = db.query(User).filter(User.org_code == org_code).first()
    org_name = user.medical_org if user else "Medical Organization"

    subq = db.query(
        Inventory.item_name, func.max(Inventory.id).label('max_id')
    ).filter(Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
    
    inv_items = db.query(Inventory).join(subq, Inventory.id == subq.c.max_id).all()
    inv_labels = [item.item_name for item in inv_items]
    inv_stocks = [item.current_stock for item in inv_items]
    
    med_labels = []
    med_stocks = []
    
    if CURRENT_DF is not None and not CURRENT_DF.empty:
        if 'Medicine_Name' in CURRENT_DF.columns:
            med_labels = CURRENT_DF['Medicine_Name'].astype(str).tolist()
        elif 'Item_Name' in CURRENT_DF.columns:
            med_labels = CURRENT_DF['Item_Name'].astype(str).tolist()
        else:
            med_labels = [f"Item {x}" for x in CURRENT_DF['Item_ID'].tolist()]
            
        if 'Current_Stock' in CURRENT_DF.columns:
            med_stocks = CURRENT_DF['Current_Stock'].fillna(0).tolist()

    combined_alerts = []
    for item in inv_items:
        if item.predicted_min and item.current_stock < item.predicted_min:
            combined_alerts.append(f"Supply Low: {item.item_name}")
            
    med_alerts_list = get_medicine_alerts()
    for ma in med_alerts_list:
        combined_alerts.append(f"Medicine Alert: {ma['item_name']}")

    return templates.TemplateResponse("home.html", {
        "request": request, "active_page": "dashboard", "org_code": org_code, "org_name": org_name,
        "inv_labels": inv_labels, "inv_stocks": inv_stocks, "med_labels": med_labels,
        "med_stocks": med_stocks, "alerts": combined_alerts, "alerts_count": len(combined_alerts),
        "stats": {
            "total_items": len(inv_items) + len(med_labels),
            "low_stock": len(combined_alerts),
            "total_patients": db.query(Staff).filter(Staff.org_code == org_code).count()
        }
    })

@app.get("/inventory")
async def inventory_page(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    subq = db.query(Inventory.item_name, func.max(Inventory.id).label('mid')).filter(
        Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
    data = db.query(Inventory).join(subq, Inventory.id == subq.c.mid).all()
    return templates.TemplateResponse("index.html", {"request": request, "data": data,
                                                    "org_code": org_code, "active_page": "inventory"})

@app.post("/predict")
async def predict_stock(data: PredictionRequest, db: Session = Depends(get_db)):
    dt = pd.to_datetime(data.date)
    name_enc = le.transform([data.itemName])[0]
    feats = scaler_feat.transform([[name_enc, data.currentStock, np.sin(2*np.pi*dt.month/12), 
                                np.cos(2*np.pi*dt.month/12), dt.year]])
    seq = np.repeat(feats[np.newaxis, :], 4, axis=1)
    pred = scaler_targ.inverse_transform(stock_model.predict(seq, verbose=0))
    
    entry = Inventory(org_code=data.orgCode, date=data.date, item_name=data.itemName, 
                    current_stock=data.currentStock, predicted_min=round(float(pred[0][0]), 2), 
                    predicted_max=round(float(pred[0][1]), 2))
    db.add(entry)
    db.commit()
    return {"status": "success", "min_req": entry.predicted_min, "max_req": entry.predicted_max}

@app.post("/add_inventory_item")
async def add_item(data: InventoryAddRequest, db: Session = Depends(get_db)):
    """
    Manually adds a new item to the inventory without ML prediction.
    """
    try:
        new_item = Inventory(
            org_code=data.org_code,
            date=data.date,
            item_name=data.item_name,
            current_stock=data.current_stock,
            predicted_min=0,
            predicted_max=0
        )
        
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        
        return {"status": "success", "message": f"Added {data.item_name} successfully"}
    except Exception as e:
        print(f"Error adding item: {e}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e)}
        )

@app.get("/staff")
async def staff_page(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    records = db.query(Staff).filter(Staff.org_code == org_code).all()
    return templates.TemplateResponse("staff.html", {"request": request, "staff_list": records, 
                                                    "org_code": org_code, "active_page": "staff"})
@app.get("/get_patient_details/{patient_id}")
async def get_patient_details(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Staff).filter(Staff.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found")
    
    return {
        "id": patient.id,
        "patient_name": patient.patient_name,
        "diagnosis": patient.diagnosis,
        "procedure": patient.procedure,
        "room_type": patient.room_type,
        "bed_days": patient.bed_days,
        "predicted_staff": patient.predicted_staff,
        "admission_date": patient.date
    }

# UPDATE RECORD (For the Edit Form submission)
@app.post("/update_patient")
async def update_patient(
    patient_id: int = Form(...),
    patient_name: str = Form(...),
    diagnosis: str = Form(...),
    procedure: str = Form(...),
    room_type: str = Form(...),
    bed_days: float = Form(...),
    db: Session = Depends(get_db)
):
    patient = db.query(Staff).filter(Staff.id == patient_id).first()
    if not patient:
        return RedirectResponse(url="/staff", status_code=303)

    # Update the data
    patient.patient_name = patient_name
    patient.diagnosis = diagnosis
    patient.procedure = procedure
    patient.room_type = room_type
    patient.bed_days = bed_days
    
    patient.predicted_staff = get_staff_prediction(diagnosis, procedure, room_type, bed_days)

    db.commit()
    return RedirectResponse(url=f"/staff?org_code={patient.org_code}", status_code=303)

# DELETE RECORD
@app.post("/delete_patient/{patient_id}")
async def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Staff).filter(Staff.id == patient_id).first()
    if patient:
        org_code = patient.org_code
        db.delete(patient)
        db.commit()
        return RedirectResponse(url=f"/staff?org_code={org_code}", status_code=303)
    return RedirectResponse(url="/staff", status_code=303)

@app.post("/add_patient")
async def add_patient(org_code: str = Form(...), admission_date: str = Form(...), 
                    patient_name: str = Form(...), diagnosis: str = Form(...), 
                    procedure: str = Form(...), room_type: str = Form(...), 
                    bed_type: str = Form(...), bed_days: float = Form(...), 
                    db: Session = Depends(get_db)):
    pred = get_staff_prediction(diagnosis, procedure, room_type, bed_days)
    new_p = Staff(org_code=org_code, date=admission_date, patient_name=patient_name, diagnosis=diagnosis, 
                procedure=procedure, room_type=room_type, bed_type=bed_type, bed_days=bed_days, 
                predicted_staff=pred)
    db.add(new_p)
    db.commit()
    return RedirectResponse(url=f"/staff?org_code={org_code}", status_code=303)

@app.get("/medicine")
async def medicine_page(request: Request, org_code: str = "DEFAULT", db_session: Session = Depends(get_db)):
    meds = []
    if CURRENT_DF is not None and not CURRENT_DF.empty:
        meds = CURRENT_DF.replace({np.nan: None}).to_dict(orient='records')
    
    for med in meds:
        item_id = str(med.get('Item_ID', ''))
        med['predicted_min'] = 0
        med['predicted_max'] = 0
        med['forecast_next'] = 0
        med['status'] = 'Unknown'
        med['status_class'] = 'status-unknown'
        
        if item_id in STATS_CACHE:
            pred = STATS_CACHE[item_id]
            med['predicted_min'] = pred['min']
            med['predicted_max'] = pred['max']
            med['forecast_next'] = pred['pred']
            
            current = med.get('Current_Stock', 0)
            if current < pred['min']:
                med['status'] = 'Critical'
                med['status_class'] = 'status-critical'
            elif current < pred['pred']:
                med['status'] = 'Low'
                med['status_class'] = 'status-low'
            else:
                med['status'] = 'Good'
                med['status_class'] = 'status-good'

    return templates.TemplateResponse("medicine.html", {
        "request": request,
        "active_page": "medicine",
        "org_code": org_code,
        "medicines": meds,
        "source": SOURCE,
        "alerts_count": get_total_alerts_count(db_session, org_code)
    })

def get_medicine_forecast():
    files = sorted(glob.glob('inv_*.csv'))
    if not files:
        print("WARNING: No inv_*.csv files found.")
        return {}
        
    try:
        df_list = [pd.read_csv(f) for f in files]
        full_df = pd.concat(df_list)
        
        if 'Item_ID' not in full_df.columns: return {}
        full_df['Item_ID'] = full_df['Item_ID'].astype(str)
        
        item_ids = full_df['Item_ID'].unique()
        months_x = np.array(range(1, len(files) + 1)).reshape(-1, 1)
        
        forecast_results = {}
        for item in item_ids:
            y_usage = []
            for df in df_list:
                df['Item_ID'] = df['Item_ID'].astype(str)
                val = df[df['Item_ID'] == str(item)]['Used_This_Month'].values
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
    except Exception as e:
        print(f"Forecast Error: {e}")
        return {}

@app.get("/api/forecast/{item_id}")
async def get_forecast_api(item_id: str):
    global STATS_CACHE
    if not STATS_CACHE:
        STATS_CACHE = get_medicine_forecast()

    item_id_str = str(item_id).strip()
    forecast = STATS_CACHE.get(str(item_id_str))

    if forecast:
        return {
            "status": "success",
            "item_id": item_id_str,
            "min": forecast['min'],
            "max": forecast['max'],
            "prediction": forecast['pred']
        }
    return JSONResponse(status_code=404, content={"error": f"Item {item_id} not found in forecast."})

@app.post("/api/update_stock")
async def update_stock(item_id: str = Form(...), new_stock: int = Form(...)):
    global CURRENT_DF
    try:
        if not SOURCE or not os.path.exists(SOURCE): return JSONResponse(status_code=404, content={"message": "No CSV"})
        df = pd.read_csv(SOURCE)
        df.columns = df.columns.str.strip()
        df.loc[df['Item_ID'].astype(str) == str(item_id), 'Current_Stock'] = new_stock
        df.to_csv(SOURCE, index=False)
        CURRENT_DF = df
        return {"message": "Updated"}
    except Exception as e: return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/alerts")
async def alerts(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    combined_alerts = []
    subq = db.query(Inventory.item_name, func.max(Inventory.id).label('mid')).filter(
        Inventory.org_code == org_code).group_by(Inventory.item_name).subquery()
    sql_items = db.query(Inventory).join(subq, Inventory.id == subq.c.mid).all()
    
    for item in sql_items:
        if item.predicted_min and item.current_stock < item.predicted_min:
            combined_alerts.append({
                "item_name": item.item_name, "current_stock": item.current_stock,
                "predicted_min": item.predicted_min, 
                "deficit": round(item.predicted_min - item.current_stock, 1),
                "type": "inventory", "date": item.date
            })

    med_alerts = get_medicine_alerts()
    combined_alerts.extend(med_alerts)

    return templates.TemplateResponse("alert.html", {
        "request": request, "alerts": combined_alerts, "alerts_count": len(combined_alerts),
        "org_code": org_code, "active_page": "alerts"
    })

@app.post("/resolve_alert")
async def resolve_alert(data: AlertResolution, db: Session = Depends(get_db)):
    try:
        if data.item_type == "inventory":
            subq = db.query(Inventory.item_name, func.max(Inventory.id).label('mid')).filter(
                Inventory.org_code == data.org_code, Inventory.item_name == data.item_name).subquery()
            latest = db.query(Inventory).join(subq, Inventory.id == subq.c.mid).first()
            
            if latest:
                safe_stock = latest.predicted_max if latest.predicted_max > 0 else latest.predicted_min + 50
                new_entry = Inventory(org_code=data.org_code, date="Restocked", item_name=data.item_name,
                                    current_stock=safe_stock, predicted_min=latest.predicted_min,
                                    predicted_max=latest.predicted_max)
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
        return {"status": "error", "message": str(e)}

# --- AI PREDICTIONS PAGE ---
@app.get("/predictions")
async def predictions_page(request: Request, org_code: str = "DEFAULT"):
    """Render the AI Predictions page"""
    return templates.TemplateResponse("predictions.html", {
        "request": request,
        "org_code": org_code,
        "active_page": "predictions"
    })


@app.post("/api/chatbot")
async def chatbot_query(data: ChatbotRequest, db: Session = Depends(get_db)):
    """
    API endpoint for the Gemini AI chatbot
    Analyzes database and CSV data to answer questions
    """
    try:
        context = get_database_context(data.org_code, db)
        answer = query_gemini_ai(data.question, context)
        return {
            "status": "success",
            "answer": answer,
            "question": data.question
        }
    except Exception as e:
        print(f"Chatbot error: {e}")
        return {
            "status": "error",
            "answer": f"I apologize, but I encountered an error: {str(e)}",
            "question": data.question
        }
        
        
# --- COMMUNITY ROUTES ---

@app.get("/community")
async def community_page(request: Request, org_code: str = "DEFAULT", db: Session = Depends(get_db)):
    # Fetch all registered hospitals except perhaps the current one (optional)
    hospitals = db.query(User).all()
    return templates.TemplateResponse("community.html", {
        "request": request, 
        "hospitals": hospitals, 
        "org_code": org_code,
        "active_page": "community"
    })

@app.get("/api/community/inventory/{target_org_code}")
async def get_hospital_inventory(target_org_code: str, db: Session = Depends(get_db)):
    """Fetches the latest inventory items for a specific hospital"""
    # Using your existing logic to get only the latest entry for each item_name
    subq = db.query(
        Inventory.item_name, 
        func.max(Inventory.id).label('max_id')
    ).filter(Inventory.org_code == target_org_code).group_by(Inventory.item_name).subquery()
    
    items = db.query(Inventory).join(subq, Inventory.id == subq.c.max_id).all()
    
    return [
        {
            "item_name": item.item_name,
            "current_stock": item.current_stock,
            "predicted_min": item.predicted_min,
            "predicted_max": item.predicted_max,
            "status": "Low" if item.current_stock < item.predicted_min else "Stable",
            "last_updated": item.date
        } for item in items
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)