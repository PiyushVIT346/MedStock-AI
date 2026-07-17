# 🏥 MedStock-AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow)
![Gemini AI](https://img.shields.io/badge/Gemini-AI-4285F4?style=for-the-badge&logo=google)
![SQLite](https://img.shields.io/badge/Database-SQLAlchemy-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### AI-Powered Smart Hospital Inventory & Resource Management System

*Predict • Monitor • Optimize • Collaborate*

</div>

---

## 📌 Overview

**MedStock-AI** is an AI-powered healthcare inventory management platform designed to help hospitals, clinics, and medical organizations efficiently manage medical supplies, predict future inventory requirements, optimize hospital staffing, and provide intelligent insights through an AI chatbot.

The system combines **Machine Learning**, **Deep Learning**, **Generative AI (Gemini)**, **FastAPI**, and **SQL databases** to create a smart healthcare management ecosystem.

---

# ✨ Features

## 📦 Smart Inventory Management

- Add and manage hospital inventory
- Track stock availability
- Predict minimum and maximum stock levels
- Automatic low-stock alerts
- Inventory visualization dashboard

---

## 💊 Medicine Forecasting

- Medicine stock prediction
- Monthly demand forecasting
- CSV-based medicine database
- Linear Regression forecasting
- Smart medicine alerts

---

## 👨‍⚕️ AI Staff Prediction

Predicts required medical staff based on:

- Patient diagnosis
- Procedure performed
- Room type
- Bed occupancy
- Length of stay

Uses a trained TensorFlow deep learning model.

---

## 🤖 Gemini AI Medical Assistant

Integrated Google Gemini AI chatbot capable of:

- Answering inventory questions
- Medicine analysis
- Patient-based insights
- Inventory recommendations
- Data-driven decision support

---

## 🌍 WHO Disease Outbreak Monitor

Fetches latest disease outbreak reports from WHO API.

Displays:

- Disease
- Country
- Date
- Suggested treatment
- Official WHO report

---

## 🚨 Smart Alert System

Automatically detects

- Low inventory
- Medicine shortage
- Critical stock
- Restocking suggestions

---

## 🏥 Community Hospital Network

Hospitals can

- View inventory of connected hospitals
- Compare stock availability
- Improve resource sharing
- Collaborate during emergencies

---

## 📊 Dashboard Analytics

Interactive dashboard displaying

- Total inventory
- Medicine stock
- Alerts
- Patient count
- Charts
- Statistics

---

# 🧠 AI Models Used

| Model | Purpose |
|---------|----------|
| TensorFlow LSTM | Inventory Stock Prediction |
| Deep Neural Network | Staff Requirement Prediction |
| Linear Regression | Medicine Demand Forecast |
| Google Gemini 2.5 Flash | AI Medical Assistant |

---

# 🛠 Tech Stack

### Backend

- FastAPI
- Python
- SQLAlchemy
- Uvicorn

### Machine Learning

- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Joblib

### AI

- Google Gemini API

### Frontend

- HTML
- CSS
- JavaScript
- Jinja2 Templates

### Database

- SQLAlchemy
- SQLite / PostgreSQL (Environment Configurable)

---

# 📂 Project Structure

```
MedStock-AI/
│
├── templates/
│   ├── landing.html
│   ├── login.html
│   ├── signin.html
│   ├── home.html
│   ├── inventory.html
│   ├── medicine.html
│   ├── staff.html
│   ├── predictions.html
│   ├── alert.html
│   └── community.html
│
├── static/
│
├── models/
│   ├── hospital_stock_model.h5
│   ├── augmented_hospital_model.h5
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   ├── scaler_staff.joblib
│   └── classes_staff.joblib
│
├── dataset/
│   └── inv_*.csv
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/MedStock-AI.git

cd MedStock-AI
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file

```env
DATABASE_URL=sqlite:///hospital_data.db

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

## Run Server

```bash
uvicorn app:app --reload
```

Application will be available at

```
http://127.0.0.1:8000
```

---

# 📈 Machine Learning Workflow

```
Hospital Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Deep Learning Models
      │
      ▼
Stock Prediction
Staff Prediction
Medicine Forecast
      │
      ▼
Dashboard Visualization
```

---

# 🔥 API Endpoints

| Method | Endpoint | Description |
|----------|------------|----------------|
| GET | / | Landing Page |
| GET | /login | Login Page |
| POST | /login | Authenticate User |
| GET | /home | Dashboard |
| GET | /inventory | Inventory Management |
| POST | /predict | Predict Inventory Stock |
| POST | /add_inventory_item | Add Inventory |
| GET | /medicine | Medicine Dashboard |
| GET | /staff | Staff Dashboard |
| POST | /add_patient | Add Patient |
| GET | /alerts | Alerts Page |
| GET | /predictions | AI Predictions |
| POST | /api/chatbot | Gemini AI Chatbot |
| GET | /community | Hospital Community |
| GET | /api/who_outbreaks | WHO Disease Outbreaks |

---

# 📸 Screenshots

```
screenshots/
│
├── dashboard.png
├── inventory.png
├── medicine.png
├── chatbot.png
├── alerts.png
├── predictions.png
└── community.png
```

Add your screenshots here:

```markdown
## Dashboard

![Dashboard](screenshots/dashboard.png)

---

## Medicine Forecast

![Medicine](screenshots/medicine.png)

---

## AI Chatbot

![Chatbot](screenshots/chatbot.png)
```

---

# 🚀 Future Improvements

- Barcode Scanner Integration
- QR Code Inventory
- Multi-Hospital Cloud Deployment
- IoT Device Integration
- Automated Purchase Orders
- AI Demand Forecasting with LSTM
- Mobile Application
- Real-time Notifications
- Voice Assistant
- Role-based Access Control

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Piyush Singh**

B.Tech CSE (Applied Machine Learning)

VIT Bhopal University

GitHub: https://github.com/PiyushVIT346

LinkedIn: https://linkedin.com/in/piyushsingh346

---

<div align="center">

### ⭐ If you found this project helpful, don't forget to star the repository!

**Made with ❤️ using FastAPI, TensorFlow, and Gemini AI**

</div>
