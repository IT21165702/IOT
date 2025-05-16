# predict.py
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import json
import os

# Load Firebase credentials from file
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ecosmarttrashbin-bcee7-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Fetch historical data from /waste_logs
ref = db.reference('waste_logs')
raw_data = ref.get()

# Extract and clean
data = []
for record in raw_data.values():
    ts = record.get('timestamp')
    wt = record.get('weight')
    if ts and wt:
        date = ts.split("T")[0]
        data.append((date, wt))

df = pd.DataFrame(data, columns=["ds", "y"])
df["ds"] = pd.to_datetime(df["ds"])
df = df.groupby("ds").sum().reset_index()

# Train Prophet
model = Prophet(daily_seasonality=True)
model.fit(df)

# Predict tomorrow
future = model.make_future_dataframe(periods=1)
forecast = model.predict(future)
tomorrow = forecast.iloc[-1]
pred_date = tomorrow['ds'].date().isoformat()
predicted = round(tomorrow['yhat'], 2)

# Upload to Firebase under /predictions
pred_ref = db.reference('predictions')
pred_ref.child(pred_date).set({'predicted_weight': predicted})

print(f"✅ Uploaded prediction for {pred_date}: {predicted}g")
