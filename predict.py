import pandas as pd
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os
import json

# Load credentials from the environment (GitHub secret)
with open("firebase_key.json") as f:
    cred_json = json.load(f)

cred = credentials.Certificate(cred_json)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://ecosmarttrashbin-bcee7-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

# Get past actual weights from Firebase
ref = db.reference('waste_logs')
data = ref.get()

# Format data to Prophet format
df = []
for entry in data.values():
    try:
        ts = entry['timestamp']
        weight = float(entry['weight'])
        date = pd.to_datetime(ts).date()
        df.append({'ds': str(date), 'y': weight})
    except:
        continue

df = pd.DataFrame(df)
df = df.groupby("ds").sum().reset_index()

# Train and forecast
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=1)
forecast = model.predict(future)

tomorrow = forecast.iloc[-1]
pred_date = str(tomorrow['ds'].date())
predicted = round(tomorrow['yhat'], 2)

print(f"📦 Predicted weight for {pred_date}: {predicted}g")

# Save back to Firebase
pred_ref = db.reference('predictions')
pred_ref.child(pred_date).set({
    'predicted_weight': predicted
})
