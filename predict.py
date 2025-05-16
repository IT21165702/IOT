import pandas as pd
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os
import json
import sys
# Load credentials from firebase_key.json with error handling
try:
   with open("firebase_key.json", "r") as f:
       cred_json = json.load(f)
except json.JSONDecodeError as e:
   print(f"Error: Invalid JSON in firebase_key.json - {e}")
   sys.exit(1)
except FileNotFoundError:
   print("Error: firebase_key.json not found.")
   sys.exit(1)
# Initialize Firebase app if not already initialized
cred = credentials.Certificate(cred_json)
if not firebase_admin._apps:
   firebase_admin.initialize_app(cred, {
       'databaseURL': 'https://ecosmarttrashbin-bcee7-default-rtdb.asia-southeast1.firebasedatabase.app'
   })
# Fetch past actual weights from Firebase
ref = db.reference('waste_logs')
data = ref.get()
# Format to Prophet-ready DataFrame
df = []
for entry in data.values():
   try:
       ts = entry['timestamp']
       weight = float(entry['weight'])
       date = pd.to_datetime(ts).date()
       df.append({'ds': str(date), 'y': weight})
   except (KeyError, ValueError, TypeError):
       continue  # skip malformed entries
df = pd.DataFrame(df)
if df.empty:
   print("Error: No valid data to train the model.")
   sys.exit(1)
# Aggregate duplicates
df = df.groupby("ds").sum().reset_index()
# Train Prophet model
model = Prophet()
model.fit(df)
# Forecast next day
future = model.make_future_dataframe(periods=1)
forecast = model.predict(future)
tomorrow = forecast.iloc[-1]
pred_date = str(tomorrow['ds'].date())
predicted = round(tomorrow['yhat'], 2)
print(f"📦 Predicted weight for {pred_date}: {predicted}g")
# Push prediction to Firebase
pred_ref = db.reference('predictions')
pred_ref.child(pred_date).set({
   'predicted_weight': predicted
})