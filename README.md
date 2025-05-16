# Smart Bin Waste Prediction

This project uses time-series forecasting (Prophet) to predict daily waste accumulation based on historical data stored in Firebase Realtime Database. It runs daily using GitHub Actions and updates Firebase with the prediction.

### Setup:
1. Upload Firebase Admin SDK as GitHub secret: `FIREBASE_KEY`
2. Customize your Firebase database URL inside `predict.py`.
3. Push to GitHub and watch predictions auto-update daily.
# IOT
