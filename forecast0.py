from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import joblib


# Create output directory if not exists
os.makedirs('D:/data_engineer/SPG/models', exist_ok=True)
df = pd.read_csv('D:/data_engineer/SPG/prescriptions3.csv')
def forecast_feature(df, feature_name, periods=48, freq='H'):
    prophet_df = df[['DATE_TIME', feature_name]].rename(columns={'DATE_TIME': 'ds', feature_name: 'y'})
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=False,
        weekly_seasonality=False,
        changepoint_prior_scale=0.5
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast_result = forecast[['ds', 'yhat']].tail(periods).rename(columns={'ds': 'DATE_TIME', 'yhat': feature_name})
    joblib.dump(model, f'D:/data_engineer/SPG/models/prophet_{feature_name}.pkl')
    return forecast_result

# Forecast each feature
periods_to_forecast = 48  # Next 48 hours (or adjust as needed)
irradiation_forecast = forecast_feature(df, 'IRRADIATION', periods=periods_to_forecast)
module_temp_forecast = forecast_feature(df, 'MODULE_TEMPERATURE', periods=periods_to_forecast)
ambient_temp_forecast = forecast_feature(df, 'AMBIENT_TEMPERATURE', periods=periods_to_forecast)
dc_power_forecast = forecast_feature(df, 'DC_POWER', periods=periods_to_forecast)  # Added DC power forecast

# Merge all feature forecasts
future_features = irradiation_forecast.merge(module_temp_forecast, on='DATE_TIME').merge(ambient_temp_forecast, on='DATE_TIME').merge(dc_power_forecast, on='DATE_TIME')

# Predict future DC_POWER using saved model
rf_model = joblib.load('solar_model_DC_POWER1.pkl')
future_features['PREDICTED_DC_POWER'] = rf_model.predict(future_features[['IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']])

# Anomaly detection on forecasted data
def detect_dc_anomaly_forecast(row):
    # Compare Prophet's DC power forecast with RF model prediction
    diff = abs(row['PREDICTED_DC_POWER'] - row['DC_POWER'])
    if diff <= 0.1 * row['PREDICTED_DC_POWER']:  # 10% threshold
        return -1
    if row['MODULE_TEMPERATURE'] > 50:
        return 1
    elif row['AMBIENT_TEMPERATURE'] > 40:
        return 0
    elif row['IRRADIATION'] > 0.4:
        return 2
    return -1

def reason_text(anomaly_code):
    return {
        1: "High Module Temperature",
        0: "High Ambient Temperature",
        2: "High Irradiation",
        -1: "No anomaly"
    }.get(anomaly_code, "Unknown")

def prescription_text(anomaly_code):
    return {
        1: "Check cooling system or module exposure.",
        0: "Inspect ventilation and surroundings.",
        2: "Verify panel performance under high irradiance.",
        -1: "No action needed"
    }.get(anomaly_code, "Unknown")

# Anomaly detection using both Prophet and RF model predictions
future_features['EFFICIENCY_ANOMALY_DC'] = future_features.apply(detect_dc_anomaly_forecast, axis=1)
future_features['REASON_DC_POWER'] = future_features['EFFICIENCY_ANOMALY_DC'].apply(reason_text)
future_features['PRESCRIPTION_DC_POWER'] = future_features['EFFICIENCY_ANOMALY_DC'].apply(prescription_text)

# Save forecasted output
future_features.to_csv('D:/data_engineer/SPG/dc_power_forecast_with_anomalies3.csv', index=False)
print("📈 DC_POWER forecast and anomaly detection completed and saved.")

