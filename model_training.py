import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Data
gen = pd.read_csv('D:/data_engineer/SPG/Plant1_Generation_Data_Cleaned.csv')
weather = pd.read_csv('D:/data_engineer/SPG/Plant1_Weather_Sensor_Data_Cleaned.csv')

gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'])
weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'])

df = pd.merge(gen, weather, on='DATE_TIME', how='inner')
df.fillna(method='ffill', inplace=True)

# Features & Target
features = ['IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']
target = 'DC_POWER'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
df['PREDICTED_DC_POWER'] = model.predict(df[features])

# Predict on test and evaluate
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)
df.loc[X_test.index, 'PREDICTED_DC_POWER'] = y_pred

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred) * 100
print(f'DC_POWER: RMSE = {rmse:.2f}, R2 = {r2:.2f}%')

# Save model and accuracy
joblib.dump(model, 'solar_model_DC_POWER1.pkl')
joblib.dump(r2, 'model_accuracy_DC_POWER.pkl')

# Anomaly detection
threshold = rmse * 1.5

def detect_dc_anomaly(row):
    if pd.isna(row['PREDICTED_DC_POWER']) or pd.isna(row['DC_POWER']):
        return np.nan

    diff = abs(row['DC_POWER'] - row['PREDICTED_DC_POWER'])
    if diff <= threshold:
        return -1  # No anomaly

    # Check each anomaly type in order of importance
    if row['MODULE_TEMPERATURE'] > 50:
        return 1  # Red - Module Temp Anomaly
    if row['AMBIENT_TEMPERATURE'] > 30:
        return 0  # Green - Ambient Temp Anomaly
    if row['IRRADIATION'] > 0.4:
        return 2  # Yellow - Irradiation Anomaly

    return -1  # Default case if none matched


def reason_text(code):
    return {
        -1: 'Normal Operation',
        0: 'High Ambient Temperature (>40°C)',
        1: 'High Module Temperature (>50°C)',
        2: 'High Irradiation with Low Output → Possible Panel Issue'
    }.get(code, 'No data')

def prescription_text(code):
    return {
        -1: 'System operating normally',
        0: 'Check cooling system and ventilation',
        1: 'Inspect inverter for overheating and reduce load',
        2: 'Check panel efficiency and clean solar panels'
    }.get(code, 'No data')

df['EFFICIENCY_ANOMALY_DC'] = df.apply(detect_dc_anomaly, axis=1)
df['REASON_DC_POWER'] = df['EFFICIENCY_ANOMALY_DC'].apply(reason_text)
df['PRESCRIPTION_DC_POWER'] = df['EFFICIENCY_ANOMALY_DC'].apply(prescription_text)

# Visualization: Last 20% actual vs predicted
def plot_actual_vs_predicted():
    valid_data = df.dropna(subset=['PREDICTED_DC_POWER', 'DC_POWER']).sort_index()
    last_20_pct = int(0.2 * len(valid_data))
    plot_data = valid_data.tail(last_20_pct)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_data['DATE_TIME'], plot_data['DC_POWER'], label='Actual DC_POWER', alpha=0.6)
    plt.plot(plot_data['DATE_TIME'], plot_data['PREDICTED_DC_POWER'], label='Predicted DC_POWER', color='red', alpha=0.6)
    plt.title('Actual vs Predicted DC_POWER (Last 20% of data)')
    plt.xlabel('Timestamp')
    plt.ylabel('DC_POWER')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted()
# Save output
df.to_csv('D:/data_engineer/SPG/prescriptions4.csv', index=False)
print("✅ DC_POWER model, anomaly detection, and outputs saved.")
