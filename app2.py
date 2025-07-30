from flask import Flask, render_template, request, send_file, Response, jsonify, flash
import pandas as pd
import plotly.express as px
import plotly.io as pio
import joblib
import os
from prophet import Prophet
from datetime import timedelta
import json
import time
import plotly.graph_objects as go
import plotly
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
app.secret_key = 'secret'

# Update Plotly theme to dark
pio.templates.default = "plotly_dark"

# Load initial data
csv_path = 'prescriptions4.csv'
model = joblib.load('solar_model_DC_POWER1.pkl')
accuracy = joblib.load('model_accuracy_DC_POWER.pkl')
data = pd.read_csv(csv_path, parse_dates=['DATE_TIME'])

# Load DC Power Prophet model
dc_power_prophet_model = joblib.load('models/prophet_DC_POWER.pkl')

# Load Prophet models
ambient_prophet_model = joblib.load('models/prophet_AMBIENT_TEMPERATURE.pkl')
irradiation_prophet_model = joblib.load('models/prophet_IRRADIATION.pkl')
module_prophet_model = joblib.load('models/prophet_MODULE_TEMPERATURE.pkl')

# Ensure filtered_data is defined
filtered_data = data.copy()

# Gemini API setup
genai.configure(api_key="AIzaSyB9tfTYe7snlqkR7z31tdOETPmXINc6IAw")
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# Email map
ISSUE_EMAIL_MAP = {
    'High Module Temperature': 'amruta.suryawanshi@c4i4.org',
    'High Ambient Temperature': 'anuj.gaikwad@c4i4.org',
    'Low Irradiation': 'amruta.suryawanshi@c4i4',
    'Sudden Module Temp Spike': 'anuj.gaikwad@c4i4.org'
}

def get_reason_prescription(amb_temp, mod_temp, irradiation):
    prompt = f"""
    You are a solar monitoring expert. Analyze:

    - Ambient Temp: {amb_temp} °C
    - Module Temp: {mod_temp} °C
    - Irradiation: {irradiation} kW/m²

    Choose from:
    1. High Module Temperature (>50°C)
    2. High Ambient Temperature (>40°C)
    3. Low Irradiation (<0.2 kW/m²)
    4. Sudden Module Temp Spike (Module - Ambient > 20°C)

    For each triggered anomaly:
    Format: Reason: <reason> | Prescription: <prescription>

    If no anomaly: "No anomaly detected. System running normally."
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def send_email(reason_text):
    recipients = []
    details = []

    if "No anomaly detected" in reason_text:
        return recipients, details

    # For each anomaly line, route mail to specific supervisor
    for line in reason_text.split('\n'):
        if "Reason:" in line and "|" in line:
            reason_text_raw = line.split("Reason:")[1].split("|")[0].strip()

            # Match to specific issue
            matched_email = None
            for issue, email in ISSUE_EMAIL_MAP.items():
                if issue.lower() in reason_text_raw.lower():
                    matched_email = email
                    break

            if matched_email:
                msg = MIMEText(f"⚠️ Solar Plant Alert\n\n{line}")
                msg['Subject'] = '⚠️ Solar Efficiency Anomaly Alert'
                msg['From'] = 'amruta.suryawanshi@c4i4.org'
                msg['To'] = matched_email

                try:
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login('amruta.suryawanshi@c4i4.org', 'ueli pbow kvve nuee')
                        smtp.send_message(msg)
                        recipients.append(matched_email)
                        details.append({
                            "failure": reason_text_raw,
                            "reason": line.split("Reason:")[1].split("|")[0].strip(),
                            "prescription": line.split("|")[1].strip(),
                            "supervisor_email": matched_email
                        })
                except Exception as e:
                    print(f"❌ Email error for {reason_text_raw}: {e}")

    return recipients, details

# Prepare real-time data
start_date = pd.to_datetime('2020-06-11 14:00:00')  # June 11th, 2 PM
realtime_data = data[data['DATE_TIME'] >= start_date].tail(int(0.2 * len(data)))
current_index = 0

def generate_realtime_data():
    global current_index
    while True:
        if current_index >= len(realtime_data):
            current_index = 0
        
        # Get last 100 points
        end_idx = current_index + 1
        start_idx = max(0, end_idx - 100)
        window_data = realtime_data.iloc[start_idx:end_idx]
        
        # Get current anomalies with their reasons and prescriptions
        anomalies = window_data[window_data['EFFICIENCY_ANOMALY_DC'] != -1].copy()
        anomaly_count = len(anomalies)
        
        # Get only the latest anomaly
        current_anomalies = []
        if not anomalies.empty:
            latest_anomaly = anomalies.iloc[-1]  # Get the most recent anomaly
            if pd.notna(latest_anomaly['REASON_DC_POWER']) and pd.notna(latest_anomaly['PRESCRIPTION_DC_POWER']):
                # Send email for the latest anomaly
                amb_temp = latest_anomaly['AMBIENT_TEMPERATURE']
                mod_temp = latest_anomaly['MODULE_TEMPERATURE']
                irradiation = latest_anomaly['IRRADIATION']
                
                result = get_reason_prescription(amb_temp, mod_temp, irradiation)
                email_recipients, email_details = send_email(result)
                
                current_anomalies.append({
                    'DATE_TIME': latest_anomaly['DATE_TIME'].strftime('%Y-%m-%d %H:%M:%S'),
                    'REASON_DC_POWER': str(latest_anomaly['REASON_DC_POWER']),
                    'PRESCRIPTION_DC_POWER': str(latest_anomaly['PRESCRIPTION_DC_POWER']),
                    'email_details': email_details
                })
        
        # Create separate charts for each parameter
        charts = {}
        
        # Ambient Temperature Chart
        ambient_fig = px.line(
            window_data,
            x='DATE_TIME',
            y='AMBIENT_TEMPERATURE',
            title='Real-time Ambient Temperature',
            labels={'AMBIENT_TEMPERATURE': 'Temperature (°C)', 'DATE_TIME': 'Time'},
            color_discrete_sequence=['#03dac6']
        )
        ambient_fig.update_layout(
            plot_bgcolor='#1f1f1f',
            paper_bgcolor='#1f1f1f',
            font=dict(color='#e0e0e0'),
            title_font=dict(color='#bb86fc'),
            showlegend=True
        )
        charts['ambient'] = pio.to_json(ambient_fig)

        # Module Temperature Chart
        module_fig = px.line(
            window_data,
            x='DATE_TIME',
            y='MODULE_TEMPERATURE',
            title='Real-time Module Temperature',
            labels={'MODULE_TEMPERATURE': 'Temperature (°C)', 'DATE_TIME': 'Time'},
            color_discrete_sequence=['#bb86fc']
        )
        module_fig.update_layout(
            plot_bgcolor='#1f1f1f',
            paper_bgcolor='#1f1f1f',
            font=dict(color='#e0e0e0'),
            title_font=dict(color='#bb86fc'),
            showlegend=True
        )
        charts['module'] = pio.to_json(module_fig)

        # Irradiation Chart
        irrad_fig = px.line(
            window_data,
            x='DATE_TIME',
            y='IRRADIATION',
            title='Real-time Irradiation',
            labels={'IRRADIATION': 'Irradiation (W/m²)', 'DATE_TIME': 'Time'},
            color_discrete_sequence=['#ff9800']
        )
        irrad_fig.update_layout(
            plot_bgcolor='#1f1f1f',
            paper_bgcolor='#1f1f1f',
            font=dict(color='#e0e0e0'),
            title_font=dict(color='#bb86fc'),
            showlegend=True
        )
        charts['irradiation'] = pio.to_json(irrad_fig)

        # DC Power Chart
        power_fig = go.Figure()
        
        # Add main DC Power line
        power_fig.add_scatter(
            x=window_data['DATE_TIME'],
            y=window_data['DC_POWER'],
            mode='lines',
            line=dict(color='#03dac6', width=2),
            name='DC Power'
        )
        
        # Add different colored markers for each anomaly type
        anomaly_types = {
            0: {'color': '#ff9800', 'name': 'Ambient Temperature Anomaly'},  # Orange
            1: {'color': '#f44336', 'name': 'Module Temperature Anomaly'},   # Red
            2: {'color': '#ffeb3b', 'name': 'Irradiation Anomaly'},          # Yellow
        }
        
        for anomaly_code, style in anomaly_types.items():
            anomaly_points = window_data[window_data['EFFICIENCY_ANOMALY_DC'] == anomaly_code]
            if not anomaly_points.empty:
                power_fig.add_scatter(
                    x=anomaly_points['DATE_TIME'],
                    y=anomaly_points['DC_POWER'],
                    mode='markers',
                    marker=dict(
                        color=style['color'],
                        size=8,
                        symbol='circle'
                    ),
                    name=style['name']
                )
        
        power_fig.update_layout(
            title='Real-time DC Power',
            plot_bgcolor='#1f1f1f',
            paper_bgcolor='#1f1f1f',
            font=dict(color='#e0e0e0'),
            title_font=dict(color='#bb86fc'),
            legend=dict(
                orientation='h',
                y=-0.3,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                title='Time',
                gridcolor='#2d2d2d',
                showgrid=True
            ),
            yaxis=dict(
                title='DC Power',
                gridcolor='#2d2d2d',
                showgrid=True
            )
        )
        charts['power'] = pio.to_json(power_fig)
        
        # Add anomaly data to the response
        charts['anomaly_data'] = {
            'count': anomaly_count,
            'anomalies': current_anomalies
        }
        
        current_index += 1
        time.sleep(2)  # Update every 2 seconds
        
        yield f"data: {json.dumps(charts)}\n\n"

def create_forecast_chart(model, days, title, color):
    # Create future dataframe for the specified number of days
    periods = days * 24 * 4  # 4 intervals per hour
    future = model.make_future_dataframe(periods=periods, freq='15T')
    forecast = model.predict(future)
    
    # Get only the future forecast
    forecast_future = forecast[forecast['ds'] > filtered_data['DATE_TIME'].max()]
    
    # Create the plot
    fig = go.Figure()
    
    # Add forecast line
    fig.add_scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines',
        line=dict(color=color, width=2),
        name='Forecast'
    )
    
    # Add confidence intervals
    fig.add_scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    )
    
    fig.add_scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
        name='Confidence Interval'
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        width=1400,
        height=600,
        margin=dict(l=50, r=50, t=50, b=100),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        yaxis=dict(
            title='Value',
            gridcolor='#2d2d2d',
            showgrid=True
        )
    )
    
    return fig

def calculate_daily_energy(data):
    # Convert DATE_TIME to datetime if it's not already
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
    
    # Extract date from datetime
    data['date'] = data['DATE_TIME'].dt.date
    
    # Calculate daily energy production (sum of DC_POWER for each day)
    daily_energy = data.groupby('date')['DC_POWER'].sum().reset_index()
    daily_energy['date'] = pd.to_datetime(daily_energy['date'])
    
    return daily_energy

def calculate_hourly_avg_temp(data):
    # Convert DATE_TIME to datetime if it's not already
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
    
    # Extract hour and day of week
    data['hour'] = data['DATE_TIME'].dt.hour
    data['day_of_week'] = data['DATE_TIME'].dt.day_name()
    
    # Calculate average temperature by hour and day of week
    hourly_avg = data.groupby(['day_of_week', 'hour'])['AMBIENT_TEMPERATURE'].mean().reset_index()
    
    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hourly_avg['day_of_week'] = pd.Categorical(hourly_avg['day_of_week'], categories=day_order, ordered=True)
    hourly_avg = hourly_avg.sort_values(['day_of_week', 'hour'])
    
    return hourly_avg

def calculate_daily_min_max_temp(data):
    # Convert DATE_TIME to datetime if it's not already
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
    
    # Extract date
    data['date'] = data['DATE_TIME'].dt.date
    
    # Calculate min and max temperature for each day
    daily_stats = data.groupby('date').agg({
        'AMBIENT_TEMPERATURE': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['date', 'min_temp', 'max_temp']
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    
    return daily_stats

def calculate_dc_ac_efficiency(data):
    # Convert DATE_TIME to datetime if it's not already
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
    
    # Extract date
    data['date'] = data['DATE_TIME'].dt.date
    
    # Calculate daily DC and AC power
    daily_power = data.groupby('date').agg({
        'DC_POWER': 'sum',
        'AC_POWER': 'sum'
    }).reset_index()
    
    # Calculate efficiency percentage
    daily_power['efficiency'] = (daily_power['AC_POWER'] / daily_power['DC_POWER'] * 100)
    daily_power['date'] = pd.to_datetime(daily_power['date'])
    
    return daily_power

def calculate_irradiation_heatmap(data):
    # Convert DATE_TIME to datetime if it's not already
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
    
    # Extract hour and day of week
    data['hour'] = data['DATE_TIME'].dt.hour
    data['day_of_week'] = data['DATE_TIME'].dt.day_name()
    
    # Calculate average irradiation by hour and day of week
    heatmap_data = data.pivot_table(
        values='IRRADIATION',
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )
    
    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    return heatmap_data

def calculate_temperature_heatmap(data):
    # Convert DATE_TIME to datetime if it's not already
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
    
    # Extract hour and day of week
    data['hour'] = data['DATE_TIME'].dt.hour
    data['day_of_week'] = data['DATE_TIME'].dt.day_name()
    
    # Calculate average temperature by hour and day of week
    heatmap_data = data.pivot_table(
        values='AMBIENT_TEMPERATURE',
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )
    
    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    return heatmap_data

@app.route('/')
def home():
    anomaly_points = filtered_data[filtered_data['EFFICIENCY_ANOMALY_DC'] == 1]

    # Calculate daily energy production
    daily_energy = calculate_daily_energy(filtered_data)
    
    # Calculate hourly average temperature
    hourly_avg_temp = calculate_hourly_avg_temp(filtered_data)
    
    # Calculate daily min/max temperatures
    daily_min_max = calculate_daily_min_max_temp(filtered_data)
    
    # Calculate DC to AC efficiency
    dc_ac_efficiency = calculate_dc_ac_efficiency(filtered_data)
    
    # Calculate irradiation heatmap data
    irradiation_heatmap = calculate_irradiation_heatmap(filtered_data)
    
    # Calculate temperature heatmap data
    temperature_heatmap = calculate_temperature_heatmap(filtered_data)
    
    # Daily Energy Production Chart
    fig_daily_energy = go.Figure()
    fig_daily_energy.add_scatter(
        x=daily_energy['date'],
        y=daily_energy['DC_POWER'],
        mode='lines+markers',
        line=dict(color='#03dac6', width=2),
        marker=dict(size=8),
        name='Daily Energy Production'
    )
    
    fig_daily_energy.update_layout(
        title='Daily Energy Production',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        xaxis=dict(
            title='Date',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        yaxis=dict(
            title='Energy Production (DC Power)',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_daily_energy = pio.to_html(fig_daily_energy, full_html=False)

    # DC to AC Efficiency Chart
    fig_efficiency = go.Figure()
    fig_efficiency.add_scatter(
        x=dc_ac_efficiency['date'],
        y=dc_ac_efficiency['efficiency'],
        mode='lines+markers',
        name='Conversion Efficiency',
        line=dict(color='#bb86fc', width=2),
        marker=dict(size=8)
    )
    
    fig_efficiency.update_layout(
        title='Daily DC to AC Power Conversion Efficiency',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        xaxis=dict(
            title='Date',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        yaxis=dict(
            title='Efficiency (%)',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_efficiency = pio.to_html(fig_efficiency, full_html=False)

    # Irradiation Heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=irradiation_heatmap.values,
        x=irradiation_heatmap.columns,
        y=irradiation_heatmap.index,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='Irradiation (W/m²)',
                font=dict(color='#e0e0e0')
            ),
            tickfont=dict(color='#e0e0e0')
        )
    ))
    
    fig_heatmap.update_layout(
        title='☀️ Hourly Average Irradiation by Day of Week ☀️',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        xaxis=dict(
            title='Hour of Day',
            gridcolor='#2d2d2d',
            showgrid=True,
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title='Day of Week',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_heatmap = pio.to_html(fig_heatmap, full_html=False)

    # Temperature Heatmap
    fig_temp_heatmap = go.Figure(data=go.Heatmap(
        z=temperature_heatmap.values,
        x=temperature_heatmap.columns,
        y=temperature_heatmap.index,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='Temperature (°C)',
                font=dict(color='#e0e0e0')
            ),
            tickfont=dict(color='#e0e0e0')
        )
    ))
    
    fig_temp_heatmap.update_layout(
        title='🌡️ Hourly Average Ambient Temperature by Day of Week 🌡️',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        xaxis=dict(
            title='Hour of Day',
            gridcolor='#2d2d2d',
            showgrid=True,
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title='Day of Week',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_temp_heatmap = pio.to_html(fig_temp_heatmap, full_html=False)

    # Hourly Average Temperature Chart
    fig_hourly_temp = go.Figure()
    
    # Create a color palette for days of week
    colors = ['#03dac6', '#bb86fc', '#ff9800', '#4caf50', '#2196f3', '#e91e63', '#9c27b0']
    
    for i, day in enumerate(hourly_avg_temp['day_of_week'].unique()):
        day_data = hourly_avg_temp[hourly_avg_temp['day_of_week'] == day]
        fig_hourly_temp.add_scatter(
            x=day_data['hour'],
            y=day_data['AMBIENT_TEMPERATURE'],
            mode='lines+markers',
            name=day,
            line=dict(color=colors[i], width=2),
            marker=dict(size=8)
        )
    
    fig_hourly_temp.update_layout(
        title='🌡️ Hourly Average Ambient Temperature by Day of Week 🌡️',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        xaxis=dict(
            title='Hour of Day',
            gridcolor='#2d2d2d',
            showgrid=True,
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title='Temperature (°C)',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_hourly_temp = pio.to_html(fig_hourly_temp, full_html=False)

    # Daily Min/Max Temperature Chart
    fig_min_max_temp = go.Figure()
    
    # Add max temperature line
    fig_min_max_temp.add_scatter(
        x=daily_min_max['date'],
        y=daily_min_max['max_temp'],
        mode='lines+markers',
        name='Maximum Temperature',
        line=dict(color='#ff9800', width=2),
        marker=dict(size=8)
    )
    
    # Add min temperature line
    fig_min_max_temp.add_scatter(
        x=daily_min_max['date'],
        y=daily_min_max['min_temp'],
        mode='lines+markers',
        name='Minimum Temperature',
        line=dict(color='#03dac6', width=2),
        marker=dict(size=8)
    )
    
    # Add area between min and max
    fig_min_max_temp.add_scatter(
        x=daily_min_max['date'],
        y=daily_min_max['max_temp'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    )
    
    fig_min_max_temp.add_scatter(
        x=daily_min_max['date'],
        y=daily_min_max['min_temp'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 152, 0, 0.1)',
        name='Temperature Range',
        showlegend=True
    )
    
    fig_min_max_temp.update_layout(
        title='Daily Minimum and Maximum Ambient Temperature',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        xaxis=dict(
            title='Date',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        yaxis=dict(
            title='Temperature (°C)',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_min_max_temp = pio.to_html(fig_min_max_temp, full_html=False)

    # Power Chart
    fig_power = px.line(
        filtered_data,
        x='DATE_TIME',
        y='DC_POWER',
        title='DC Power with Anomalies',
        color_discrete_sequence=['#03dac6']
    )
    
    # Add different colored markers for each anomaly type
    anomaly_types = {
        0: {'color': '#ff9800', 'name': 'Ambient Temperature Anomaly'},  # Orange
        1: {'color': '#f44336', 'name': 'Module Temperature Anomaly'},   # Red
        2: {'color': '#ffeb3b', 'name': 'Irradiation Anomaly'},          # Yellow
    }
    
    for anomaly_code, style in anomaly_types.items():
        anomaly_points = filtered_data[filtered_data['EFFICIENCY_ANOMALY_DC'] == anomaly_code]
        if not anomaly_points.empty:
            fig_power.add_scatter(
                x=anomaly_points['DATE_TIME'],
                y=anomaly_points['DC_POWER'],
                mode='markers',
                marker=dict(
                    color=style['color'],
                    size=8,
                    symbol='diamond'
                ),
                name=style['name']
            )
    
    fig_power.update_layout(
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc')
    )
    graph_power = pio.to_html(fig_power, full_html=False)

    # Environmental Parameter Charts
    # Module Temperature Chart
    fig_module = go.Figure()
    fig_module.add_scatter(
        x=filtered_data['DATE_TIME'],
        y=filtered_data['MODULE_TEMPERATURE'],
        mode='lines',
        line=dict(color='#bb86fc', width=2),
        name='Module Temperature'
    )
    # Add threshold line
    fig_module.add_hline(
        y=50,
        line=dict(color='red', width=1, dash='dash'),
        name='Threshold (50°C)'
    )
    fig_module.update_layout(
        title='Module Temperature',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        xaxis=dict(
            title='Date',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        yaxis=dict(
            title='Module Temperature',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_module = pio.to_html(fig_module, full_html=False)

    # Irradiation Chart
    fig_irradiation = go.Figure()
    fig_irradiation.add_scatter(
        x=filtered_data['DATE_TIME'],
        y=filtered_data['IRRADIATION'],
        mode='lines',
        line=dict(color='#ff9800', width=2),
        name='Irradiation'
    )
    # Add threshold line
    fig_irradiation.add_hline(
        y=0.4,
        line=dict(color='red', width=1, dash='dash'),
        name='Threshold (0.4)'
    )
    fig_irradiation.update_layout(
        title='Irradiation',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#ff9800'),
        xaxis=dict(
            title='Date',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        yaxis=dict(
            title='Irradiation',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_irradiation = pio.to_html(fig_irradiation, full_html=False)

    # Ambient Temperature Chart
    fig_ambient = go.Figure()
    fig_ambient.add_scatter(
        x=filtered_data['DATE_TIME'],
        y=filtered_data['AMBIENT_TEMPERATURE'],
        mode='lines',
        line=dict(color='#03dac6', width=2),
        name='Ambient Temperature'
    )
    # Add threshold line
    fig_ambient.add_hline(
        y=30,
        line=dict(color='red', width=1, dash='dash'),
        name='Threshold (30°C)'
    )
    fig_ambient.update_layout(
        title='Ambient Temperature',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#03dac6'),
        xaxis=dict(
            title='Date',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        yaxis=dict(
            title='Ambient Temperature',
            gridcolor='#2d2d2d',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.3,
            x=0.5,
            xanchor='center'
        ),
        width=1200,
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        autosize=False,
    )
    graph_ambient = pio.to_html(fig_ambient, full_html=False)

    # Ensure proper split
    split_index = int(0.8 * len(filtered_data))
    first_80_data = filtered_data.iloc[:split_index]
    last_20_data = filtered_data.iloc[split_index:]

    # Forecast future
    future = dc_power_prophet_model.make_future_dataframe(periods=5 * 24 * 4, freq='15T')
    forecast = dc_power_prophet_model.predict(future)
    forecast_future = forecast[forecast['ds'] > filtered_data['DATE_TIME'].max()]

    # Plot setup
    fig_dc_power = go.Figure()

    # Train Data (Neon)
    fig_dc_power.add_scatter(
        x=first_80_data['DATE_TIME'],
        y=first_80_data['DC_POWER'],
        mode='lines',
        line=dict(color='#03dac6'),  # Neon green
        name='Train Data (80%)'
    )

    # Test Data (Blue)
    fig_dc_power.add_scatter(
        x=last_20_data['DATE_TIME'],
        y=last_20_data['DC_POWER'],
        mode='lines',
        line=dict(color='#2196f3'),  # Blue
        name='Test Data (20%)'
    )

    # Forecast (Orange Dotted)
    fig_dc_power.add_scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines',
        line=dict(color='#ff9800', dash='dot'),
        name='Forecast (Next 5 Days)'
    )

    # Layout
    fig_dc_power.update_layout(
        title='DC Power - Train, Test, and Forecast',
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center'),
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        title_font=dict(color='#bb86fc'),
        width=1400,  # Increased width
        height=600,  # Maintained height
        margin=dict(l=50, r=50, t=50, b=100)  # Added margins for better visibility
    )

    graph_dc_power = pio.to_html(fig_dc_power, full_html=False)
    return render_template("index.html", 
        accuracy=accuracy,
        power_graph=graph_power,
        dc_power_graph=graph_dc_power,
        env_graphs=[graph_module, graph_irradiation, graph_ambient],
        daily_energy_graph=graph_daily_energy,
        hourly_temp_graph=graph_hourly_temp,
        min_max_temp_graph=graph_min_max_temp,
        efficiency_graph=graph_efficiency,
        heatmap_graph=graph_heatmap,
        temp_heatmap_graph=graph_temp_heatmap,
        table=anomaly_points[['DATE_TIME', 'REASON_DC_POWER', 'PRESCRIPTION_DC_POWER']].to_dict('records')
    )

@app.route('/download')
def download():
    return send_file(csv_path, as_attachment=True)

@app.route('/stream')
def stream():
    return Response(generate_realtime_data(), mimetype='text/event-stream')

@app.route('/get_forecast/<parameter>/<int:days>')
def get_forecast(parameter, days):
    model_map = {
        'dc_power': (dc_power_prophet_model, 'DC Power Forecast', '#03dac6'),
        'ambient': (ambient_prophet_model, 'Ambient Temperature Forecast', '#ff9800'),
        'irradiation': (irradiation_prophet_model, 'Irradiation Forecast', '#4caf50'),
        'module': (module_prophet_model, 'Module Temperature Forecast', '#2196f3')
    }
    
    if parameter not in model_map:
        return jsonify({'error': 'Invalid parameter'}), 400
    
    model, title, color = model_map[parameter]
    fig = create_forecast_chart(model, days, title, color)
    
    return jsonify({
        'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    })

if __name__ == '__main__':
    app.run(debug=True)









