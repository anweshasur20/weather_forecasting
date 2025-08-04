import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# === CONFIG ===
API_KEY = "a961a5dad0162cdf5fab3830ea1737bf"  # your real key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
HISTORICAL_DATA_PATH = "weather(1).csv"  # put your file in same folder or adjust path

# === Streamlit UI ===
st.title("🌦️ Weather Forecast & Rain Prediction")
city = st.text_input("Enter a city name", "Mumbai")

if city:
    # --- 1. Fetch Current Weather ---
    def get_current_weather(city):
        url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code != 200 or "main" not in response.json():
            return None
        data = response.json()
        return {
            'city': data.get('name', city),
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'pressure': round(data['main']['pressure']),
            'description': data['weather'][0]['description'].capitalize(),
            'wind_gust_dir': data.get('wind', {}).get('deg', 0) % 360,
            'wind_gust_speed': data.get('wind', {}).get('speed', 0)
        }

    def deg_to_compass(wind_deg):
        wind_deg = wind_deg % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75), ("N", 348.75, 360)
        ]
        for label, start, end in compass_points:
            if start <= wind_deg < end:
                return label
        return "N"

    def read_historical_data(filename):
        df = pd.read_csv(filename)
        df = df.dropna().drop_duplicates()
        return df

    def prepare_data(data):
        le = LabelEncoder()
        data = data.copy()
        data['WindGustDir_encoded'] = le.fit_transform(data['WindGustDir'])
        data['RainTomorrow_encoded'] = le.fit_transform(data['RainTomorrow'])
        feature_cols = ['Mintemp', 'Maxtemp', 'WindGustDir_encoded', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']
        x = data[feature_cols]
        y = data['RainTomorrow_encoded']
        return x, y, le

    def train_rain_model(x, y):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x, y)
        return model

    def prepare_regression_data(data, feature):
        data = data.copy().reset_index(drop=True)
        x = np.arange(len(data)).reshape(-1, 1)
        y = data[feature].values
        return x, y

    def train_regression_model(x, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x, y)
        return model

    def predict_future(model, current_index, steps=5):
        future_indices = np.arange(current_index + 1, current_index + 1 + steps).reshape(-1, 1)
        return model.predict(future_indices)

    # === MAIN LOGIC ===
    current = get_current_weather(city)

    if current:
        st.subheader(f"📍 Current Weather in {current['city']}")
        st.write(f"🌡️ Temperature: {current['current_temp']}°C (Feels like {current['feels_like']}°C)")
        st.write(f"🔻 Min: {current['temp_min']}°C | 🔺 Max: {current['temp_max']}°C")
        st.write(f"💧 Humidity: {current['humidity']}% | 🧭 Wind Direction: {deg_to_compass(current['wind_gust_dir'])}")
        st.write(f"📋 Description: {current['description']}")
        
        with st.spinner("🔄 Processing predictions..."):
            df = read_historical_data(HISTORICAL_DATA_PATH)
            x_rain, y_rain, le_rain = prepare_data(df)
            rain_model = train_rain_model(x_rain, y_rain)

            wind_label = deg_to_compass(current['wind_gust_dir'])
            try:
                wind_encoded = le_rain.transform([wind_label])[0]
            except:
                wind_encoded = 0  # fallback

            current_df = pd.DataFrame([{
                'Mintemp': current['temp_min'],
                'Maxtemp': current['temp_max'],
                'WindGustDir_encoded': wind_encoded,
                'WindGustSpeed': current['wind_gust_speed'],
                'Humidity': current['humidity'],
                'Pressure': current['pressure'],
                'Temp': current['current_temp']
            }])

            rain_pred_code = rain_model.predict(current_df)[0]
            try:
                rain_pred = le_rain.inverse_transform([rain_pred_code])[0]
            except:
                rain_pred = "Yes" if rain_pred_code == 1 else "No"

            st.markdown(f"🌧️ **Rain Prediction:** `{rain_pred}`")

            x_temp, y_temp = prepare_regression_data(df, 'Temp')
            temp_model = train_regression_model(x_temp, y_temp)

            x_hum, y_hum = prepare_regression_data(df, 'Humidity')
            hum_model = train_regression_model(x_hum, y_hum)

            idx = len(df) - 1
            future_temp = predict_future(temp_model, idx, 5)
            future_hum = predict_future(hum_model, idx, 5)

            timezone = pytz.timezone('Asia/Karachi')
            now = datetime.now(timezone)
            next_hour = now + timedelta(hours=1)
            next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
            times = [(next_hour + timedelta(hours=i)).strftime("%H:%M") for i in range(5)]

            st.subheader("📈 Forecast (Next 5 Hours)")
            st.write("Temperature (°C):")
            st.dataframe(pd.DataFrame({'Time': times, 'Temp': future_temp.round(2)}))

            st.write("Humidity (%):")
            st.dataframe(pd.DataFrame({'Time': times, 'Humidity': future_hum.round(2)}))

    else:
        st.error("Could not retrieve weather data. Please check the city name.")

