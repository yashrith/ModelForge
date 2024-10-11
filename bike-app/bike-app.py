import streamlit as st
import numpy as np
import pandas as pd
import pickle


def transform_cyclical(value, count):
    sin_value = np.sin(2 * np.pi * value / count)
    cos_value = np.cos(2 * np.pi * value / count)
    return sin_value, cos_value
    

def main():
    st.title(":bike: Bike Rental Predictor")
    holiday = int(st.sidebar.toggle("Holiday"))
    workingday = int(st.sidebar.toggle("Workingday"))

    weather_options = {1: "Clear",
                       2: "Mist",
                       3: "Light Rain",
                       4: "Heavy Rain"}
    weather = st.sidebar.selectbox("Weather",
                                   options=weather_options.keys(),
                                   format_func=lambda x: weather_options[x])
    temp = st.sidebar.slider("Temp", min_value=0, max_value=50)
    atemp = st.sidebar.slider("Feels-Like Temp", min_value=0, max_value=50)
    humidity = st.sidebar.slider("Humidity", min_value=0, max_value=100)
    windspeed = st.sidebar.slider("Windspeed", min_value=0, max_value=100)

    season_options = {1: "Spring",
                       2: "Summer",
                       3: "Fall",
                       4: "Winter"}
    season = st.sidebar.selectbox("Season",
                                   options=season_options.keys(),
                                   format_func=lambda x: season_options[x])
    date = st.sidebar.date_input("Date")
    month = date.month
    day = date.day
    weekday = date.weekday()

    season_sin, season_cos = transform_cyclical(season, 4)
    month_sin, month_cos = transform_cyclical(month, 12)
    day_sin, day_cos = transform_cyclical(day, 19)
    weekday_sin, weekday_cos = transform_cyclical(weekday, 7)

    columns = ['holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity',
       'windspeed', 'season_sin', 'season_cos', 'month_sin', 'month_cos',
       'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos', 'hour_sin',
       'hour_cos']

    data = []

    for hour in range(24):
        hour_sin, hour_cos = transform_cyclical(hour, 24)
        data.append([holiday, workingday, weather, temp, atemp, humidity,
                    windspeed, season_sin, season_cos, month_sin, month_cos,
                    day_sin, day_cos, weekday_sin, weekday_cos,
                    hour_sin, hour_cos])

    st.subheader("Original Data")
    df = pd.DataFrame(data=data, columns=columns)
    st.write(df)

    st.subheader("Scaling")
    features_to_scale = ["weather", "temp", "atemp", "humidity", "windspeed"]
    scaler = pickle.load(open("bike-scaler.pkl", "rb"))
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    st.write(df)

    st.subheader("Prediction")
    model = pickle.load(open("bike-model.pkl", "rb"))
    predictions = model.predict(df)
    predictions = np.expm1(predictions)
    # st.write(predictions)
    st.line_chart(predictions, x_label="Hour", y_label="# of Customers")
    

if __name__ == "__main__":
    main()
    