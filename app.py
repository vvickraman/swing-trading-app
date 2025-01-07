# Import necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Function to get the swing trading signal
def swing_trade_signal(ticker, model, lookback=60):
    df = yf.download(ticker, period="6mo")

    if df.empty or len(df) < lookback:
        return {"Ticker": ticker, "Error": f"Not enough data to make a prediction for {ticker}."}

    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                  -df['Close'].diff().clip(upper=0).rolling(14).mean())))

    # Fetch latest data
    latest_close_prices = df['Close'].values[-lookback:].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    latest_scaled = scaler.fit_transform(latest_close_prices)
    latest_input = latest_scaled.reshape(1, lookback, 1)

    # Predict future price
    predicted_price_scaled = model.predict(latest_input)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]

    current_price = latest_close_prices[-1][0]

    # Decision based on indicators
    recommendation = "Hold"
    if predicted_price > current_price * 1.05 and df['RSI'].iloc[-1] < 70:
        recommendation = "Long Call Option"
    elif predicted_price < current_price * 0.95 and df['RSI'].iloc[-1] > 30:
        recommendation = "Long Put Option"

    return {
        "Ticker": ticker,
        "Current Price": current_price,
        "Predicted Price": predicted_price,
        "RSI": df['RSI'].iloc[-1],
        "Recommendation": recommendation
    }

# Streamlit web app
def main():
    st.title("Swing Trading Recommendations")
    st.subheader("Select Stocks to Evaluate:")

    # Path to saved models (assumes models are uploaded in the same folder as app.py)
    model_path = "./"  # Upload models in the same folder as this file
    stock_list = ["AAPL", "TSLA", "MSFT", "NVDA", "PLTR", "GOOG", "SOUN", "META", "AVGO", "AMD", "AMZN"]

    # Multiselect widget for stock tickers
    selected_stocks = st.multiselect("Select Stocks", stock_list, default=stock_list)

    if st.button("Get Recommendations"):
        recommendations = []
        for ticker in selected_stocks:
            model_file = os.path.join(model_path, f"{ticker}_model.h5")

            if not os.path.exists(model_file):
                recommendations.append({"Ticker": ticker, "Error": f"Model file not found for {ticker}"})
                continue

            # Load the model and make a prediction
            model = load_model(model_file)
            signal = swing_trade_signal(ticker, model)
            recommendations.append(signal)

        # Convert recommendations to DataFrame and display it
        df = pd.DataFrame(recommendations)
        st.write("Swing Trading Recommendations:")
        st.dataframe(df)

if __name__ == "__main__":
    main()
