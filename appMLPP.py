import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

# === Load historical data ===
def load_data(ticker, period='2y'):
    df = yf.download(ticker, period=period, interval='1d', group_by='column', auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# === Tambahkan indikator teknikal dan fitur ===
def add_all_features(df):
    for period in [5, 10, 21, 50, 100, 200]:
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() /
                             (-df['Close'].diff().clip(upper=0).rolling(14).mean())))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['MA20'] + 2 * df['STD20']
    df['LowerBB'] = df['MA20'] - 2 * df['STD20']
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR14'] = df['TR'].rolling(window=14).mean()
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * (df['Close'] - low14) / ((high14 - low14) + 1e-10)
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (df['TP'] - df['TP'].rolling(20).mean()) / (0.015 * df['TP'].rolling(20).std())
    df['ROC'] = df['Close'].pct_change(periods=12) * 100
    df['Williams_%R'] = -100 * ((high14 - df['Close']) / (high14 - low14 + 1e-10))

    for lag in range(1, 6):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_lag_{lag}'] = df['Close'].pct_change().shift(lag)
        df[f'Direction_lag_{lag}'] = (df['Close'].pct_change().shift(lag) > 0).astype(int)

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Price_Range'] = df['High'] - df['Low']
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Day_of_Week'] = df.index.dayofweek
    df['MACD_RSI'] = df['MACD'] * df['RSI']
    df['ADX_Body'] = df['ATR14'] * df['Body_Size']
    df['Month'] = df.index.month
    df['Is_Month_Start'] = df.index.is_month_start.astype(int)
    df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)
    return df

# === Load model, scaler, selector, dan fitur ===
def load_artifacts():
    model = joblib.load("xgb_model.joblib")
    scaler = joblib.load("scaler.joblib")
    selector = joblib.load("selector.joblib")
    features = joblib.load("features.joblib")
    return model, scaler, selector, features

# === Streamlit App ===
st.set_page_config(page_title="Prediksi Harga Aset", layout="wide")
st.title("📈 Prediksi Harga Penutupan Aset - Streamlit")

with st.sidebar:
    ticker = st.text_input("Masukkan Ticker (misal: AAPL, BTC-USD, etc)", value="AAPL")

if ticker:
    with st.spinner("Mengambil dan memproses data..."):
        df = load_data(ticker)
        df = add_all_features(df)
        df.dropna(inplace=True)
        model, scaler, selector, features = load_artifacts()
        missing = [col for col in features if col not in df.columns]
        if missing:
            st.error(f"❌ Kolom berikut tidak ditemukan dalam DataFrame: {missing}")
            st.write("Kolom tersedia:", list(df.columns))
            st.write("Fitur yang diminta:", features)
            st.stop()

        X = df[features]
        X_scaled = scaler.transform(X)
        X_selected = selector.transform(X_scaled)
        y_pred_return = model.predict(X_selected[-1:])[0]
        last_close = df['Close'].iloc[-1]
        pred_price = float(last_close * (1 + y_pred_return))
        arah = "⬆️ Naik" if y_pred_return > 0 else "⬇️ Turun"

    st.metric("Prediksi Harga Besok", f"${pred_price:.2f}", delta=f"{pred_price - last_close:.2f} ({arah})")

    # === Visualisasi
    fig = go.Figure()
    try:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Harga Close'))
        fig.add_trace(go.Scatter(
            x=[df.index[-1] + timedelta(days=1)],
            y=[pred_price],
            name='Prediksi Besok',
            mode='markers+text',
            text=[f"${pred_price:.2f}"],
            textposition="top center",
            marker=dict(color='red', size=10)
        ))
        fig.update_layout(title=f"{ticker} - Harga Penutupan & Prediksi Besok", xaxis_title="Tanggal", yaxis_title="Harga")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("⚠️ Gagal menampilkan grafik.")
        st.text(str(e))

    with st.expander("Lihat Data Terbaru"):
        st.dataframe(df.tail(10))
