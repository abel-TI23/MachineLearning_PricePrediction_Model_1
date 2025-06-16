import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go
import datetime

# === Sidebar ===
st.sidebar.title("Prediksi Harga Saham")
ticker = st.sidebar.text_input("Masukkan Ticker (misal: AAPL, BTC-USD, etc)", "AAPL")
show_historical_prediksi = st.sidebar.checkbox("Tampilkan Grafik Prediksi Historis", value=True)
train_model = st.sidebar.button("Train Model Ulang")

# === Load data ===
@st.cache_data
def load_data(ticker='AAPL', start='2018-01-01'):
    df = yf.download(ticker, start=start)
    df.dropna(inplace=True)
    return df

df = load_data(ticker)
st.title(f"Prediksi Harga Besok")

# === Fitur teknikal sederhana + lag ===
def add_features(data):
    df = data.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df.dropna(inplace=True)
    return df

df_feat = add_features(df)

# === Split feature & target ===
X = df_feat[['Return', 'MA5', 'MA10', 'Lag1', 'Lag2']]
y = df_feat['Close'].shift(-1).dropna()
X = X.iloc[:-1, :]  # agar X dan y sama panjang

# === Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train Model ===
model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
if train_model:
    model.fit(X_scaled, y)

    # Simpan model & scaler
    joblib.dump(model, "xgb_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    # Simpan prediksi historis
    y_pred_all = pd.Series(model.predict(X_scaled), index=y.index)
    joblib.dump(y_pred_all, "y_pred_all.joblib")

# === Load Model & Scaler jika tidak training ulang ===
else:
    try:
        model = joblib.load("xgb_model.joblib")
        scaler = joblib.load("scaler.joblib")
    except:
        st.error("Model belum dilatih. Silakan klik 'Train Model Ulang'.")

# === Prediksi Harga Besok ===
latest_feat = df_feat[['Return', 'MA5', 'MA10', 'Lag1', 'Lag2']].iloc[-1:]
latest_scaled = scaler.transform(latest_feat)
pred_tomorrow = model.predict(latest_scaled)[0]

# === Hitung selisih dan arah ===
latest_close = df['Close'].iloc[-1]
delta = pred_tomorrow - latest_close
arah = "Naik" if delta.item() > 0 else "Turun"

delta_value = float(delta)  # atau delta.item() jika yakin Series-nya hanya 1 elemen

# === Tampilkan Hasil Prediksi ===
st.markdown(f"""
    <span style='color:{"green" if delta_value > 0 else "red"}'>
    {"▲" if delta_value > 0 else "▼"} {abs(delta_value):.2f}
    </span>
""", unsafe_allow_html=True)


# === Grafik Harga & Prediksi Besok ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Harga Close"))

# Titik prediksi besok
fig.add_trace(go.Scatter(
    x=[df.index[-1] + pd.Timedelta(days=1)],
    y=[pred_tomorrow],
    name="Prediksi Besok",
    mode="markers",
    marker=dict(size=10, color='red')
))

# === Tambahkan Prediksi Historis jika diminta ===
if show_historical_prediksi:
    try:
        y_pred_all = joblib.load("y_pred_all.joblib")
        fig.add_trace(go.Scatter(
            x=y_pred_all.index,
            y=y_pred_all.values,
            name="Harga Prediksi Historis",
            line=dict(color='orange', dash='dot')
        ))
    except Exception as e:
        st.warning("⚠️ Prediksi historis tidak tersedia.")
        st.text(str(e))

fig.update_layout(title=f"{ticker} - Harga Penutupan & Prediksi Besok",
                  xaxis_title="Tanggal", yaxis_title="Harga")

st.plotly_chart(fig, use_container_width=True)
