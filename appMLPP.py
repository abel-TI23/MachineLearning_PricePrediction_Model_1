import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go
import os

# === Direktori untuk simpan model di cloud ===
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Sidebar ===
st.sidebar.title("Prediksi Harga Saham")
ticker = st.sidebar.text_input("Masukkan Ticker (misal: AAPL, BTC-USD, etc)", "AAPL")
show_historical_prediksi = st.sidebar.checkbox("Tampilkan Grafik Prediksi Historis", value=True)
train_model = st.sidebar.button("Train Model Ulang")

# === Load Data ===
@st.cache_data
def load_data(ticker='AAPL', start='2018-01-01'):
    df = yf.download(ticker, start=start)
    df.dropna(inplace=True)
    return df

df = load_data(ticker)
st.title(f"Prediksi Harga Besok")

# === Tambahkan Fitur Teknikal ===
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

# === Feature & Target ===
X = df_feat[['Return', 'MA5', 'MA10', 'Lag1', 'Lag2']]
y = df_feat['Close'].shift(-1).dropna()
X = X.iloc[:-1, :]  # agar panjang X dan y sama

# === Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Path file model ===
model_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_model.joblib")
scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
pred_path = os.path.join(MODEL_DIR, f"{ticker}_y_pred_all.joblib")

# === Train Model ===
model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)

if train_model:
    model.fit(X_scaled, y)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    y_pred_all = pd.Series(model.predict(X_scaled), index=y.index)
    joblib.dump(y_pred_all, pred_path)
    st.success("✅ Model berhasil dilatih ulang.")
else:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        st.warning("⚠️ Model belum tersedia. Silakan klik 'Train Model Ulang'.")
        st.stop()

# === Prediksi Harga Besok ===
latest_feat = df_feat[['Return', 'MA5', 'MA10', 'Lag1', 'Lag2']].iloc[-1:]
latest_scaled = scaler.transform(latest_feat)
pred_tomorrow = model.predict(latest_scaled)[0]

# === Hitung Delta dan Arah ===
latest_close = df['Close'].iloc[-1]
delta = float(pred_tomorrow - latest_close)

# === Tampilkan Arah dan Nilai ===
st.markdown(f"""
    <h3>Prediksi Harga Besok</h3>
    <span style='color:{"green" if delta > 0 else "red"}; font-size:24px'>
    {"▲" if delta > 0 else "▼"} {abs(delta):.2f}
    </span>
""", unsafe_allow_html=True)

# === Grafik Harga & Prediksi ===
df = df.reset_index()

fig = go.Figure()

# Harga aktual
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Close'],
    mode='lines+markers',
    name='Harga Close',
    line=dict(color='deepskyblue')
))

# Prediksi Historis
if show_historical_prediksi and os.path.exists(pred_path):
    y_pred_all = joblib.load(pred_path)
    fig.add_trace(go.Scatter(
        x=df['Date'].iloc[-len(y_pred_all):],
        y=y_pred_all,
        mode='lines',
        name='Harga Prediksi Historis',
        line=dict(color='orange', dash='dot')
    ))

# Titik Prediksi Besok
fig.add_trace(go.Scatter(
    x=[df['Date'].iloc[-1] + pd.Timedelta(days=1)],
    y=[pred_tomorrow],
    mode='markers+text',
    name='Prediksi Besok',
    marker=dict(size=12, color='red', symbol='triangle-down'),
    text=[f"{pred_tomorrow:.2f}"],
    textposition="bottom center"
))

fig.update_layout(
    title=f"{ticker.upper()} - Harga Penutupan & Prediksi Besok",
    xaxis_title="Tanggal",
    yaxis_title="Harga",
    template='plotly_dark',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)
