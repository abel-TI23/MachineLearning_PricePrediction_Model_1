import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import plotly.graph_objects as go
from datetime import timedelta
import traceback

# === 1. Konfigurasi Halaman Streamlit ===
st.set_page_config(page_title="Prediksi Harga Aset Dinamis", layout="wide")
st.title("📈 Pelatihan & Prediksi Harga Aset Secara Dinamis")
st.write("""
Aplikasi ini memungkinkan Anda untuk melatih model prediksi harga aset secara langsung.
Masukkan ticker saham atau aset kripto, lalu klik tombol untuk memulai proses pelatihan
dan melihat hasil prediksinya dibandingkan dengan data historis.
""")

# === 2. Fungsi-Fungsi Inti (Helper Functions) ===

# Mengambil data dari Yahoo Finance (dengan cache untuk efisiensi)
@st.cache_data
def load_data(ticker, period='5y'):
    st.write(f"Mengunduh data untuk **{ticker}**...")
    df = yf.download(ticker, period=period, interval='1d')
    if df.empty:
        st.error("Gagal mengunduh data. Pastikan ticker valid.")
        return None
    df.dropna(inplace=True)
    st.write("✅ Data berhasil diunduh.")
    return df

# Menambahkan fitur-fitur teknikal ke DataFrame
@st.cache_data
def add_all_features(_df):
    st.write("Menambahkan fitur teknikal...")
    df = _df.copy()
    # Moving Averages
    for period in [5, 10, 21, 50, 100, 200]:
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['MA20'] + 2 * df['STD20']
    df['LowerBB'] = df['MA20'] - 2 * df['STD20']
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Fitur-fitur lain
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Price_Range'] = df['High'] - df['Low']
    df['Day_of_Week'] = df.index.dayofweek
    # Target variable (perubahan harga di hari berikutnya)
    df['Target'] = df['Close'].pct_change().shift(-1)
    st.write("✅ Fitur teknikal ditambahkan.")
    return df

# Fungsi untuk melatih model dari awal
def train_model(df):
    st.write("Memulai proses pelatihan model...")
    df_processed = df.copy()
    df_processed.dropna(inplace=True)

    X = df_processed.drop(['Target', 'Close'], axis=1, errors='ignore')
    y = df_processed['Target']
    
    # Memastikan tidak ada kolom non-numerik
    X = X.select_dtypes(include=np.number)
    features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    selector_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train_scaled, y_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    X_train_selected = selector.transform(X_train_scaled)

    st.write("Sedang melatih model utama (ini mungkin butuh waktu)...")
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train)
    
    st.success("✅ Model berhasil dilatih!")
    return model, scaler, selector, features, X, y

# Fungsi untuk membuat plot perbandingan
def create_comparison_chart(df, model, scaler, selector, features):
    st.write("Membuat prediksi historis untuk visualisasi...")
    X_full = df[features].copy()
    X_full_scaled = scaler.transform(X_full)
    X_full_selected = selector.transform(X_full_scaled)
    
    predictions_pct = model.predict(X_full_selected)
    
    # Membuat DataFrame untuk plot
    plot_df = pd.DataFrame(index=X_full.index)
    plot_df['Harga Aktual'] = df['Close']
    
    # Menghitung harga prediksi dari persentase perubahan
    predicted_prices = plot_df['Harga Aktual'] * (1 + predictions_pct)
    plot_df['Harga Prediksi'] = predicted_prices.shift(1) # Geser agar sejajar
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Harga Aktual'], name='Harga Aktual', line=dict(color='deepskyblue')))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Harga Prediksi'], name='Prediksi Historis Model', line=dict(color='tomato', dash='dot')))

    # Prediksi untuk besok
    last_close = df['Close'].iloc[-1]
    pred_price_tomorrow = float(last_close * (1 + predictions_pct[-1]))

    fig.add_trace(go.Scatter(
        x=[df.index[-1] + timedelta(days=1)], y=[pred_price_tomorrow], name='Prediksi Besok',
        mode='markers+text', text=[f"${pred_price_tomorrow:.2f}"], textposition="top center",
        marker=dict(color='red', size=12, symbol='star')
    ))

    fig.update_layout(title=f"Perbandingan Harga Aktual vs. Prediksi Model untuk {st.session_state.ticker}",
                      xaxis_title="Tanggal", yaxis_title="Harga", template='plotly_dark',
                      legend=dict(x=0.01, y=0.98))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan metric untuk prediksi besok
    st.metric(
        "Prediksi Harga Besok",
        f"${pred_price_tomorrow:.2f}",
        delta=f"{pred_price_tomorrow - last_close:.2f} ({'⬆️ Naik' if pred_price_tomorrow > last_close else '⬇️ Turun'})"
    )

# === 3. Tampilan Aplikasi (UI) ===

# Sidebar untuk input pengguna
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    ticker_input = st.text_input("Masukkan Ticker (misal: BBCA.JK, BTC-USD)", value="BBCA.JK")
    train_button = st.button("Latih Model & Tampilkan Prediksi", type="primary")

# Area utama
if train_button:
    st.session_state.ticker = ticker_input
    
    # Bungkus seluruh proses dalam spinner untuk UX yang lebih baik
    with st.spinner(f"Memproses data dan melatih model untuk **{ticker_input}**..."):
        try:
            # Langkah 1: Muat dan proses data
            raw_df = load_data(st.session_state.ticker)
            if raw_df is not None:
                featured_df = add_all_features(raw_df)
                
                # Langkah 2: Latih model
                model, scaler, selector, features, X, y = train_model(featured_df)
                
                # Simpan hasil ke session state agar tidak hilang saat rerun
                st.session_state.artifacts = {
                    "model": model, "scaler": scaler, "selector": selector,
                    "features": features, "df": featured_df
                }
                st.session_state.train_success = True

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan fatal selama proses:")
            st.error(str(e))
            st.code(traceback.format_exc())
            st.session_state.train_success = False

# Tampilkan chart jika pelatihan berhasil
if 'train_success' in st.session_state and st.session_state.train_success:
    st.header("📊 Hasil Visualisasi")
    artifacts = st.session_state.artifacts
    create_comparison_chart(
        artifacts['df'], artifacts['model'], artifacts['scaler'],
        artifacts['selector'], artifacts['features']
    )
