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
import time

# === 1. Konfigurasi Halaman Streamlit ===
st.set_page_config(page_title="Prediksi Harga Aset", layout="wide")

# === 2. Fungsi-Fungsi Inti (Helper Functions) ===

# Mengambil data dari Yahoo Finance (dengan cache untuk efisiensi)
@st.cache_data
def load_data(ticker, period='5y'):
    st.toast(f"Mengunduh data untuk **{ticker}**...")
    df = yf.download(ticker, period=period, interval='1d')
    if df.empty:
        st.error(f"Gagal mengunduh data untuk **{ticker}**. Pastikan ticker valid.")
        return None
    df.dropna(inplace=True)
    return df

# Menambahkan fitur-fitur teknikal ke DataFrame
@st.cache_data
def add_all_features(_df):
    st.toast("Menambahkan fitur teknikal...")
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
    return df

# Fungsi untuk melatih model dari awal
def train_model(df):
    st.toast("Memulai proses pelatihan model...")
    df_processed = df.copy()
    df_processed.dropna(inplace=True)

    if df_processed.empty:
        st.error("Data tidak mencukupi untuk melatih model. Coba periode yang lebih panjang.")
        return None, None

    X = df_processed.drop(['Target', 'Close'], axis=1, errors='ignore')
    y = df_processed['Target']
    X = X.select_dtypes(include=np.number)
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    selector_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train_scaled, y_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    X_train_selected = selector.transform(X_train_scaled)

    st.toast("Melatih model utama (ini butuh waktu)...")
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train)
    
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    predictions_pct = model.predict(X_scaled)
    
    return predictions_pct, df_processed

# === 3. Tampilan Aplikasi (UI) ===

st.title("📈 Prediksi Harga Penutupan Aset")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    ticker_input = st.text_input(
        "Masukkan Ticker lalu tekan Enter", 
        value="BTC-USD", 
        key="ticker_input_main" # Key untuk text input
    )
    show_history = st.checkbox("Tampilkan Prediksi Historis", value=True, key="show_history_main")

# Inisialisasi session state untuk melacak ticker terakhir
if 'last_processed_ticker' not in st.session_state:
    st.session_state.last_processed_ticker = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Logika utama: Jalankan hanya jika ticker baru dimasukkan
if ticker_input and ticker_input != st.session_state.last_processed_ticker:
    # Hapus cache untuk memastikan data baru diproses
    load_data.clear()
    add_all_features.clear()
    
    with st.spinner(f"Memproses untuk **{ticker_input}**..."):
        try:
            raw_df = load_data(ticker_input)
            if raw_df is not None:
                featured_df = add_all_features(raw_df)
                predictions_pct, final_df = train_model(featured_df)
                
                if predictions_pct is not None and final_df is not None:
                    st.session_state.results = {
                        "df": final_df,
                        "predictions_pct": predictions_pct,
                        "ticker": ticker_input
                    }
                    st.toast("✅ Proses Selesai!", icon="🎉")
                else:
                    st.session_state.results = None # Hapus hasil jika gagal
            
            # Update ticker terakhir yang diproses
            st.session_state.last_processed_ticker = ticker_input
        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan fatal:")
            st.code(traceback.format_exc())
            st.session_state.results = None

# Tampilkan hasil jika ada di session state
if st.session_state.results:
    results = st.session_state.results
    df = results['df']
    predictions_pct = results['predictions_pct']
    ticker = results['ticker']

    if not df.empty and len(predictions_pct) > 0:
        # --- Tampilkan Metric Prediksi Besok ---
        last_close = float(df['Close'].iloc[-1])
        pred_price_tomorrow = float(last_close * (1 + predictions_pct[-1]))
        delta_value = float(pred_price_tomorrow - last_close)

        st.metric(
            f"Prediksi Harga Besok untuk {ticker}",
            f"${pred_price_tomorrow:,.2f}",
            delta=f"{delta_value:,.2f} ({'⬆️ Naik' if delta_value > 0 else '⬇️ Turun'})"
        )

        # --- Buat dan Tampilkan Chart ---
        plot_df = pd.DataFrame(index=df.index)
        plot_df['Harga Aktual'] = df['Close']
        predicted_prices = plot_df['Harga Aktual'] * (1 + predictions_pct)
        plot_df['Harga Prediksi'] = predicted_prices.shift(1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Harga Aktual'], name='Harga Close', line=dict(color='deepskyblue')))
        
        if show_history:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Harga Prediksi'], name='Prediksi Historis', line=dict(color='tomato', dash='dot')))

        fig.add_trace(go.Scatter(
            x=[df.index[-1] + timedelta(days=1)], y=[pred_price_tomorrow], name='Prediksi Besok',
            mode='markers', marker=dict(color='yellow', size=12, symbol='dot')
        ))

        fig.update_layout(
            title=f"{ticker} - Harga Penutupan & Prediksi Besok",
            xaxis_title="Tanggal", yaxis_title="Harga", template='plotly_dark',
            legend=dict(x=0.01, y=0.98, orientation='h')
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Tampilkan Tabel Data ---
        st.subheader("Lihat Data Terbaru")
        st.dataframe(df.sort_index(ascending=False))
else:
    st.info("Masukkan ticker di sidebar dan tekan Enter untuk memulai analisis.")

