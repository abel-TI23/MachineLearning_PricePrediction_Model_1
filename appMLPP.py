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
st.set_page_config(page_title="Prediksi Harga Aset Real-time", layout="wide")

# === 2. Fungsi-Fungsi Inti (Helper Functions) ===

# Mengambil data historis dari Yahoo Finance (dengan cache untuk efisiensi)
@st.cache_data
def load_historical_data(ticker, period='10y'): # PERUBAHAN: Periode diubah ke 10 tahun
    """Mengunduh data historis untuk periode yang lebih panjang."""
    st.info(f"Mengunduh data historis ({period}) untuk **{ticker}**...")
    df = yf.download(ticker, period=period, interval='1d')
    if df.empty:
        st.error("Gagal mengunduh data historis. Pastikan ticker valid.")
        return None
    # Menambahkan pengecekan jumlah data yang diunduh
    if len(df) < 250:
        st.warning(f"Data yang diunduh untuk {ticker} hanya {len(df)} baris. Mungkin tidak cukup untuk training.")
    return df

# Fungsi baru untuk mengambil data real-time
@st.cache_data(ttl=300) # Cache data real-time selama 5 menit
def get_current_data(ticker):
    """Mengunduh data terbaru (intraday) untuk hari ini."""
    st.info(f"Mengambil data real-time untuk **{ticker}**...")
    today_df = yf.Ticker(ticker).history(period='2d', interval='1d')
    if today_df.empty:
        st.warning("Tidak dapat mengambil data real-time. Prediksi akan didasarkan pada data penutupan terakhir.")
        return None
    # Mengembalikan hanya baris terakhir
    return today_df.iloc[-1:].copy()

# Menambahkan fitur-fitur teknikal ke DataFrame
@st.cache_data
def add_all_features(_df):
    st.info("Menambahkan fitur teknikal...")
    df = _df.copy()
    
    # PERBAIKAN: Standarkan indeks ke UTC untuk mengatasi error timezone
    df.index = pd.to_datetime(df.index, utc=True)
    
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

# Fungsi training dirombak untuk memisahkan prediksi real-time
def train_and_predict(df):
    """
    Melatih model pada data historis dan memprediksi hari berikutnya
    berdasarkan data baris terakhir (real-time).
    """
    st.info("Memulai proses pelatihan dan prediksi...")
    
    # 1. Pisahkan data: data untuk prediksi besok (baris terakhir) dan data training (semua kecuali baris terakhir)
    prediction_input_df = df.iloc[-1:].copy()
    train_df = df.iloc[:-1].copy()
    
    # 2. Siapkan data training
    train_df.dropna(inplace=True)
    if train_df.empty or len(train_df) < 50:
        st.error("Data historis tidak cukup untuk melatih model setelah membersihkan data. Coba periode yang lebih panjang.")
        return None, None, None

    X_train_full = train_df.drop(['Target', 'Close'], axis=1, errors='ignore').select_dtypes(include=np.number)
    y_train_full = train_df['Target']
    
    # 3. Scaling dan Seleksi Fitur (fit hanya pada data training)
    st.info("Menyesuaikan skala dan memilih fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)

    selector_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train_scaled, y_train_full)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    
    X_train_selected = selector.transform(X_train_scaled)

    # 4. Latih Model Utama
    st.info("Melatih model utama...")
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train_full)
    st.success("✅ Model berhasil dilatih!")

    # 5. Buat Prediksi Historis (untuk grafik)
    historical_preds_pct = model.predict(X_train_selected)
    
    # 6. Siapkan data real-time dan buat prediksi untuk besok
    st.info("Membuat prediksi untuk hari berikutnya...")
    X_predict = prediction_input_df.drop(['Target', 'Close'], axis=1, errors='ignore').select_dtypes(include=np.number)
    
    # Pastikan kolomnya sama persis dengan data training
    X_predict = X_predict[X_train_full.columns]

    X_predict_scaled = scaler.transform(X_predict)
    X_predict_selected = selector.transform(X_predict_scaled)
    
    prediction_for_tomorrow_pct = model.predict(X_predict_selected)[0]
    
    return train_df, historical_preds_pct, prediction_for_tomorrow_pct


# === 3. Tampilan Aplikasi (UI) ===

# Judul Utama Aplikasi
st.title("📈 Prediksi Harga Penutupan Aset dengan Data Real-time")
st.caption("Memadukan data historis dan harga terkini untuk prediksi yang lebih akurat.")

# Sidebar untuk input pengguna
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    ticker_input = st.text_input("Masukkan Ticker (misal: BBCA.JK, BTC-USD, ETH-USD)", value="BTC-USD")
    show_history = st.checkbox("Tampilkan Prediksi Historis", value=True)
    train_button = st.button("Latih & Prediksi Harga Besok", type="primary", use_container_width=True)

# Area utama
if train_button:
    # Selalu bersihkan cache untuk mendapatkan data terbaru saat tombol ditekan
    st.cache_data.clear()
    
    st.session_state.ticker = ticker_input
    st.session_state.train_success = False # Reset status
    
    with st.spinner(f"Memproses data dan melatih model untuk **{ticker_input}**..."):
        try:
            # Alur pengambilan data baru
            historical_df = load_historical_data(st.session_state.ticker)
            if historical_df is not None:
                current_df = get_current_data(st.session_state.ticker)
                
                # Gabungkan data historis dengan data hari ini
                if current_df is not None:
                    # Menghapus baris terakhir dari historis jika tanggalnya sama dengan data saat ini
                    if not historical_df.empty and historical_df.index[-1].date() == current_df.index[0].date():
                        historical_df = historical_df.iloc[:-1]
                    combined_df = pd.concat([historical_df, current_df])
                else:
                    combined_df = historical_df
                
                # Pastikan tidak ada duplikat indeks, ambil yang terakhir (paling update)
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                st.info(f"Data berhasil digabungkan. Titik data terakhir pada: **{combined_df.index[-1].strftime('%Y-%m-%d %H:%M')}**")
                
                featured_df = add_all_features(combined_df)
                
                # Panggil fungsi training dan prediksi yang baru
                train_df, hist_preds, tomorrow_pred = train_and_predict(featured_df)
                
                if train_df is not None:
                    st.session_state.results = {
                        "train_df": train_df,
                        "hist_preds": hist_preds,
                        "tomorrow_pred": tomorrow_pred,
                        "last_close": combined_df['Close'].iloc[-1],
                        "last_date": combined_df.index[-1]
                    }
                    st.session_state.train_success = True

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan fatal selama proses:")
            st.error(str(e))
            st.code(traceback.format_exc())
            st.session_state.train_success = False

# Tampilkan hasil jika pelatihan berhasil
if 'train_success' in st.session_state and st.session_state.train_success:
    results = st.session_state.results
    ticker = st.session_state.ticker

    # --- Tampilkan Metric Prediksi Besok ---
    last_close = results['last_close']
    tomorrow_pred_pct = results['tomorrow_pred']
    
    pred_price_tomorrow = last_close * (1 + tomorrow_pred_pct)
    delta_value = pred_price_tomorrow - last_close
    
    st.metric(
        f"Prediksi Harga Besok untuk {ticker}",
        f"${pred_price_tomorrow:,.4f}",
        f"{delta_value:,.4f} ({tomorrow_pred_pct:.2%}) {'⬆️' if delta_value > 0 else '⬇️'}"
    )
    st.divider()

    # --- Buat dan Tampilkan Chart ---
    st.subheader("Visualisasi Harga dan Prediksi")
    
    plot_df = results['train_df'].copy()
    plot_df['Harga Prediksi'] = plot_df['Close'] * (1 + results['hist_preds'])

    fig = go.Figure()
    # Plot harga aktual historis
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name='Harga Aktual (Historis)', line=dict(color='deepskyblue')))
    
    # Plot harga terkini (titik terakhir)
    fig.add_trace(go.Scatter(
        x=[results['last_date']], y=[last_close], name='Harga Terkini',
        mode='markers', marker=dict(color='orange', size=10, symbol='diamond')
    ))

    if show_history:
        # Plot prediksi historis
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Harga Prediksi'], name='Prediksi Historis', line=dict(color='tomato', dash='dot', width=1.5)))

    # Plot prediksi untuk besok
    fig.add_trace(go.Scatter(
        x=[results['last_date'] + timedelta(days=1)], y=[pred_price_tomorrow], name='Prediksi Besok',
        mode='markers', marker=dict(color='red', size=15, symbol='star')
    ))

    fig.update_layout(
        title=f"{ticker} - Harga Aktual vs. Prediksi",
        xaxis_title="Tanggal (UTC)", yaxis_title="Harga", template='plotly_dark',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Tampilkan Tabel Data ---
    st.subheader("Data Terakhir yang Digunakan untuk Training")
    display_df = results['train_df'].copy()
    display_df['Prediksi_Perubahan_Persen'] = results['hist_preds']
    display_df['Harga_Prediksi'] = display_df['Close'] * (1 + display_df['Prediksi_Perubahan_Persen'])
    st.dataframe(display_df[['Close', 'Harga_Prediksi', 'Prediksi_Perubahan_Persen', 'Volume']].sort_index(ascending=False).head(10))
