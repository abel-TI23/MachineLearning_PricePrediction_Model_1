import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
from datetime import timedelta
import traceback

# === 1. Konfigurasi Halaman Streamlit ===
st.set_page_config(page_title="Prediksi Harga Aset dengan Evaluasi", layout="wide")

# === 2. Fungsi-Fungsi Inti (Helper Functions) ===

@st.cache_data
def load_historical_data(ticker, period='10y'):
    """Mengunduh data historis untuk periode yang lebih panjang."""
    st.info(f"Mengunduh data historis ({period}) untuk **{ticker}**...")
    df = yf.download(ticker, period=period, interval='1d')
    if df.empty:
        st.error("Gagal mengunduh data historis. Pastikan ticker valid.")
        return None
    if len(df) < 250:
        st.warning(f"Data yang diunduh untuk {ticker} hanya {len(df)} baris. Mungkin tidak cukup untuk training.")
    return df

@st.cache_data(ttl=300)
def get_current_data(ticker):
    """Mengunduh data terbaru (intraday) untuk hari ini."""
    st.info(f"Mengambil data real-time untuk **{ticker}**...")
    today_df = yf.Ticker(ticker).history(period='2d', interval='1d')
    if today_df.empty:
        st.warning("Tidak dapat mengambil data real-time. Prediksi akan didasarkan pada data penutupan terakhir.")
        return None
    return today_df.iloc[-1:].copy()

@st.cache_data
def add_all_features(_df):
    """Menambahkan semua indikator teknikal yang dibutuhkan ke DataFrame."""
    st.info("Menambahkan fitur teknikal...")
    df = _df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    
    for period in [5, 10, 21, 50, 100, 200]:
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['MA20'] + 2 * df['STD20']
    df['LowerBB'] = df['MA20'] - 2 * df['STD20']

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Price_Range'] = df['High'] - df['Low']
    df['Day_of_Week'] = df.index.dayofweek
    df['Target'] = df['Close'].pct_change().shift(-1)
    return df

# === PERBAIKAN: Fungsi training dirombak untuk menyertakan evaluasi ===
def train_evaluate_and_predict(df):
    """
    Melatih, mengevaluasi model, dan membuat prediksi.
    """
    st.info("Memulai proses pelatihan, evaluasi, dan prediksi...")
    
    prediction_input_df = df.iloc[-1:].copy()
    train_val_df = df.iloc[:-1].copy()
    train_val_df.dropna(inplace=True)

    if train_val_df.empty or len(train_val_df) < 50:
        st.error("Data historis tidak cukup untuk melatih model setelah membersihkan data.")
        # PERBAIKAN: Mengembalikan 6 nilai None agar sesuai dengan ekspektasi unpack
        return None, None, None, None, None, None

    X = train_val_df.drop(['Target', 'Close'], axis=1, errors='ignore').select_dtypes(include=np.number)
    y = train_val_df['Target']

    # --- Bagian Evaluasi ---
    st.info("Membagi data untuk training dan validasi...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    st.info("Menyesuaikan skala dan memilih fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    selector_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector_model.fit(X_train_scaled, y_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    st.info("Melatih model utama...")
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train)
    st.success("✅ Model berhasil dilatih!")

    st.info("Mengevaluasi kinerja model...")
    y_pred_test = model.predict(X_test_selected)
    evaluation_metrics = {
        "r2": r2_score(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "mape": mean_absolute_percentage_error(y_test, y_pred_test)
    }
    
    # --- Bagian Prediksi ---
    st.info("Membuat prediksi untuk hari berikutnya...")
    X_predict = prediction_input_df.drop(['Target', 'Close'], axis=1, errors='ignore').select_dtypes(include=np.number)
    X_predict = X_predict[X.columns] # Pastikan kolom sama

    X_predict_scaled = scaler.transform(X_predict)
    X_predict_selected = selector.transform(X_predict_scaled)
    prediction_for_tomorrow_pct = model.predict(X_predict_selected)[0]
    
    # Membuat prediksi historis untuk seluruh data (bukan hanya test set)
    X_full_scaled = scaler.transform(X)
    X_full_selected = selector.transform(X_full_scaled)
    historical_preds_pct = model.predict(X_full_selected)

    return train_val_df, historical_preds_pct, float(prediction_for_tomorrow_pct), evaluation_metrics, model, X.columns[selector.get_support()]

# === 3. Tampilan Aplikasi (UI) ===

st.title("📈 Prediksi Harga Aset dengan Evaluasi Model")
st.caption("Memadukan data historis dan harga terkini, dilengkapi evaluasi kinerja model.")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    ticker_input = st.text_input("Masukkan Ticker (misal: BBCA.JK, BTC-USD, ETH-USD)", value="BTC-USD")
    show_history = st.checkbox("Tampilkan Prediksi Historis", value=True)
    train_button = st.button("Latih, Evaluasi & Prediksi", type="primary", use_container_width=True)

if train_button:
    st.cache_data.clear()
    st.session_state.ticker = ticker_input
    st.session_state.train_success = False
    
    with st.spinner(f"Memproses data untuk **{ticker_input}**..."):
        try:
            historical_df = load_historical_data(st.session_state.ticker)
            if historical_df is not None:
                current_df = get_current_data(st.session_state.ticker)
                
                if current_df is not None:
                    if not historical_df.empty and historical_df.index[-1].date() == current_df.index[0].date():
                        historical_df = historical_df.iloc[:-1]
                    combined_df = pd.concat([historical_df, current_df])
                else:
                    combined_df = historical_df
                
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                st.info(f"Data berhasil digabungkan. Titik data terakhir: **{combined_df.index[-1].strftime('%Y-%m-%d %H:%M')}**")
                
                featured_df = add_all_features(combined_df)
                
                train_df, hist_preds, tomorrow_pred, metrics, model, selected_features = train_evaluate_and_predict(featured_df)
                
                # Tambahkan pengecekan jika training gagal dan mengembalikan None
                if train_df is not None:
                    st.session_state.results = {
                        "train_df": train_df,
                        "hist_preds": hist_preds,
                        "tomorrow_pred": tomorrow_pred,
                        "metrics": metrics,
                        "model": model,
                        "selected_features": selected_features,
                        "last_close": combined_df['Close'].iloc[-1],
                        "last_date": combined_df.index[-1]
                    }
                    st.session_state.train_success = True
                else:
                    # Jika gagal, pastikan state train_success adalah False
                    st.session_state.train_success = False

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan fatal selama proses:")
            st.error(str(e))
            st.code(traceback.format_exc())
            st.session_state.train_success = False

if 'train_success' in st.session_state and st.session_state.train_success:
    results = st.session_state.results
    ticker = st.session_state.ticker
    
    st.header(f"Hasil Prediksi untuk {ticker}")

    # --- Tampilkan Metric Prediksi Besok ---
    last_close = results['last_close']
    tomorrow_pred_pct = results['tomorrow_pred']
    pred_price_tomorrow = last_close * (1 + tomorrow_pred_pct)
    delta_value = pred_price_tomorrow - last_close
    
    st.metric(
        "Prediksi Harga Besok",
        f"${pred_price_tomorrow:,.4f}",
        f"{delta_value:,.4f} ({tomorrow_pred_pct:.2%}) {'⬆️' if delta_value > 0 else '⬇️'}"
    )
    st.divider()

    # --- Tampilkan Evaluasi Kinerja dan Feature Importance ---
    with st.expander("Lihat Detail Kinerja & Evaluasi Model", expanded=True):
        st.subheader("📊 Kinerja Model pada Data Validasi")
        
        metrics = results['metrics']
        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared (R²)", f"{metrics['r2']:.2f}", help="Semakin mendekati 1, semakin baik model menjelaskan variasi data.")
        col2.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}", help="Rata-rata kesalahan absolut prediksi (dalam satuan target, yaitu % perubahan).")
        col3.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['mape']:.2%}", help="Rata-rata persentase kesalahan prediksi.")

        st.subheader("🧠 Fitur Paling Berpengaruh")
        
        feature_importances = pd.DataFrame({
            'feature': results['selected_features'],
            'importance': results['model'].feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        fig_imp = go.Figure(go.Bar(
            x=feature_importances['importance'],
            y=feature_importances['feature'],
            orientation='h'
        ))
        fig_imp.update_layout(
            title="Top 10 Fitur Terpenting Menurut Model",
            xaxis_title="Tingkat Kepentingan",
            yaxis_title="Fitur",
            yaxis=dict(autorange="reversed"),
            template='plotly_dark'
        )
        st.plotly_chart(fig_imp, use_container_width=True)


    # --- Tampilkan Chart Utama ---
    st.header("Visualisasi Harga dan Prediksi")
    plot_df = results['train_df'].copy()
    plot_df['Harga Prediksi'] = plot_df['Close'] * (1 + results['hist_preds'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name='Harga Aktual (Historis)', line=dict(color='deepskyblue')))
    fig.add_trace(go.Scatter(
        x=[results['last_date']], y=[last_close], name='Harga Terkini',
        mode='markers', marker=dict(color='orange', size=10, symbol='diamond')
    ))
    if show_history:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Harga Prediksi'], name='Prediksi Historis', line=dict(color='tomato', dash='dot', width=1.5)))
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
