import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# === Load data ===
def load_data(ticker='BTC-USD', start='2018-01-01'):
    df = yf.download(ticker, start=start)
    df.dropna(inplace=True)
    return df

# === Tambahkan semua fitur teknikal dan statistik ===
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
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
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
    df['Target'] = df['Close'].pct_change().shift(-1)
    return df

# === Training pipeline + evaluasi + penyimpanan prediksi ===
def train_and_save_model():
    df = load_data()
    df = add_all_features(df)
    df.dropna(inplace=True)

    # Fitur & target
    X = df.drop(['Target', 'Close'], axis=1)
    y = df['Target']

    # Simpan nama fitur awal
    features = X.columns.tolist()
    joblib.dump(features, 'features.joblib')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    selector_model = xgb.XGBRegressor(n_estimators=200, random_state=42)
    selector_model.fit(X_train_scaled, y_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    # Model akhir
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=10, random_state=42)
    model.fit(X_train_selected, y_train)

    # Evaluasi
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

    print("=== Evaluasi Model ===")
    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.6f}")
    print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.6f}")
    print(f"Test R²: {r2_score(y_test, y_test_pred):.4f}")

    # Simpan hasil test prediksi
    joblib.dump(pd.Series(y_test.values, index=y_test.index), 'y_test.joblib')
    joblib.dump(pd.Series(y_test_pred, index=y_test.index), 'y_pred.joblib')

    # Simpan artefak model
    joblib.dump(model, 'xgb_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(selector, 'selector.joblib')

    print("✅ Model dan artefak disimpan.")

    # ============ Tambahan: Prediksi seluruh data historis (untuk visualisasi) ============
    try:
        X_all = df.drop(['Target', 'Close'], axis=1)
        X_scaled_all = scaler.transform(X_all)
        X_selected_all = selector.transform(X_scaled_all)
        y_pred_all = model.predict(X_selected_all)
        y_test_all = df['Close'].pct_change().shift(-1)

        close_prices = df['Close']
        y_pred_all_prices = close_prices * (1 + y_pred_all)

        y_pred_all_series = pd.Series(y_pred_all_prices, index=df.index)[:-1]
        y_test_all_series = pd.Series(close_prices.shift(-1), index=df.index)[:-1]

        joblib.dump(y_pred_all_series, "y_pred_all.joblib")
        joblib.dump(y_test_all_series, "y_test_all.joblib")

        print("📈 Prediksi historis disimpan untuk visualisasi.")
    except Exception as e:
        print("❌ Gagal menyimpan prediksi historis.")
        print(str(e))

# === Jalankan jika script dijalankan langsung ===
if __name__ == "__main__":
    train_and_save_model()
