import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Stock Prediction App", page_icon="📈", layout="wide")
st.title("Stock Prediction App 📈")

stocks = ("AAPL", "META", "GOOG", "MSFT", "BTC-USD", "ETH-USD")
selected_stock = st.selectbox("Wybierz akcje do analizy", stocks)

n_days = st.slider("Ilość dni do prognozy (krótkoterminowa)", 1, 30, 5)



@st.cache_data(ttl=3600) 
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)


    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Ładowanie danych...")
data = load_data(selected_stock)


if data.empty:
    st.error(
        f"⚠️ API Yahoo Finance zwróciło puste dane dla {selected_stock}. Wyczyść cache (klawisz 'C') i spróbuj ponownie za chwilę.")
    st.stop()

data_load_state.text("Dane załadowane!")

st.subheader("Dane historyczne")
st.dataframe(data.tail())


def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = np.where(loss == 0, np.inf, gain / loss)
    rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
    return pd.Series(rsi, index=data.index)


data['SMA'] = calculate_sma(data, window=20)
data['RSI'] = calculate_rsi(data, window=14)

data['Return'] = data['Close'].pct_change()
data['SMA_Ratio'] = data['Close'] / data['SMA']

st.subheader("Dane z wskaźnikami technicznymi")
st.dataframe(data.tail())


fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Cena Zamknięcia i SMA", "Wskaźnik RSI (14)"))

fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], name="SMA 20", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI", line=dict(color="green")), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1)

fig.update_layout(height=600, title_text="Analiza Techniczna", showlegend=True)
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Prognoza na {n_days} dni (XGBoost ML)")

df_ml = data[['Date', 'Close', 'Return', 'SMA_Ratio', 'RSI']].copy()


df_ml['Lag_1_Ret'] = df_ml['Return'].shift(1)
df_ml['Lag_2_Ret'] = df_ml['Return'].shift(2)
df_ml['Lag_3_Ret'] = df_ml['Return'].shift(3)
df_ml['SMA_Ratio_Lag1'] = df_ml['SMA_Ratio'].shift(1)
df_ml['RSI_Lag1'] = df_ml['RSI'].shift(1)

df_ml.dropna(inplace=True)


X = df_ml[['Lag_1_Ret', 'Lag_2_Ret', 'Lag_3_Ret', 'SMA_Ratio_Lag1', 'RSI_Lag1']]
y = df_ml['Return']


model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X, y)


y_pred_history = model.predict(X)
df_ml['Predicted_Return'] = y_pred_history
df_ml['Predicted_Close'] = df_ml['Close'].shift(1) * (1 + df_ml['Predicted_Return'])

df_ml_eval = df_ml.dropna(subset=['Predicted_Close'])
mae_usd = mean_absolute_error(df_ml_eval['Close'], df_ml_eval['Predicted_Close'])

st.info(
    f"💡 **Ocena modelu na danych historycznych:** Średni błąd bezwzględny (MAE) wynosi **${mae_usd:.2f}** na jednej sesji.")

# --- ITERACYJNE PRZEWIDYWANIE PRZYSZŁOŚCI ---
ostatnia_data = df_ml['Date'].iloc[-1]
przyszle_daty = pd.date_range(start=ostatnia_data + pd.Timedelta(days=1), periods=n_days, freq='B')

historia_cen = df_ml['Close'].tail(30).tolist()
historia_zwrotow = df_ml['Return'].tail(10).tolist()
prognozy_cen = []

for _ in range(n_days):
    lag_1_ret = historia_zwrotow[-1]
    lag_2_ret = historia_zwrotow[-2]
    lag_3_ret = historia_zwrotow[-3]

    sma_obecne = sum(historia_cen[-20:]) / 20
    sma_ratio_obecne = historia_cen[-1] / sma_obecne

    zmiany = np.diff(historia_cen[-15:])
    zyski = np.where(zmiany > 0, zmiany, 0)
    straty = np.where(zmiany < 0, -zmiany, 0)
    avg_gain = np.mean(zyski)
    avg_loss = np.mean(straty)
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi_obecne = 100 if avg_loss == 0 else 100 - (100 / (1 + rs))


    x_pred = np.array([[lag_1_ret, lag_2_ret, lag_3_ret, sma_ratio_obecne, rsi_obecne]])
    pred_ret = model.predict(x_pred)[0]

    ostatnia_cena = historia_cen[-1]
    nowa_cena = ostatnia_cena * (1 + pred_ret)

    prognozy_cen.append(nowa_cena)
    historia_cen.append(nowa_cena)
    historia_zwrotow.append(pred_ret)


df_prognoza = pd.DataFrame({
    'Data': przyszle_daty.date,
    'Prognozowana Cena ($)': prognozy_cen
})
df_prognoza['Prognozowana Cena ($)'] = df_prognoza['Prognozowana Cena ($)'].round(2)

st.dataframe(df_prognoza, use_container_width=True)

fig_xgb = go.Figure()
fig_xgb.add_trace(go.Scatter(x=df_ml['Date'].tail(60), y=df_ml['Close'].tail(60), name="Historia (ostatnie 60 dni)",
                             line=dict(color="blue")))
fig_xgb.add_trace(go.Scatter(x=df_prognoza['Data'], y=df_prognoza['Prognozowana Cena ($)'], name="Prognoza XGBoost",
                             line=dict(color="red", dash="dash")))

fig_xgb.update_layout(title_text=f"Wykres Prognozy ({n_days} dni)", hovermode="x unified")
st.plotly_chart(fig_xgb, use_container_width=True)
