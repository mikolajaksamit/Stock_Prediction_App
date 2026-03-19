import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ustawienia początkowe
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App 📈")

stocks = ("AAPL", "META", "GOOG", "MSFT", "BTC-USD", "ETH-USD")
selected_stock = st.selectbox("Wybierz akcje do analizy", stocks)

n_years = st.slider("Ilość lat do prognozy", 1, 5)
period = n_years * 365

# Pobieranie danych
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    
    # Zabezpieczenie przed nowymi wersjami yfinance (MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Ładowanie danych...")
data = load_data(selected_stock)
data_load_state.text("Dane załadowane!")

st.subheader("Dane historyczne")
st.write(data.tail())

# Obliczanie wskaźników technicznych
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Zabezpieczenie przed dzieleniem przez zero
    rs = np.where(loss == 0, np.inf, gain / loss)
    rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
    return pd.Series(rsi, index=data.index)

data['SMA'] = calculate_sma(data, window=20)
data['RSI'] = calculate_rsi(data, window=14)

st.subheader("Dane z wskaźnikami technicznymi")
st.write(data.tail())

# Profesjonalna wizualizacja na dwóch osiach Y (Subplots)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.1, 
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Cena Zamknięcia i SMA", "Wskaźnik RSI (14)"))

# Górny wykres: Cena i SMA
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], name="SMA 20", line=dict(color="orange")), row=1, col=1)

# Dolny wykres: RSI
fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI", line=dict(color="green")), row=2, col=1)

# Linie referencyjne dla RSI (przewartościowanie / niedowartościowanie)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1)

fig.update_layout(height=600, title_text="Analiza Techniczna", showlegend=True)
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
st.plotly_chart(fig)

# Przygotowanie danych do Prophet
df_train = data[["Date", "Close"]].copy()
df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
df_train.dropna(inplace=True)

# KRYTYCZNA POPRAWKA: Usunięcie strefy czasowej, której Prophet nie obsługuje
df_train["ds"] = pd.to_datetime(df_train["ds"]).dt.tz_localize(None)
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")
df_train.dropna(subset=['y'], inplace=True)

# Trenowanie modelu
m = Prophet()
m.fit(df_train)

# Przewidywanie przyszłych wartości
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Prognoza")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Wizualizacja prognozy
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Składowe prognozy")
fig2 = m.plot_components(forecast)
# KRYTYCZNA POPRAWKA: Prawidłowe renderowanie obiektu matplotlib
st.pyplot(fig2)
