import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

# --- Load FinBERT once (cached for speed) ---
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("text-classification", model="ProsusAI/finbert")

analyzer = load_sentiment_analyzer()

# --- Prediction Logic ---
def get_price_prediction(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period="2y")
    history['5-Day MA'] = history['Close'].rolling(window=5).mean()
    history['10-Day MA'] = history['Close'].rolling(window=10).mean()
    history['Target'] = (history['Close'].shift(-1) > history['Close']).astype(int)
    history = history.dropna()
    
    features = ['Close', '5-Day MA', '10-Day MA']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(history[features][:-1], history['Target'][:-1])
    
    prediction = model.predict(history[features].tail(1))[0]
    return "UP" if prediction == 1 else "DOWN"

def get_news_sentiment(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    news = stock.news
    headlines = [a.get('title') for a in news if a.get('title')]
    if not headlines: return "NEUTRAL"
    
    scores = [1 if analyzer(h)[0]['label'] == 'positive' else -1 if analyzer(h)[0]['label'] == 'negative' else 0 for h in headlines]
    total = sum(scores)
    return "BULLISH" if total > 0 else "BEARISH" if total < 0 else "NEUTRAL"

# --- Streamlit UI ---
st.title("📈 AI Stock Observer")
ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL):", "NVDA")

if st.button("Analyze"):
    with st.spinner('Analyzing market data...'):
        price_signal = get_price_prediction(ticker)
        news_signal = get_news_sentiment(ticker)
        
        st.subheader(f"Results for {ticker}")
        col1, col2 = st.columns(2)
        col1.metric("Price Trend", price_signal)
        col2.metric("News Sentiment", news_signal)
        
        # Simple Decision Logic
        st.write("---")
        if price_signal == "UP" and news_signal == "BULLISH":
            st.success("Verdict: STRONG BUY")
        elif price_signal == "DOWN" and news_signal == "BEARISH":
            st.error("Verdict: STRONG SELL")
        else:
            st.warning("Verdict: NEUTRAL / CAUTION")