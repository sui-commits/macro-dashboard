
import streamlit as st
import pandas as pd
import yfinance as yf
from fredapi import Fred

# 認証設定
API_KEY = st.secrets["FRED_API_KEY"]
SHEET_URL = st.secrets["SHEET_URL"]
fred = Fred(api_key=API_KEY)

# スプレッドシートの読み込み（キャッシュ化）
@st.cache_data(ttl=600)
def load_settings():
    try:
        df = pd.read_csv(SHEET_URL)
        for col in df.columns: df[col] = df[col].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"スプレッドシートの読み込みに失敗しました: {e}")
        return pd.DataFrame()

# Yahoo Financeデータの取得（キャッシュ化で高速化）
@st.cache_data(ttl=3600)
def fetch_market_data(tickers, period="5y"):
    try:
        data = yf.download(tickers, period=period, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        return data
    except Exception as e:
        st.toast(f"⚠️ Yahoo Financeからのデータ取得エラー: {e}")
        return pd.DataFrame()
