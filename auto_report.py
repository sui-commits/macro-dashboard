import os
import smtplib
from email.mime.text import MIMEText
import google.generativeai as genai
import yfinance as yf
from fredapi import Fred
import pandas as pd
from datetime import datetime

# 鍵（シークレット）の読み込み
FRED_API_KEY = os.environ.get("FRED_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

def get_market_context():
    spy = yf.download(["SPY", "^VIX"], period="1mo", progress=False)['Close']
    curr_vix = spy['^VIX'].iloc[-1]
    spy_ret = (spy['SPY'].iloc[-1] / spy['SPY'].iloc[0] - 1) * 100
    
    fred = Fred(api_key=FRED_API_KEY)
    t10y3m = fred.get_series('T10Y3M').dropna().iloc[-1]
    
    regime = "🟢 Normal (Risk-On)" if curr_vix < 20 else "🔴 Crisis (Risk-Off)"
    
    return f"レジーム: {regime}\nVIX: {curr_vix:.2f}\nSPY1ヶ月リターン: {spy_ret:+.2f}%\nT10Y3M: {t10y3m:.2f}%"

def generate_debate_report(context):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    あなたはトップクオンツファンドの投資委員会です。以下のデータに基づきアロケーション（株、債券、金、現金）を議論してください。
    データ: {context}
    
    1. 🗣️マクロアナリスト
    2. 📈フロートレーダー
    3. ⚖️PM（結論と現金比率）
    の3名でクロストークを展開してください。
    """
    return model.generate_content(prompt).text

def send_email(report_text):
    msg = MIMEText(report_text)
    msg['Subject'] = f"📊 AIクオンツ投資委員会 レポート ({datetime.now().strftime('%Y-%m-%d')})"
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = GMAIL_ADDRESS
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    context = get_market_context()
    report = generate_debate_report(context)
    send_email(report)
