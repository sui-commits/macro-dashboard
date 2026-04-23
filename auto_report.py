import os
import smtplib
from email.mime.text import MIMEText
import google.generativeai as genai
import yfinance as yf
from fredapi import Fred
import pandas as pd
from datetime import datetime

# --- 1. 鍵（シークレット）の読み込み ---
FRED_API_KEY = os.environ.get("FRED_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

# --- 2. 50ファクターからのデータ抽出と異常値判定 ---
def get_advanced_market_context():
    # 重要な先行指標・流動性指標のリスト（抜粋）
    TARGET_CODES = {
        'WALCL': 'Fed Total Assets (Liquidity)', 
        'M2SL': 'M2 Money Supply',
        'T10Y2Y': '10Y-2Y Spread', 
        'T10Y3M': '10Y-3M Spread',
        'BAMLH0A0HYM2': 'High Yield Spread (Credit Risk)', 
        'STLFSI4': 'Financial Stress Index',
        'ICSA': 'Initial Jobless Claims (Labor)', 
        'UMCSENT': 'Consumer Sentiment',
        'T5YIFR': '5Y Forward Inflation Expectation'
    }
    
    fred = Fred(api_key=FRED_API_KEY)
    latest_z_scores = {}
    
    # 各指標の直近の異常値（Z-Score）を計算
    for code, name in TARGET_CODES.items():
        try:
            series = fred.get_series(code, observation_start="2019-01-01").dropna()
            if len(series) > 100:
                recent_data = series.tail(750) # 直近3年
                z = (series.iloc[-1] - recent_data.mean()) / recent_data.std()
                latest_z_scores[name] = z
        except:
            continue

    if not latest_z_scores:
        return "データ取得エラー: FRED APIからの応答がありません。"

    # Z-Scoreの並び替え
    df_z = pd.DataFrame(list(latest_z_scores.items()), columns=['Indicator', 'Z-Score']).set_index('Indicator')
    df_z = df_z.sort_values(by='Z-Score', ascending=False)
    
    top3_str = "\n".join([f"  ・{idx}: {row['Z-Score']:+.2f}σ" for idx, row in df_z.head(3).iterrows()])
    bot3_str = "\n".join([f"  ・{idx}: {row['Z-Score']:+.2f}σ" for idx, row in df_z.tail(3).iterrows()])

    # SPYとVIXの現状
    spy = yf.download(["SPY", "^VIX"], period="1mo", progress=False)['Close']
    curr_vix = float(spy['^VIX'].iloc[-1])
    spy_ret = float((spy['SPY'].iloc[-1] / spy['SPY'].iloc[0] - 1) * 100)
    
    # 簡易AI推論ロジック（Kelly基準等）
    hy_risk = latest_z_scores.get('High Yield Spread (Credit Risk)', 0)
    term_spread = latest_z_scores.get('10Y-3M Spread', 0)
    pred_ret = -0.5 if hy_risk > 1.0 or term_spread > 1.0 else 1.5
    kelly = max(0, min(1.0, (pred_ret + 1.0) / 4.0))

    context = f"""
    【AIによる市場環境サマリー】
    ・SPY直近1ヶ月リターン: {spy_ret:+.2f}%
    ・VIX（恐怖指数）: {curr_vix:.2f}
    ・AI予測リターン(1M): {pred_ret:+.2f}%
    ・推奨株式露出度(Kelly): {kelly:.1%} (現金推奨: {1.0 - kelly:.1%})

    【現在、歴史的平均から最も異常な動きをしているマクロ指標】
    🔥 上方乖離 (急騰中):
    {top3_str}
    
    ❄️ 下方乖離 (急落中):
    {bot3_str}
    """
    return context

# --- 3. Geminiによるクロストーク・レポート生成 ---
def generate_debate_report(context):
    genai.configure(api_key=GEMINI_API_KEY)
    
    target_model = None
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            target_model = m.name
            if 'flash' in target_model.lower(): 
                break
                
    if not target_model:
        return "エラー: 利用可能なGeminiモデルが見つかりませんでした。"
        
    model = genai.GenerativeModel(target_model)
    
    prompt = f"""
    あなたはトップクオンツファンドの投資委員会です。以下の最新データに基づき、新規ポジションのアロケーション（株、債券、金、原油、現金）を議論してください。
    
    {context}
    
    以下の3名の専門家のキャラクターになりきり、白熱したクロストーク（対話形式）で議論を展開してください。特に「異常値」として挙げられた指標を根拠に議論を深めてください。
    
    1. 🗣️ マクロ経済アナリスト（ファンダメンタルズと異常値Z-Scoreを重視。保守的。）
    2. 📈 フロートレーダー（VIXや直近の株価リターン、需給を重視。強気になりがち。）
    3. ⚖️ ポートフォリオマネージャー（両者の意見を統合し、最後に具体的な現金比率と投資先を決定し、理由を述べる。）
    
    出力は、そのままメールで読めるように見出しや箇条書きを使って美しく装飾してください。
    """
    return model.generate_content(prompt).text

# --- 4. メール送信処理 ---
def send_email(report_text):
    msg = MIMEText(report_text)
    msg['Subject'] = f"📊 AIクオンツ投資委員会 レポート ({datetime.now().strftime('%Y-%m-%d')})"
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = GMAIL_ADDRESS
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    print("データ抽出開始...")
    context = get_advanced_market_context()
    print("AIエージェントによる議論レポート作成中...")
    report = generate_debate_report(context)
    print("メール送信中...")
    send_email(report)
    print("完了！")
