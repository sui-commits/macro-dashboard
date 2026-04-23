import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from dbnomics import fetch_series
from sklearn.ensemble import RandomForestRegressor
import warnings

# エラー抑制
warnings.filterwarnings('ignore')

# --- 1. ページ基本設定 & プロ仕様CSS ---
st.set_page_config(page_title="Macro Quant Terminal Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Consolas', monospace; }
div[data-testid="metric-container"] {
    background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;
}
.insight-box { background-color: #161b22; border-left: 5px solid #58a6ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
.strategy-card { background-color: #1c2128; border: 1px solid #444c56; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. 認証 & データロード準備 ---
API_KEY = st.secrets["FRED_API_KEY"]
SHEET_URL = st.secrets["SHEET_URL"]
fred = Fred(api_key=API_KEY)

@st.cache_data(ttl=600)
def load_settings():
    df = pd.read_csv(SHEET_URL)
    for col in df.columns: df[col] = df[col].astype(str).str.strip()
    return df

# --- 3. サイドバーナビゲーション ---
try:
    settings_df = load_settings()
    st.sidebar.title("💎 Macro Navigation")
    page = st.sidebar.radio("機能を選択", [
        "1. Market Dynamics (現在)", 
        "2. Macro Economics (深層)", 
        "3. Historical Analysis (過去比較)",
        "4. Investment Strategy (AI分析)"
    ])

    # ==========================================
    # PAGE 1: Market Dynamics (現在の需給・勢い)
    # ==========================================
    if page == "1. Market Dynamics (現在)":
        st.title("📈 Market Dynamics & Capital Rotation")
        
        # 225225風ボード
        st.markdown("### ⚡ Real-time Macro Board")
        board_cols = st.columns(5)
        count = 0
        for _, row in settings_df.iterrows():
            if row['ソース'] in ['FRED', 'Yahoo'] and count < 15:
                try:
                    if row['ソース'] == 'FRED':
                        d = fred.get_series(row['ティッカー']).dropna()
                        board_cols[count % 5].metric(label=row['データ名'], value=f"{d.iloc[-1]:.2f}", delta=f"{d.iloc[-1]-d.iloc[-2]:.2f}")
                    elif row['ソース'] == 'Yahoo':
                        d = yf.Ticker(row['ティッカー']).history(period="5d")['Close'].dropna()
                        pct = ((d.iloc[-1]-d.iloc[-2])/d.iloc[-2])*100
                        board_cols[count % 5].metric(label=row['データ名'], value=f"{d.iloc[-1]:.2f}", delta=f"{pct:.2f}%")
                    count += 1
                except: pass

        st.markdown("---")
        # セクターローテーション
        st.markdown("### 🔄 Sector Rotation (1-Month Momentum)")
        sectors = {'XLK':'Tech', 'XLF':'Finance', 'XLV':'Health', 'XLY':'ConsDis', 'XLC':'Comm', 'XLI':'Indust', 'XLP':'Staples', 'XLE':'Energy', 'XLU':'Utility', 'XLRE':'REIT', 'XLB':'Basic'}
        sector_data = yf.download(list(sectors.keys()), period="1mo", progress=False)['Close']
        perf = ((sector_data.iloc[-1] / sector_data.iloc[0]) - 1) * 100
        perf_df = pd.DataFrame({'Sector': [sectors[t] for t in perf.index], 'Perf': perf.values}).sort_values('Perf')
        st.plotly_chart(px.bar(perf_df, x='Perf', y='Sector', orientation='h', color='Perf', color_continuous_scale='RdYlGn', template="plotly_dark", height=400), use_container_width=True)

        # CTA / Options インサイト
        st.markdown("### 🛡️ Positioning & Flows")
        c1, c2 = st.columns(2)
        with c1:
            spy_d = yf.Ticker("SPY").history(period="2y")
            sma200 = spy_d['Close'].rolling(200).mean().iloc[-1]
            dist = ((spy_d['Close'].iloc[-1] - sma200) / sma200) * 100
            st.info(f"**CTA Trend:** 200日線乖離率 {dist:.1f}% ({'強気維持' if dist > 0 else '弱気圏'})")
        with c2:
            try:
                spy_obj = yf.Ticker("SPY")
                exp = spy_obj.options[0]
                calls = spy_obj.option_chain(exp).calls
                puts = spy_obj.option_chain(exp).puts
                strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
                mp, min_l = 0, float('inf')
                for s in strikes:
                    l = calls[calls['strike']<s].apply(lambda x:(s-x['strike'])*x['openInterest'], axis=1).sum() + puts[puts['strike']>s].apply(lambda x:(x['strike']-s)*x['openInterest'], axis=1).sum()
                    if l < min_l: min_l, mp = l, s
                st.success(f"**Option Wall:** Max Pain ${mp:.0f} (現在のターゲット価格)")
            except: st.write("Options data unavailable")

    # ==========================================
    # PAGE 2: Macro Economics (マクロ指標と相関)
    # ==========================================
    elif page == "2. Macro Economics (深層)":
        st.title("🏛️ Macro Economic Analysis")
        st.markdown("### 🔗 Swing Correlation Matrix (1Week / 1Hour)")
        try:
            assets = {'SPY':'Stock', 'TLT':'Bond', 'GLD':'Gold', 'USO':'Oil', 'UUP':'Dollar', '^VIX':'VIX'}
            corr_data = yf.download(list(assets.keys()), period="1wk", interval="1h", progress=False)['Close'].rename(columns=assets).corr()
            st.plotly_chart(px.imshow(corr_data, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark", height=450), use_container_width=True)
        except: pass

        st.markdown("---")
        tabs = st.tabs(settings_df['タブ名'].unique().tolist())
        for i, t_name in enumerate(settings_df['タブ名'].unique()):
            with tabs[i]:
                t_df = settings_df[settings_df['タブ名'] == t_name]
                cols = st.columns(2)
                for g_idx, g_name in enumerate(t_df['グラフ名'].unique()):
                    with cols[g_idx % 2]:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        g_data = t_df[t_df['グラフ名'] == g_name]
                        max_dt = None
                        for _, r in g_data.iterrows():
                            try:
                                if r['ソース'] == 'FRED': d = fred.get_series(r['ティッカー']).loc['2022-01-01':]
                                elif r['ソース'] == 'Yahoo': d = yf.Ticker(r['ティッカー']).history(start='2022-01-01')['Close']
                                elif r['ソース'] == 'DBnomics': d = fetch_series(r['ティッカー']).set_index('period')['value']
                                if d is not None:
                                    d.index = pd.to_datetime(d.index).tz_localize(None)
                                    if max_dt is None or d.index.max() > max_dt: max_dt = d.index.max()
                                    fig.add_trace(go.Scatter(x=d.index, y=d.values, name=r['データ名']), secondary_y=(r['軸']=='副軸'))
                            except: pass
                        if max_dt: fig.update_xaxes(range=[max_dt - pd.DateOffset(months=6), max_dt])
                        fig.update_layout(title=g_name, height=300, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PAGE 3: Historical Analysis (レジーム比較)
    # ==========================================
    elif page == "3. Historical Analysis (過去比較)":
        st.title("🕰️ Historical Regime Alignment")
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("分析指数", ["^GSPC", "^DJI", "^NDX"])
        regimes = {"2022 Inflation": "2022-01-03", "2008 Lehman": "2008-09-15", "2000 IT Bubble": "2000-03-24"}
        past_ev = c2.selectbox("過去のレジーム", list(regimes.keys()))
        curr_ev = pd.to_datetime(c3.date_input("現在の起点", pd.to_datetime("2024-01-01")))
        
        if st.button("軌跡を比較"):
            d = yf.Ticker(target).history(start="1970-01-01")['Close']
            d.index = pd.to_datetime(d.index).tz_localize(None)
            p_idx = d.index.get_indexer([pd.to_datetime(regimes[past_ev])], method='nearest')[0]
            c_idx = d.index.get_indexer([curr_ev], method='nearest')[0]
            p_s = (d.iloc[p_idx-20 : p_idx+200] / d.iloc[p_idx]) * 100
            c_s = (d.iloc[c_idx-20 : c_idx+200] / d.iloc[c_idx]) * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(-20, len(p_s)-20), y=p_s.values, name=past_ev, line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=np.arange(-20, len(c_s)-20), y=c_s.values, name="Current", line=dict(width=3, color='#ff4b4b')))
            fig.update_layout(title="Regime Comparison (T=0: 100)", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PAGE 4: Investment Strategy (AI & 乖離分析)
    # ==========================================
    elif page == "4. Investment Strategy (AI分析)":
        st.title("🧭 AI Strategy & Market Gap Analysis")
        
        with st.spinner('AIモデルを構築中...'):
            try:
                # データの取得と正規化
                spy = yf.Ticker("SPY").history(period="10y")['Close'].rename("SPY")
                vix = yf.Ticker("^VIX").history(period="10y")['Close'].rename("VIX")
                dxy = yf.Ticker("DX-Y.NYB").history(period="10y")['Close'].rename("USD")
                t10y2y = fred.get_series("T10Y2Y").rename("Yield_Curve")
                hy = fred.get_series("BAMLH0A0HYM2").rename("HY_Spread")
                
                series = [spy, vix, dxy, t10y2y, hy]
                for s in series: s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                df_ml = pd.concat(series, axis=1).ffill().dropna()

                # 特徴量エンジニアリング
                df_ml['Mom'] = df_ml['SPY'].pct_change(21)
                df_ml['Vol'] = df_ml['SPY'].pct_change().rolling(21).std() * np.sqrt(252)
                df_ml['Target'] = df_ml['SPY'].pct_change(21).shift(-21)
                
                features = ['VIX', 'USD', 'Yield_Curve', 'HY_Spread', 'Mom', 'Vol']
                df_train = df_ml.dropna()
                X, y = df_train[features], df_train['Target']
                
                # 重み付け学習 (直近重視)
                weights = np.exp(np.linspace(-1, 0, len(X)))
                model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y, sample_weight=weights)
                
                latest_x = df_ml[features].iloc[-1:]
                pred_ret = model.predict(latest_x)[0] * 100
                
                # --- 乖離分析セクション ---
                st.subheader("1. Market vs AI Gap Analysis")
                col_g1, col_g2 = st.columns([2, 1])
                with col_g1:
                    test_df = df_ml.tail(60).copy()
                    test_df['AI_Fair'] = (model.predict(test_df[features]) + 1) * test_df['SPY']
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Scatter(x=test_df.index, y=test_df['SPY'], name="Market", line=dict(color='#ff4b4b', width=2)))
                    fig_g.add_trace(go.Scatter(x=test_df.index, y=test_df['AI_Fair'], name="AI Theoretical", line=dict(dash='dot', color='#58a6ff')))
                    fig_g.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig_g, use_container_width=True)
                with col_g2:
                    gap = ((df_ml['SPY'].iloc[-1] - test_df['AI_Fair'].iloc[-1]) / test_df['AI_Fair'].iloc[-1]) * 100
                    st.metric("市場の乖離率 (vs AI理論値)", f"{gap:+.2f}%")
                    if gap > 2: st.warning("⚠️ 市場はマクロ理論値を超えて過熱中")
                    elif gap < -2: st.success("✅ 市場はマクロ的に割安圏内")
                    else: st.info("⚖️ 適正水準を維持")

                # --- 統合戦略判断 ---
                st.markdown("---")
                st.subheader("2. Executive Strategy Decision")
                c_main, c_sub = st.columns([2, 1])
                
                with c_main:
                    score = np.clip(pred_ret * 2, -5, 5) + (1 if hy.iloc[-1] < 4 else -1)
                    if score > 2: status, color, icon = "積極的投資", "#00ff00", "🚀"
                    elif score > -1: status, color, icon = "部分的投資", "#58a6ff", "⚖️"
                    else: status, color, icon = "防御的待機", "#f85149", "🛡️"
                    
                    st.markdown(f"""
                    <div style="border-left:10px solid {color}; background:#161b22; padding:20px; border-radius:10px;">
                        <h2 style="color:{color};">{icon} {status}</h2>
                        <p>AI予測 1ヶ月リターン: <b>{pred_ret:+.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c_sub:
                    risk = max(0, min(100, (1/vix.iloc[-1])*1500))
                    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=risk, title={'text':"推奨露出度"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':color}})).update_layout(height=250, template="plotly_dark", margin=dict(t=50,b=0)), use_container_width=True)

                st.markdown("#### 📝 具体的な戦略案")
                if score > 2: st.write("・SPYロング継続。モメンタム追随。\n・押し目があれば積極的に追加建玉。")
                elif score > -1: st.write("・比率を半分に抑え、ヘッジを検討。\n・セクター選別を強化（ディフェンシブ混合）。")
                else: st.write("・現金化を優先。ボラティリティ低下を待つ。\n・プット購入による下落プロテクトを推奨。")

            except Exception as e: st.error(f"分析エラー: {e}")

except Exception as e:
    st.error(f"System Critical Error: {e}")
