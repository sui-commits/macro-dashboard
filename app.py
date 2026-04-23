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
warnings.filterwarnings('ignore')

# --- ページ設定 ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Consolas', 'Courier New', monospace !important; }
div[data-testid="metric-container"] {
    background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s;
}
div[data-testid="metric-container"]:hover { transform: translateY(-2px); border-color: #58a6ff; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { background-color: #21262d; border-radius: 4px; padding: 8px 16px; }
.stTabs [aria-selected="true"] { border-bottom: 2px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)

API_KEY = st.secrets["FRED_API_KEY"]
SHEET_URL = st.secrets["SHEET_URL"]
fred = Fred(api_key=API_KEY)

@st.cache_data(ttl=3600)
def load_settings():
    df = pd.read_csv(SHEET_URL)
    for col in df.columns: df[col] = df[col].astype(str).str.strip()
    return df

try:
    settings_df = load_settings()
    st.sidebar.title("💎 Macro Navigation")
    page = st.sidebar.radio("機能を選択", [
        "1. Market Dynamics (現在)", 
        "2. Macro Economics (深層)", 
        "3. Historical Analysis (過去比較)",
        "4. ML Investment Plan (投資計画)"
    ])

    # ==========================================
    # PAGE 1: Market Dynamics
    # ==========================================
    if page == "1. Market Dynamics (現在)":
        st.title("📈 Market Dynamics & Capital Rotation")
        st.markdown("### ⚡ Real-time Market Board")
        board_cols = st.columns(5)
        count = 0
        for _, row in settings_df.iterrows():
            if row['ソース'] in ['FRED', 'Yahoo'] and count < 25:
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
        st.markdown("### 🔄 US Sector Rotation (1-Month Momentum)")
        with st.spinner('セクターパフォーマンスを取得中...'):
            try:
                sectors = {'XLK': '情報技術', 'XLF': '金融', 'XLV': 'ヘルスケア', 'XLY': '一般消費財', 'XLC': '通信', 'XLI': '資本財', 'XLP': '生活必需品', 'XLE': 'エネルギー', 'XLU': '公益事業', 'XLRE': '不動産', 'XLB': '素材'}
                sector_data = yf.download(list(sectors.keys()), period="1mo", progress=False)['Close']
                perf = ((sector_data.iloc[-1] / sector_data.iloc[0]) - 1) * 100
                perf_df = pd.DataFrame({'Sector': [sectors[tic] for tic in perf.index], 'Performance (%)': perf.values}).sort_values(by='Performance (%)', ascending=True)
                fig_sec = px.bar(perf_df, x='Performance (%)', y='Sector', orientation='h', color='Performance (%)', color_continuous_scale='RdYlGn')
                fig_sec.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_sec, use_container_width=True)
            except: st.warning("セクターデータの取得をスキップしました。")

        st.markdown("---")
        st.markdown("### 🛡️ Positioning & Options")
        c1, c2 = st.columns(2)
        cta_opts = settings_df[settings_df['ソース'].isin(['CTA', 'Options'])]
        for idx, row in cta_opts.iterrows():
            with (c1 if idx % 2 == 0 else c2):
                if row['ソース'] == 'CTA':
                    try:
                        d = yf.Ticker(row['ティッカー']).history(period="2y")
                        d.index = pd.to_datetime(d.index).tz_localize(None)
                        d['SMA20'], d['SMA200'] = d['Close'].rolling(20).mean(), d['Close'].rolling(200).mean()
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=d.index, y=d['Close'], name='Price', line=dict(color='#c9d1d9')))
                        fig.add_trace(go.Scatter(x=d.index, y=np.where(d['SMA20']>d['SMA200'], d['Close'].max(), d['Close'].min()), fill='tozeroy', fillcolor='rgba(63,185,80,0.1)', line=dict(width=0)))
                        max_d = d.index.max()
                        fig.update_xaxes(range=[max_d - pd.DateOffset(months=6), max_d + pd.DateOffset(days=5)])
                        fig.update_layout(title=row['データ名'], height=300, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    except: pass
                elif row['ソース'] == 'Options':
                    try:
                        s = yf.Ticker(row['ティッカー'])
                        exp = s.options[0]
                        c, p = s.option_chain(exp).calls, s.option_chain(exp).puts
                        strikes = sorted(list(set(c['strike']).union(set(p['strike']))))
                        mp, min_l = 0, float('inf')
                        for strk in strikes:
                            l = c[c['strike']<strk].apply(lambda x:(strk-x['strike'])*x['openInterest'], axis=1).sum() + p[p['strike']>strk].apply(lambda x:(x['strike']-strk)*x['openInterest'], axis=1).sum()
                            if l < min_l: min_l, mp = l, strk
                        curr = s.history(period='1d')['Close'].iloc[-1]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=c['strike'], y=c['openInterest'], name='Call', marker_color='#58a6ff'))
                        fig.add_trace(go.Bar(x=p['strike'], y=p['openInterest'], name='Put', marker_color='#f85149'))
                        fig.add_vline(x=mp, line_dash="dash", line_color="yellow", annotation_text=f"MP:{mp}", annotation_position="top left")
                        fig.add_vline(x=curr, line_dash="solid", line_color="#3fb950", annotation_text=f"Price:{curr:.1f}", annotation_position="bottom right")
                        fig.update_layout(title=f"{row['データ名']} ({exp})", height=300, template="plotly_dark", barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                    except: pass

    # ==========================================
    # PAGE 2: Macro Economics
    # ==========================================
    elif page == "2. Macro Economics (深層)":
        st.title("🏛️ Macro Economic Analysis (PhD Level)")
        
        # ▼ スイングトレード特化：直近1週間の1時間足相関マトリックス
        st.markdown("### 🔗 Cross-Asset Correlation Matrix (1 Week / 1H Interval)")
        st.markdown("スイングトレード向けに解像度を上げ、**直近1週間の「1時間足」**を用いた精密な相関関係を可視化します。短期的な資金逃避や連動性の崩れを察知します。")
        with st.spinner('スイング用・高解像度相関データを計算中...'):
            try:
                assets = {'SPY': 'S&P500', 'TLT': '米国債20年', 'GLD': 'ゴールド', 'USO': '原油', 'UUP': 'ドル指数', '^VIX': 'VIX'}
                corr_data = yf.download(list(assets.keys()), period="1wk", interval="1h", progress=False)['Close'].rename(columns=assets)
                corr_matrix = corr_data.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1))
                fig_corr.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e: st.warning(f"相関データの計算に失敗しました（時間外等の理由）: {e}")

        st.markdown("---")
        tabs_names = [t for t in settings_df['タブ名'].unique() if t not in ['オプション動向', 'CTAトレンド']]
        tabs = st.tabs(tabs_names)
        for i, t_name in enumerate(tabs_names):
            with tabs[i]:
                t_df = settings_df[settings_df['タブ名'] == t_name]
                g_names = t_df['グラフ名'].unique()
                cols = st.columns(2)
                for g_idx, g_name in enumerate(g_names):
                    with cols[g_idx % 2]:
                        g_df = t_df[t_df['グラフ名'] == g_name]
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        max_dt = None
                        for _, r in g_df.iterrows():
                            try:
                                d = None
                                if r['ソース'] == 'FRED': d = fred.get_series(r['ティッカー']).loc['2020-01-01':]
                                elif r['ソース'] == 'Yahoo': d = yf.Ticker(r['ティッカー']).history(start='2020-01-01')['Close']
                                elif r['ソース'] == 'DBnomics':
                                    db_df = fetch_series(r['ティッカー'])
                                    if not db_df.empty: d = db_df[['period', 'value']].dropna().set_index('period')['value']
                                if d is not None and not d.empty:
                                    d.index = pd.to_datetime(d.index).tz_localize(None)
                                    if max_dt is None or d.index.max() > max_dt: max_dt = d.index.max()
                                    fig.add_trace(go.Scatter(x=d.index, y=d.values, name=r['データ名']), secondary_y=(r['軸']=='副軸'))
                            except: pass
                        if max_dt: fig.update_xaxes(range=[max_dt - pd.DateOffset(months=6), max_dt + pd.DateOffset(days=5)])
                        fig.update_layout(title=g_name, height=300, template="plotly_dark", hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PAGE 3: Historical Analysis
    # ==========================================
    elif page == "3. Historical Analysis (過去比較)":
        st.title("🕰️ Historical Regime & Forward Return")
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("分析指数", ["^GSPC (S&P 500)", "^DJI (Dow Jones)", "^NDX (Nasdaq 100)"])
        ticker = target.split(" ")[0]
        regimes = {"2022 Inflation Shock": "2022-01-03", "2008 Lehman Shock": "2008-09-15", "2000 Dot-com Bubble": "2000-03-24"}
        past_ev = c2.selectbox("過去のレジーム(T=0)", list(regimes.keys()))
        curr_ev = pd.to_datetime(c3.date_input("現在の比較起点(T=0)", pd.to_datetime("2024-01-01"))).tz_localize(None)
        days_to_track = st.slider("比較する営業日数", 50, 500, 250)

        if st.button("結論を算出"):
            try:
                d = yf.Ticker(ticker).history(start="1970-01-01")['Close']
                d.index = pd.to_datetime(d.index).tz_localize(None)
                p_idx = d.index.get_indexer([pd.to_datetime(regimes[past_ev])], method='nearest')[0]
                c_idx = d.index.get_indexer([curr_ev], method='nearest')[0]
                offset = int(days_to_track / 4)
                p_slice = (d.iloc[max(0, p_idx - offset) : p_idx + days_to_track] / d.iloc[p_idx]) * 100
                c_slice = (d.iloc[max(0, c_idx - offset) : c_idx + days_to_track] / d.iloc[c_idx]) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.arange(-len(p_slice[:p_idx - max(0, p_idx - offset)]), len(p_slice) - len(p_slice[:p_idx - max(0, p_idx - offset)])), y=p_slice.values, name=f"Past: {past_ev}", line=dict(dash='dash', color='gray')))
                fig.add_trace(go.Scatter(x=np.arange(-len(c_slice[:c_idx - max(0, c_idx - offset)]), len(c_slice) - len(c_slice[:c_idx - max(0, c_idx - offset)])), y=c_slice.values, name=f"Current", line=dict(width=3, color='#ff4b4b')))
                fig.update_layout(title=f"{ticker} Regime Alignment", height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(e)

# ==========================================
    # PAGE 4: Advanced Investment Strategy & AI Analysis
    # ==========================================
    elif page == "4. Investment Strategy":
        st.title("🧭 Professional Investment Strategy & AI Predictor")
        st.markdown("マクロ、需給、機械学習を統合し、具体的な『投資アクション』を導き出します。")

        # --- データ取得と特徴量エンジニアリング ---
        with st.spinner('クオンツモデルを構築中...'):
            try:
                # 10年分の多角的なデータを取得
                spy = yf.Ticker("SPY").history(period="10y")['Close'].rename("SPY")
                vix = yf.Ticker("^VIX").history(period="10y")['Close'].rename("VIX")
                dxy = yf.Ticker("DX-Y.NYB").history(period="10y")['Close'].rename("USD")
                t10y2y = fred.get_series("T10Y2Y").rename("Yield_Curve")
                hy = fred.get_series("BAMLH0A0HYM2").rename("HY_Spread")
                
                # 同期処理
                df_pro = pd.concat([spy, vix, dxy, t10y2y, hy], axis=1)
                df_pro.index = pd.to_datetime(df_pro.index).tz_localize(None).normalize()
                df_pro = df_pro.ffill().dropna()

                # ★ 高度な特徴量生成 (Advanced Feature Engineering)
                # 1. モメンタム（勢い）
                df_pro['Mom_1m'] = df_pro['SPY'].pct_change(21)
                df_pro['RSI'] = (df_pro['SPY'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / 
                                 df_pro['SPY'].diff().abs().rolling(14).mean()) * 100
                
                # 2. ボラティリティ（リスク環境）
                df_pro['Vol_21d'] = df_pro['SPY'].pct_change().rolling(21).std() * np.sqrt(252)
                
                # 3. 相関の変化（マクロの連動性）
                df_pro['Stock_Bond_Corr'] = df_pro['SPY'].rolling(63).corr(df_pro['Yield_Curve'])
                
                # ターゲット: 21営業日後のリターン
                df_pro['Target_Return'] = df_pro['SPY'].pct_change(21).shift(-21)
                
                # 学習準備
                features = ['VIX', 'USD', 'Yield_Curve', 'HY_Spread', 'Mom_1m', 'RSI', 'Vol_21d', 'Stock_Bond_Corr']
                df_train = df_pro.dropna()
                X = df_train[features]
                y = df_train['Target_Return']
                
                # 最新トレンド重視の重み付け学習
                weights = np.exp(np.linspace(-1, 0, len(X))) # 直近ほど指数関数的に重く
                model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                model.fit(X, y, sample_weight=weights)
                
                # 未来予測
                latest_x = df_pro[features].iloc[-1:]
                pred_return = model.predict(latest_x)[0] * 100
                
            except Exception as e:
                st.error(f"データ処理エラー: {e}")

        # --- 統合ダッシュボードレイアウト ---
        col_main, col_sub = st.columns([2, 1])

        with col_main:
            st.markdown("### 🏹 総合投資判断 (Executive Summary)")
            
            # 判断ロジック
            vix_now = df_pro['VIX'].iloc[-1]
            hy_now = df_pro['HY_Spread'].iloc[-1]
            rsi_now = df_pro['RSI'].iloc[-1]
            
            # スコアリング算出
            ai_score = np.clip(pred_return * 2, -5, 5) # AIリターンを5点満点でスコア化
            macro_score = (1 if hy_now < 4 else -1) + (1 if vix_now < 20 else -1)
            total_score = ai_score + macro_score
            
            if total_score > 3:
                status, color, icon = "積極的投資 (Aggressive)", "#00ff00", "🚀"
            elif total_score > 0:
                status, color, icon = "部分的投資 (Cautious Long)", "#58a6ff", "⚖️"
            else:
                status, color, icon = "防御的待機 (Defensive/Cash)", "#f85149", "🛡️"

            st.markdown(f"""
            <div style="background-color:#161b22; padding:20px; border-radius:10px; border-left:10px solid {color};">
                <h2 style="color:{color};">{icon} {status}</h2>
                <p style="font-size:18px;"><b>AI予測 1ヶ月リターン:</b> {pred_return:+.2f}%</p>
                <p><b>現在のマクロ環境:</b> {'健全' if hy_now < 4 else 'ストレス増大'} (HY Spread: {hy_now:.2f}%)</p>
                <p><b>テクニカル過熱度:</b> {'過熱感あり' if rsi_now > 70 else '調整済み' if rsi_now < 30 else '中立'} (RSI: {rsi_now:.1f})</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🧠 AIの判断根拠 (Model Insights)")
            imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance')
            fig_imp = px.bar(imp, x='importance', y='feature', orientation='h', template="plotly_dark", 
                             color='importance', color_continuous_scale='Blues')
            fig_imp.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_sub:
            st.markdown("### 📊 推奨ポジション比率")
            # リスク調整後の比率算出 (VIXが高いほどキャッシュを増やす)
            risk_budget = max(0, min(100, (1 / vix_now) * 1500)) # 簡易的なボラティリティ・ターゲティング
            if pred_return < 0: risk_budget *= 0.5 # 予測がマイナスなら比率を半分に
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_budget,
                title = {'text': "推奨リスク露出度 (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [{'range': [0, 50], 'color': "#333"}, {'range': [50, 100], 'color': "#444"}]
                }
            ))
            fig_gauge.update_layout(height=250, template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("#### 📝 具体的な戦略案")
            if status == "積極的投資 (Aggressive)":
                st.write("・SPYへのロングポジションを構築
except Exception as e:
    st.error(f"System Error: {e}")
