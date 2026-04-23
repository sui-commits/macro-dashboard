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
from sklearn.model_selection import train_test_split

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
    # PAGE 1: Market Dynamics (資金循環と需給)
    # ==========================================
    if page == "1. Market Dynamics (現在)":
        st.title("📈 Market Dynamics & Capital Rotation")
        
        # 1. 225225風 マーケットボード
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
        
        # 2. セクター・ローテーション（資金の逃避先を解析）
        st.markdown("### 🔄 US Sector Rotation (1-Month Momentum)")
        st.markdown("米国株の11セクターの直近1ヶ月のパフォーマンスから、現在の「リスクオン・オフ」の傾きを解析します。")
        with st.spinner('セクターパフォーマンスを取得中...'):
            try:
                sectors = {
                    'XLK': '情報技術', 'XLF': '金融', 'XLV': 'ヘルスケア', 'XLY': '一般消費財', 
                    'XLC': '通信', 'XLI': '資本財', 'XLP': '生活必需品', 'XLE': 'エネルギー', 
                    'XLU': '公益事業', 'XLRE': '不動産', 'XLB': '素材'
                }
                sector_data = yf.download(list(sectors.keys()), period="1mo")['Close']
                perf = ((sector_data.iloc[-1] / sector_data.iloc[0]) - 1) * 100
                perf_df = pd.DataFrame({'Sector': [sectors[tic] for tic in perf.index], 'Performance (%)': perf.values}).sort_values(by='Performance (%)', ascending=True)
                
                fig_sec = px.bar(perf_df, x='Performance (%)', y='Sector', orientation='h', color='Performance (%)', color_continuous_scale='RdYlGn')
                fig_sec.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_sec, use_container_width=True)
            except Exception as e: st.warning("セクターデータの取得をスキップしました。")

        st.markdown("---")
        
        # 3. CTA & Options
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
    # PAGE 2: Macro Economics (深層と相関)
    # ==========================================
    elif page == "2. Macro Economics (深層)":
        st.title("🏛️ Macro Economic Analysis (PhD Level)")
        
        # 1. クロスアセット相関マトリックス
        st.markdown("### 🔗 Cross-Asset Correlation Matrix (1 Year)")
        st.markdown("株、債券、為替、コモディティの直近1年間の相関関係（連動性）をヒートマップで可視化します。教科書通りの逆相関（株と債券など）が崩れている場合、マクロの潮目が変わっています。")
        with st.spinner('相関マトリックスを計算中...'):
            try:
                assets = {'SPY': 'S&P500(株)', 'TLT': '米国債20年(債券)', 'GLD': 'ゴールド(金)', 'USO': '原油', 'UUP': 'ドル指数', '^VIX': 'VIX(恐怖)'}
                corr_data = yf.download(list(assets.keys()), period="1y")['Close'].rename(columns=assets)
                corr_matrix = corr_data.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1))
                fig_corr.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e: st.warning("相関データの計算をスキップしました。")

        st.markdown("---")
        st.markdown("### 📊 Macro Indicators Time-Series")

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
    # PAGE 4: ML Investment Plan (機械学習予測)
    # ==========================================
    elif page == "4. ML Investment Plan (投資計画)":
        st.title("🤖 ML Price Prediction & Asset Allocation")
        
        # --- セクション1: マクロ・レジーム判定 ---
        st.markdown("### 1. Macro Regime Asset Allocation")
        with st.spinner('マクロ環境を解析中...'):
            try:
                sp500 = yf.Ticker("^GSPC").history(period="1y")['Close']
                growth_momentum = (sp500.iloc[-1] / sp500.iloc[-126] - 1) * 100 
                cpi = fred.get_series('CPIAUCSL').dropna()
                inflation_momentum = (cpi.iloc[-1] / cpi.iloc[-7] - 1) * 100

                if growth_momentum > 0 and inflation_momentum > 0:
                    regime, desc = "Overheating (オーバーヒート)", "成長とインフレが共存。金融引き締めに警戒。"
                    portfolio, color_scheme = {"株式 (VTI)": 40, "コモディティ (DBC)": 30, "金 (GLD)": 15, "短期債・現金": 15}, ['#ff4b4b', '#ffa500', '#ffd700', '#808080']
                elif growth_momentum < 0 and inflation_momentum > 0:
                    regime, desc = "Stagflation (スタグフレーション)", "成長鈍化とインフレが併存。現金と実物資産へ避難。"
                    portfolio, color_scheme = {"金 (GLD)": 30, "コモディティ (DBC)": 20, "短期債・現金": 40, "株式 (VTI)": 10}, ['#ffd700', '#ffa500', '#808080', '#ff4b4b']
                elif growth_momentum > 0 and inflation_momentum < 0:
                    regime, desc = "Goldilocks (ゴルディロックス)", "インフレが落ち着き成長が続く適温相場。株へ全力投球。"
                    portfolio, color_scheme = {"株式 (VTI/QQQ)": 60, "長期国債 (TLT)": 30, "社債 (LQD)": 10, "現金": 0}, ['#3fb950', '#58a6ff', '#8a2be2', '#000000']
                else:
                    regime, desc = "Deflation / Recession (デフレ・景気後退)", "インフレ鎮静と景気後退。中央銀行の利下げに備え長期債を選好。"
                    portfolio, color_scheme = {"長期国債 (TLT)": 50, "短期債・現金": 30, "ディフェンシブ株": 10, "金 (GLD)": 10}, ['#58a6ff', '#808080', '#3fb950', '#ffd700']

                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.info(f"**判定:** {regime}\n\n{desc}")
                    fig_pie = go.Figure(data=[go.Pie(labels=list(portfolio.keys()), values=list(portfolio.values()), hole=.4, marker=dict(colors=color_scheme))])
                    fig_pie.update_layout(template="plotly_dark", height=280, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e: st.error(f"レジーム判定エラー: {e}")

        st.markdown("---")

        # --- セクション2: ランダムフォレストによる予測モデル ---
        st.markdown("### 2. Random Forest Price Prediction (1-Month Forward)")
        st.markdown("ランダムフォレスト（アンサンブル学習）を利用し、マクロ変数間の**非線形な関係性**から 21営業日（約1ヶ月）後 のS&P 500の予測値と、現在市場を支配している要因（Feature Importance）を導き出します。")
        
        with st.spinner('ランダムフォレスト・モデルを学習・評価中...'):
            try:
                # データの取得 (10年分の学習データで精度を高める)
                spy = yf.Ticker("SPY").history(period="10y")['Close'].rename("SPY")
                vix = yf.Ticker("^VIX").history(period="10y")['Close'].rename("VIX")
                dxy = yf.Ticker("DX-Y.NYB").history(period="10y")['Close'].rename("Dollar_Index")
                t10y2y = fred.get_series("T10Y2Y").rename("Yield_Curve")
                hy = fred.get_series("BAMLH0A0HYM2").rename("HY_Spread")
                breakeven = fred.get_series("T10YIE").rename("Inflation_Expectation")

                # データの結合と前処理
                df_ml = pd.concat([spy, vix, dxy, t10y2y, hy, breakeven], axis=1).fillna(method='ffill').dropna()
                
                # 目的変数の作成 (21日先のSPY価格)
                df_ml['Target_SPY'] = df_ml['SPY'].shift(-21)
                df_train = df_ml.dropna()
                
                features = ['SPY', 'VIX', 'Dollar_Index', 'Yield_Curve', 'HY_Spread', 'Inflation_Expectation']
                X = df_train[features]
                y = df_train['Target_SPY']
                
                # 直近のデータセットを分離（予測用）
                latest_data = df_ml[features].iloc[-1:]
                
                # ランダムフォレストの構築と学習 (決定木100本)
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rf_model.fit(X, y)
                
                # 予測の実行
                current_price = latest_data['SPY'].values[0]
                predicted_price = rf_model.predict(latest_data)[0]
                predicted_return = ((predicted_price / current_price) - 1) * 100

                # 結果の表示
                st.write("#### 🤖 AI予測結果 (S&P 500 ETF: SPY)")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(label="現在価格 (Current)", value=f"${current_price:.2f}")
                col_m2.metric(label="1ヶ月後 AI予測価格 (Predicted)", value=f"${predicted_price:.2f}", delta=f"{predicted_return:.2f}%")
                col_m3.metric(label="モデル決定係数 (R² Score)", value=f"{rf_model.score(X, y):.3f}")

                # 特徴量重要度の可視化（AIが何を重視して予測したか）
                st.write("#### 🧠 AIの判断根拠 (Feature Importance)")
                st.markdown("モデルが今後の株価を予測する上で、**現在どのマクロ変数を最も重視しているか**（影響度の割合）を示します。")
                importances = rf_model.feature_importances_ * 100
                imp_df = pd.DataFrame({'Feature': features, 'Importance (%)': importances}).sort_values(by='Importance (%)', ascending=True)
                
                fig_imp = px.bar(imp_df, x='Importance (%)', y='Feature', orientation='h', color='Importance (%)', color_continuous_scale='Blues')
                fig_imp.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.caption("※ 免責事項: 本機械学習モデルは過去10年間のマクロデータに基づく統計的予測であり、将来の価格やリターンを保証するものではありません。")

            except Exception as e:
                st.error(f"機械学習モデルの計算エラー: 詳細を確認してください ({e})")

except Exception as e:
    st.error(f"System Error: {e}")
