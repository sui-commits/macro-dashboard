from sklearn.mixture import GaussianMixture
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
.metric-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center;}
.alert-box-red { background-color: #211111; border-left: 5px solid #f85149; padding: 15px; border-radius: 5px;}
.alert-box-green { background-color: #112111; border-left: 5px solid #3fb950; padding: 15px; border-radius: 5px;}
.insight-box { background-color: #161b22; border-left: 5px solid #58a6ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. 認証 & データロード準備 ---
API_KEY = st.secrets["FRED_API_KEY"]
SHEET_URL = st.secrets["SHEET_URL"]
fred = Fred(api_key=API_KEY)

@st.cache_data(ttl=600)
def load_settings():
    try:
        df = pd.read_csv(SHEET_URL)
        for col in df.columns: df[col] = df[col].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"スプレッドシートの読み込みに失敗しました: {e}")
        return pd.DataFrame()

# --- 3. 共通データ処理関数 ---
def normalize_data(series_list):
    for s in series_list:
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return series_list

# ==========================================
# メイン処理開始
# ==========================================
try:
    settings_df = load_settings()
    st.sidebar.title("💎 Macro Navigation")
    page = st.sidebar.radio("機能を選択", [
        "1. Market Dynamics (現在)", 
        "2. Asset Class Macro (アセット別分析)", 
        "3. Historical Analysis (過去比較)",
        "4. Investment Strategy (AI戦略)"
    ])

    # ==========================================
    # PAGE 1: Market Dynamics (現在の需給・勢い)
    # ==========================================
    if page == "1. Market Dynamics (現在)":
        st.title("📈 Market Dynamics & Flows")
        
        # --- VIX 期間構造 ---
        st.markdown("### 🚨 Volatility Term Structure (恐怖の構造)")
        vix_data = yf.download(["^VIX", "^VIX3M"], period="5d", progress=False)['Close']
        v_now, v3m_now = vix_data['^VIX'].iloc[-1], vix_data['^VIX3M'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-box'><h4>VIX (短期)</h4><h2>{v_now:.2f}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><h4>VIX3M (中期)</h4><h2>{v3m_now:.2f}</h2></div>", unsafe_allow_html=True)
        with c3:
            if v_now > v3m_now:
                st.markdown("<div class='alert-box-red'><h4>⚠️ バックワーデーション</h4>短期パニック状態です。</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-box-green'><h4>✅ コンタンゴ (正常)</h4>相場は安定しています。</div>", unsafe_allow_html=True)

        st.markdown("---")
        # --- セクターヒートマップ ---
        st.markdown("### 🔄 Sector Rotation (1-Month Heatmap)")
        sectors = {'XLK':'Tech', 'XLF':'Finance', 'XLV':'Health', 'XLY':'ConsDis', 'XLC':'Comm', 'XLI':'Indust', 'XLP':'Staples', 'XLE':'Energy', 'XLU':'Utility', 'XLRE':'REIT', 'XLB':'Basic'}
        sec_df = yf.download(list(sectors.keys()), period="1mo", progress=False)['Close']
        perf_df = pd.DataFrame([{"Sector": sectors[t], "Return": ((sec_df[t].iloc[-1]/sec_df[t].iloc[0])-1)*100, "Size":1} for t in sectors.keys()])
        fig_hm = px.treemap(perf_df, path=['Sector'], values='Size', color='Return', color_continuous_scale='RdYlGn', color_continuous_midpoint=0, template="plotly_dark")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("---")
        # --- CTA & Options ---
        c_cta, c_opt = st.columns(2)
        with c_cta:
            st.markdown("#### 🤖 S&P500 CTA Trend (200SMA)")
            spy_y = yf.Ticker("SPY").history(period="1y")
            curr, sma = spy_y['Close'].iloc[-1], spy_y['Close'].rolling(200).mean().iloc[-1]
            st.info(f"200日線乖離率: {((curr-sma)/sma)*100:+.2f}%")
            st.plotly_chart(px.line(spy_y, y='Close', template="plotly_dark", height=250), use_container_width=True)
        with c_opt:
            st.markdown("#### 🎯 Options Max Pain Magnet")
            try:
                spy_o = yf.Ticker("SPY")
                exp = spy_o.options[0]
                calls, puts = spy_o.option_chain(exp).calls, spy_o.option_chain(exp).puts
                strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
                mp, min_l = 0, float('inf')
                for s in strikes:
                    l = calls[calls['strike']<s].apply(lambda x:(s-x['strike'])*x['openInterest'], axis=1).sum() + puts[puts['strike']>s].apply(lambda x:(x['strike']-s)*x['openInterest'], axis=1).sum()
                    if l < min_l: min_l, mp = l, s
                fig_g = go.Figure(go.Indicator(mode="gauge+number+delta", value=curr, delta={'reference':mp}, title={'text':f"Max Pain: ${mp:.0f}"}, gauge={'axis':{'range':[mp*0.9, mp*1.1]}, 'threshold':{'line':{'color':'yellow','width':4},'value':mp}}))
                st.plotly_chart(fig_g.update_layout(template="plotly_dark", height=250), use_container_width=True)
            except: st.write("オプションデータ取得不可")

    # ==========================================
    # PAGE 2: Asset Class Macro (アセット別分析)
    # ==========================================
    elif page == "2. Asset Class Macro (アセット別分析)":
        st.title("🏦 Asset Class Macro Analysis")
        st.markdown("各アセット（株・債券・金・原油・仮想通貨）を動かす根源的なマクロ指標との相関と乖離を分析します。")

        # スイング相関マトリックス
        st.markdown("### 🔗 Swing Correlation Matrix (1W/1H)")
        try:
            assets = {'SPY':'Stock', 'TLT':'Bond', 'GLD':'Gold', 'USO':'Oil', 'UUP':'Dollar', 'BTC-USD':'Crypto'}
            c_data = yf.download(list(assets.keys()), period="1wk", interval="1h", progress=False)['Close'].rename(columns=assets).corr()
            st.plotly_chart(px.imshow(c_data, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark", height=400), use_container_width=True)
        except Exception as e:
            st.warning("相関マトリックスのデータ取得に失敗しました。")

        # ★ 修正ポイント1: 制限を解除し、スプレッドシートに登録されたすべてのタブを表示
        tabs_names = settings_df['タブ名'].unique()
        tabs = st.tabs(tabs_names)
        
        for i, t_name in enumerate(tabs_names):
            with tabs[i]:
                t_df = settings_df[settings_df['タブ名'] == t_name]
                cols = st.columns(2)
                for g_idx, g_name in enumerate(t_df['グラフ名'].unique()):
                    with cols[g_idx % 2]:
                        g_data = t_df[t_df['グラフ名'] == g_name]
                        fig = None
                        
                        # ★ 修正ポイント2: CTA専用のグラフ描画ロジックを追加
                        if len(g_data) == 1 and g_data.iloc[0]['ソース'] == 'CTA':
                            r = g_data.iloc[0]
                            try:
                                d = yf.Ticker(r['ティッカー']).history(period="2y")
                                d.index = pd.to_datetime(d.index).tz_localize(None)
                                d['SMA200'] = d['Close'].rolling(200).mean()
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=d.index, y=d['Close'], name='Price', line=dict(color='#c9d1d9')))
                                fig.add_trace(go.Scatter(x=d.index, y=d['SMA200'], name='200 SMA', line=dict(color='#ff4b4b', dash='dot')))
                                max_d = d.index.max()
                                fig.update_xaxes(range=[max_d - pd.DateOffset(months=12), max_d])
                                fig.update_layout(title=f"🤖 {g_name} (CTA Trend)", height=300, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                            except: pass

                        # ★ 修正ポイント3: Options専用のグラフ描画ロジックを追加
                        elif len(g_data) == 1 and g_data.iloc[0]['ソース'] == 'Options':
                            r = g_data.iloc[0]
                            try:
                                s = yf.Ticker(r['ティッカー'])
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
                                fig.add_vline(x=mp, line_dash="dash", line_color="yellow")
                                fig.add_vline(x=curr, line_dash="solid", line_color="#3fb950")
                                fig.update_layout(title=f"🎯 {g_name} (Max Pain: {mp:.0f})", height=300, template="plotly_dark", barmode='group', margin=dict(l=0,r=0,t=30,b=0))
                            except: pass

                        # 通常の折れ線グラフ (FRED, Yahoo, DBnomics)
                        else:
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            max_dt = None
                            for _, r in g_data.iterrows():
                                try:
                                    d = None
                                    if r['ソース'] == 'FRED': d = fred.get_series(r['ティッカー']).loc['2022-01-01':]
                                    elif r['ソース'] == 'Yahoo': d = yf.Ticker(r['ティッカー']).history(start='2022-01-01')['Close']
                                    elif r['ソース'] == 'DBnomics': 
                                        db_df = fetch_series(r['ティッカー'])
                                        if not db_df.empty: d = db_df[['period', 'value']].dropna().set_index('period')['value']
                                    
                                    if d is not None and not d.empty:
                                        d.index = pd.to_datetime(d.index).tz_localize(None)
                                        if max_dt is None or d.index.max() > max_dt: max_dt = d.index.max()
                                        fig.add_trace(go.Scatter(x=d.index, y=d.values, name=r['データ名']), secondary_y=(r['軸']=='副軸'))
                                except: pass
                            if max_dt: fig.update_xaxes(range=[max_dt - pd.DateOffset(years=2), max_dt])
                            fig.update_layout(title=g_name, height=300, template="plotly_dark", hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0))
                        
                        # グラフが生成された場合のみ描画
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PAGE 3: Historical Analysis (過去比較)
    # ==========================================
    elif page == "3. Historical Analysis (過去比較)":
        st.title("🕰️ Historical Regime Alignment")
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("分析指数", ["^GSPC", "^DJI", "^NDX", "GC=F"])
        regimes = {"2022 Inflation": "2022-01-03", "2008 Lehman": "2008-09-15", "2000 IT Bubble": "2000-03-24"}
        past_ev = c2.selectbox("過去レジーム", list(regimes.keys()))
        curr_ev = pd.to_datetime(c3.date_input("起点日", pd.to_datetime("2024-01-01")))
        
        if st.button("比較実行"):
            d = yf.Ticker(target).history(start="1970-01-01")['Close']
            d.index = pd.to_datetime(d.index).tz_localize(None)
            p_idx = d.index.get_indexer([pd.to_datetime(regimes[past_ev])], method='nearest')[0]
            c_idx = d.index.get_indexer([curr_ev], method='nearest')[0]
            p_s = (d.iloc[p_idx-20 : p_idx+200] / d.iloc[p_idx]) * 100
            c_s = (d.iloc[c_idx-20 : c_idx+200] / d.iloc[c_idx]) * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(-20, len(p_s)-20), y=p_s.values, name=past_ev, line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=np.arange(-20, len(c_s)-20), y=c_s.values, name="Current", line=dict(width=3, color='#ff4b4b')))
            st.plotly_chart(fig.update_layout(template="plotly_dark", height=500), use_container_width=True)
  # ==========================================
    # PAGE 4: Institutional Quant Engine (Pro UI/UX)
    # ==========================================
    elif page == "4. Investment Strategy (AI戦略)":
        st.markdown("""
        <style>
        .kpi-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .kpi-title { color: #8b949e; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .kpi-value { color: #c9d1d9; font-size: 28px; font-weight: 700; margin-bottom: 0px; }
        .kpi-sub { font-size: 13px; margin-top: 5px; }
        </style>
        """, unsafe_allow_html=True)

        st.title("🧠 Quantitative AI Strategy & Allocation")
        st.markdown("マクロ先行指標、流動性、需給データに基づく1ヶ月先（21営業日）の予測と、ケリー基準による最適資金配分。")

        # --- サイドバー設定 ---
        with st.sidebar.expander("⚙️ Model Architecture Settings", expanded=True):
            indicator_mode = st.radio("Macro Mode", ["Leading (先行指標特化)", "Full Macro (遅行指標含む)"])
            exclude_other_stocks = st.checkbox("Cross-Asset Exclusion (株価指数の除外)", value=True)
            include_anomaly = st.checkbox("Presidential Cycle (アノマリー追加)", value=False)
            lookback_years = st.slider("Lookback Window (学習期間)", 3, 10, 5)

        with st.spinner('Compiling Macro Data & Executing Ensemble Models...'):
            try:
                target_ticker = "SPY"
                
                # --- 1. データ取得パイプライン ---
                yahoo_tickers = settings_df[settings_df['ソース'] == 'Yahoo']['ティッカー'].unique().tolist()
                market_implied_tickers = ['HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                all_yahoo = list(set(yahoo_tickers + market_implied_tickers + [target_ticker]))

                base_fred = ['ANFCI', 'STLFSI4', 'T10Y3M', 'BAMLH0A0HYM2', 'WALCL', 'M2SL', 'ICSA', 'AWHAEMAN', 'T5YIFR', 'PERMIT', 'UMCSENT']
                if "Full Macro" in indicator_mode:
                    base_fred.extend(['CPIAUCSL', 'UNRATE', 'PAYEMS', 'INDPRO'])
                all_fred = list(set(settings_df[settings_df['ソース'] == 'FRED']['ティッカー'].unique().tolist() + base_fred))

                series_list = []
                y_data = yf.download(all_yahoo, period=f"{lookback_years}y", progress=False)['Close']
                if isinstance(y_data, pd.Series): y_data = y_data.to_frame(all_yahoo[0])
                for col in y_data.columns: series_list.append(y_data[col].dropna().rename(col))
                
                for tic in all_fred:
                    try: series_list.append(fred.get_series(tic).loc[f"{2024-lookback_years}-01-01":].dropna().rename(tic))
                    except: pass

                for i in range(len(series_list)):
                    series_list[i].index = pd.to_datetime(series_list[i].index).tz_localize(None).normalize()
                df_ml = pd.concat(series_list, axis=1).ffill()
                
                # --- 2. 特徴量エンジニアリング ---
                df_ml['CTA_200D_Bias'] = df_ml[target_ticker] / df_ml[target_ticker].rolling(200).mean() - 1
                df_ml['Vol_Term_Spread'] = df_ml['^VIX'] / df_ml['^VIX3M']
                df_ml['Copper_Gold_Ratio'] = df_ml['HG=F'] / df_ml['GC=F']
                df_ml['Presidential_Cycle'] = df_ml.index.year % 4
                df_ml['Target'] = df_ml[target_ticker].pct_change(21).shift(-21)
                df_train = df_ml.dropna()

                drop_cols = ['Target', 'HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                if exclude_other_stocks:
                    drop_cols.extend([t for t in yahoo_tickers if t != target_ticker])
                if not include_anomaly:
                    drop_cols.append('Presidential_Cycle')

                features = [col for col in df_train.columns if col not in drop_cols]
                X, y = df_train[features], df_train['Target']
                
                # --- 3. アンサンブル学習と不確実性(Confidence)の計算 ---
                weights = np.exp(np.linspace(-1, 0, len(X)))
                model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
                model.fit(X, y, sample_weight=weights)
                
                latest_x = df_ml[features].iloc[-1:]
                
                # 全ての決定木(300本)の予測値を集計し、分布を計算
                all_tree_preds = np.array([tree.predict(latest_x.values) for tree in model.estimators_])
                pred_ret = np.mean(all_tree_preds) * 100
                pred_std = np.std(all_tree_preds) * 100 # 予測のばらつき（不確実性）
                
                # 95%信頼区間
                ci_lower = pred_ret - (1.96 * pred_std)
                ci_upper = pred_ret + (1.96 * pred_std)

                # --- 4. Z-Score 解析 ---
                df_3y = df_ml.tail(252*3)
                macro_only = [f for f in features if f != 'Presidential_Cycle']
                z_scores = (latest_x[macro_only].iloc[0] - df_3y[macro_only].mean()) / df_3y[macro_only].std()
                
                imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                imp_df['Z_Score'] = imp_df['Feature'].map(z_scores).fillna(0)
                imp_df = imp_df.sort_values('Importance', ascending=False).head(10)

                # --- 5. ケリー基準・リスク指標計算 ---
                td = df_ml.tail(120).copy()
                # 過去の予測推移と信頼区間
                preds_history = np.array([tree.predict(td[features].values) for tree in model.estimators_])
                td['AI_Fair'] = (np.mean(preds_history, axis=0) + 1) * td[target_ticker]
                td['Fair_Upper'] = (np.mean(preds_history, axis=0) + 1.96 * np.std(preds_history, axis=0) + 1) * td[target_ticker]
                td['Fair_Lower'] = (np.mean(preds_history, axis=0) - 1.96 * np.std(preds_history, axis=0) + 1) * td[target_ticker]
                
                gap = ((df_ml[target_ticker].iloc[-1] - td['AI_Fair'].iloc[-1]) / td['AI_Fair'].iloc[-1]) * 100
                
                # ケリー基準ベースの最適ウェイト = E[R] / Variance
                # ※実運用向けにスケーリング
                vix_dec = df_ml['^VIX'].iloc[-1] / 100
                market_var = vix_dec ** 2
                kelly_f = (pred_ret / 100) / market_var if market_var > 0 else 0
                optimal_weight = max(0, min(100, kelly_f * 100 * 0.5)) # ハーフ・ケリー（安全重視）

                # ==========================================
                # UI Rendering (Dashboard)
                # ==========================================
                
                # --- KPI Cards ---
                c1, c2, c3, c4 = st.columns(4)
                
                color_ret = "#3fb950" if pred_ret > 0 else "#f85149"
                c1.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>1M Expected Return</div>
                    <div class='kpi-value' style='color:{color_ret}'>{pred_ret:+.2f}%</div>
                    <div class='kpi-sub'>95% CI: [{ci_lower:+.1f}%, {ci_upper:+.1f}%]</div>
                </div>""", unsafe_allow_html=True)
                
                color_gap = "#f85149" if gap > 2 else "#3fb950" if gap < -2 else "#58a6ff"
                c2.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Fair Value Spread</div>
                    <div class='kpi-value' style='color:{color_gap}'>{gap:+.2f}%</div>
                    <div class='kpi-sub'>Market vs AI Pricing</div>
                </div>""", unsafe_allow_html=True)
                
                c3.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Optimal Exposure (Half-Kelly)</div>
                    <div class='kpi-value' style='color:#e3b341'>{optimal_weight:.0f}%</div>
                    <div class='kpi-sub'>Cash Recommendation: {100-optimal_weight:.0f}%</div>
                </div>""", unsafe_allow_html=True)
                
                conf_score = max(0, 100 - (pred_std * 10))
                c4.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Model Confidence</div>
                    <div class='kpi-value' style='color:#a371f7'>{conf_score:.1f}/100</div>
                    <div class='kpi-sub'>Based on Ensemble Variance</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- Charts Row ---
                col_chart1, col_chart2 = st.columns([1.2, 1])
                
                with col_chart1:
                    st.markdown("#### 📉 Market Price vs AI Fair Value Band")
                    fig_gap = go.Figure()
                    # 信頼区間リボン
                    fig_gap.add_trace(go.Scatter(x=td.index.tolist() + td.index[::-1].tolist(),
                                                 y=td['Fair_Upper'].tolist() + td['Fair_Lower'][::-1].tolist(),
                                                 fill='toself', fillcolor='rgba(88, 166, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Band'))
                    fig_gap.add_trace(go.Scatter(x=td.index, y=td['AI_Fair'], name="AI Fair Value", line=dict(dash='dot', color='#58a6ff', width=2)))
                    fig_gap.add_trace(go.Scatter(x=td.index, y=td[target_ticker], name="SPY Price", line=dict(color='#ff4b4b', width=2)))
                    
                    fig_gap.update_layout(template="plotly_dark", height=380, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig_gap, use_container_width=True)

                with col_chart2:
                    st.markdown("#### 🔍 Feature Attribution (Impact x Anomaly)")
                    fig_brain = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h',
                                       color='Z_Score', color_continuous_scale='RdBu_r', range_color=[-3, 3], template="plotly_dark")
                    fig_brain.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), coloraxis_colorbar=dict(title="Z-Score"))
                    st.plotly_chart(fig_brain, use_container_width=True)

                # ==========================================
                # 🤖 LLM Prompt Generator
                # ==========================================
                with st.expander("📝 Generate LLM Quantitative Report Prompt", expanded=False):
                    st.markdown("以下のテキストをコピーし、Gemini等のLLMに送信してアナリストレポートを生成します。")

                    top_features_text = ""
                    for _, row in imp_df.head(8).iterrows():
                        top_features_text += f"- {row['Feature']}: 重要度 {row['Importance']:.4f}, 現在のZ-Score {row['Z_Score']:+.2f}\n"

                    anfci_val = df_ml['ANFCI'].iloc[-1] if 'ANFCI' in df_ml.columns else 0.0

                    llm_prompt = f"""あなたはトップティアのヘッジファンドのシニア・マクロ・ストラテジストです。
以下の自社開発AI（Random Forest Ensemble）の推論結果を元に、PM（ポートフォリオマネージャー）向けの投資判断レポートを作成してください。

【AI予測メトリクス (対象: 米国株, ホライズン: 1ヶ月)】
・予測リターン: {pred_ret:+.2f}% (95%信頼区間: {ci_lower:+.2f}% 〜 {ci_upper:+.2f}%)
・モデル確信度スコア: {conf_score:.1f}/100
・AI適正価格との乖離: {gap:+.2f}% (プラス=市場が割高)
・VIX: {df_ml['^VIX'].iloc[-1]:.2f} / 金融ストレス(ANFCI): {anfci_val:.2f}
・最適ポジション露出度 (ハーフ・ケリー基準): {optimal_weight:.0f}%

【AI決定要因: 上位特徴量とZ-Score異常値】
{top_features_text}

【出力要件】
1. マクロ環境の総括: なぜAIはそのリターンを予測したのか、異常値（Z-Score）を示している指標を絡めて因果関係を解説。
2. 信頼区間とリスク: 予測のばらつき（信頼区間幅）や、市場価格と理論値の乖離（Spread）から見えるダウンサイド・リスク。
3. クロスアセット・ランキング: 上記環境下において、[株, 債券, 現金, 原油, 金] の中で、今後1ヶ月でアウトパフォームする順に1〜5位のランキングと、クオンツ的根拠を提示。
"""
                    st.code(llm_prompt, language="text")

            except Exception as e: st.error(f"分析モデル・エラー: {e}")

    # ==========================================
    # PAGE 5: Headline Reverse-Engineering
    # ==========================================
    elif page == "5. Headline Reverse-Engineering (イベント逆引き)":
        st.title("🗞️ Volatility Anomaly & Reverse-Engineering")
        lookback_months = st.slider("分析期間 (ヶ月)", 1, 12, 6)
        
        with st.spinner('市場の異常変動日を検出中...'):
            try:
                spy_data = yf.Ticker("SPY").history(period=f"{lookback_months}mo")['Close']
                spy_ret = spy_data.pct_change().dropna() * 100
                z_scores = (spy_ret - spy_ret.mean()) / spy_ret.std()
                top_moves = spy_ret.abs().nlargest(5).index
                
                st.subheader("📊 統計的異常変動の検出 (Top 5 Volatility Spikes)")
                fig_events = go.Figure()
                fig_events.add_trace(go.Scatter(x=spy_data.index, y=spy_data.values, name="SPY Price", line=dict(color='#8b949e', width=2)))
                
                for date in top_moves:
                    ret_val = spy_ret.loc[date]
                    color = '#3fb950' if ret_val > 0 else '#f85149'
                    fig_events.add_trace(go.Scatter(
                        x=[date], y=[spy_data.loc[date]], mode='markers+text',
                        marker=dict(color=color, size=12, symbol='star'),
                        text=[f"{ret_val:+.1f}%"], textposition="top center", name=f"{date.strftime('%Y-%m-%d')}"
                    ))
                st.plotly_chart(fig_events.update_layout(template="plotly_dark", height=400, showlegend=False), use_container_width=True)

                st.markdown("---")
                with st.expander("🤖 ニュース逆引きプロンプトの生成", expanded=True):
                    events_text = "".join([f"- {d.strftime('%Y-%m-%d')}: {spy_ret.loc[d]:+.2f}%\n" for d in top_moves])
                    llm_event_prompt = f"""あなたはマクロ経済の歴史に精通したアナリストです。
過去{lookback_months}ヶ月間でS&P500が統計的な異常変動（3シグマ級）を記録した以下の5つの日付について、そのトリガーとなったニュースや経済指標発表をウェブ検索等で特定し、解説してください。

【異常変動日リスト】
{events_text}
"""
                    st.code(llm_event_prompt, language="text")

            except Exception as e: st.error(f"イベント特定エラー: {e}")

# ===== システム全体を囲む try の except =====
except Exception as e:
    st.error(f"System Critical Error: {e}")
