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
    # PAGE 4: Institutional Quant Engine & XAI
    # ==========================================
    elif page == "4. Investment Strategy (AI戦略)":
        st.title("🧠 Institutional Quant Strategy Engine")
        st.markdown("グローバル・マクロファンド水準の先行指標群を統合し、AIの思考プロセスをZ-Scoreで可視化します。")

        # --- アノマリー切り替えトグル ---
        anomaly_mode = st.radio(
            "🇺🇸 モデルの選択 (大統領選サイクル・アノマリーの反映)", 
            ["除外する (Standard Macro Model)", "反映する (Anomaly-Adjusted Model)"], 
            horizontal=True
        )

        with st.spinner('マクロデータ網の構築、アノマリーのパース、および推論を実行中...'):
            try:
                target_ticker = "SPY"
                
                # --- 1. 基本ティッカー群 ---
                yahoo_tickers = settings_df[settings_df['ソース'] == 'Yahoo']['ティッカー'].unique().tolist()
                fred_tickers = settings_df[settings_df['ソース'] == 'FRED']['ティッカー'].unique().tolist()

                # --- 2. クオンツレベル：超・先行指標群の強制注入 ---
                market_implied_tickers = ['HG=F', 'GC=F', 'XLY', 'XLP']
                for tic in market_implied_tickers:
                    if tic not in yahoo_tickers: yahoo_tickers.append(tic)
                if target_ticker not in yahoo_tickers: yahoo_tickers.append(target_ticker)

                # プロ用FRED指標 (流動性、純粋ストレス、労働先行、期待インフレ、住宅、マインド)
                pro_fred_tickers = [
                    'ANFCI',        # シカゴ連銀 調整済み金融環境指数
                    'STLFSI4',      # セントルイス連銀 金融ストレス指数
                    'T10Y3M',       # イールドカーブ (10年-3ヶ月)
                    'BAMLH0A0HYM2', # HY債スプレッド
                    'WALCL',        # FRB総資産 (流動性)
                    'M2SL',         # M2マネーストック (通貨供給量)
                    'ICSA',         # 新規失業保険申請件数 (週次)
                    'AWHAEMAN',     # 製造業 平均週労働時間
                    'T5YIFR',       # 5年先5年物期待インフレ率
                    'PERMIT',       # 住宅建築許可件数 (実体経済の最強先行指標)
                    'UMCSENT'       # ミシガン大消費者態度指数
                ]
                for tic in pro_fred_tickers:
                    if tic not in fred_tickers: fred_tickers.append(tic)

                # --- 3. データのバルク取得 ---
                series_list = []
                if yahoo_tickers:
                    y_data = yf.download(yahoo_tickers, period="5y", progress=False)['Close']
                    if len(yahoo_tickers) == 1: y_data = pd.DataFrame(y_data, columns=yahoo_tickers)
                    for col in y_data.columns:
                        series_list.append(y_data[col].dropna().rename(col))
                
                for tic in fred_tickers:
                    try:
                        series_list.append(fred.get_series(tic).loc['2019-01-01':].dropna().rename(tic))
                    except: pass

                # タイムゾーン正規化と結合
                for i in range(len(series_list)):
                    series_list[i].index = pd.to_datetime(series_list[i].index).tz_localize(None).normalize()
                df_ml = pd.concat(series_list, axis=1).ffill()
                
                # --- 4. 高度な特徴量エンジニアリング ---
                if 'HG=F' in df_ml.columns and 'GC=F' in df_ml.columns:
                    df_ml['Copper_Gold_Ratio'] = df_ml['HG=F'] / df_ml['GC=F']
                if 'XLY' in df_ml.columns and 'XLP' in df_ml.columns:
                    df_ml['Risk_Appetite_Ratio'] = df_ml['XLY'] / df_ml['XLP']

                df_ml['SPY_Mom_1M'] = df_ml[target_ticker].pct_change(21)
                df_ml['SPY_Vol_1M'] = df_ml[target_ticker].pct_change().rolling(21).std() * np.sqrt(252)
                
                # 大統領選サイクル・アノマリーの計算 (0:選挙年, 1:就任年, 2:中間選挙, 3:選挙前年)
                df_ml['Presidential_Cycle'] = df_ml.index.year % 4
                
                # Target: 21営業日後(約1ヶ月後)のリターン
                df_ml['Target'] = df_ml[target_ticker].pct_change(21).shift(-21)
                
                # データクレンジング
                df_train = df_ml.dropna()
                
                # 特徴量の選別 (アノマリーモード判定)
                exclude_cols = ['Target', 'HG=F', 'GC=F', 'XLY', 'XLP']
                if "除外する" in anomaly_mode:
                    exclude_cols.append('Presidential_Cycle')
                
                features = [col for col in df_train.columns if col not in exclude_cols]
                X, y = df_train[features], df_train['Target']
                
                # --- 5. 機械学習 (Random Forest) ---
                weights = np.exp(np.linspace(-1, 0, len(X)))
                model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42)
                model.fit(X, y, sample_weight=weights)
                
                latest_x = df_ml[features].iloc[-1:]
                pred_ret = model.predict(latest_x)[0] * 100

                # --- 6. Explainable AI (Z-Score & Feature Attribution) ---
                df_3y = df_ml.tail(252 * 3)
                
                # アノマリー(カテゴリ変数)はZ-Score計算から除外し、マクロ指標のみでZ-Scoreを計算
                macro_features = [f for f in features if f != 'Presidential_Cycle']
                z_scores_macro = (latest_x[macro_features].iloc[0] - df_3y[macro_features].mean()) / df_3y[macro_features].std()
                
                # 特徴量重要度DFの作成
                imp_data = []
                for f, imp in zip(features, model.feature_importances_):
                    z = z_scores_macro[f] if f in macro_features else 0.0 # サイクル変数は色付けなし(0)とする
                    imp_data.append({'Feature': f, 'Importance': imp, 'Current_Z_Score': z})
                
                imp_df = pd.DataFrame(imp_data).sort_values('Importance', ascending=False).head(12)

                # ==========================================
                # UI Rendering
                # ==========================================
                st.markdown("---")
                
                # --- クオンツディクショナリー (折りたたみ) ---
                with st.expander("📚 クオンツ指標ディクショナリー (各指標の解説と見方)", expanded=False):
                    st.markdown("""
                    **【流動性・金融ストレス指標】**
                    * **ANFCI (調整済み金融環境指数):** シカゴ連銀が算出。インフレや経済成長の影響を統計的に排除した「純粋な金融システムのストレス」。プラスなら引き締め的。
                    * **STLFSI4 (金融ストレス指数):** 金利スプレッドなど18指標の合成。ゼロが平均。暴落時には急騰する。
                    * **WALCL (FRB総資産):** 中央銀行のバランスシート。上昇は市場への資金供給（株高要因）、下落はQT（株安要因）。
                    * **M2SL (M2マネーストック):** 市中に出回る通貨の総量。これが縮小するとリスク資産は構造的に上がりづらくなる。
                    
                    **【実体経済の先行指標】**
                    * **T10Y3M (10年-3ヶ月金利差):** FRBのパウエル議長も注視する最も確実なリセッション先行指標。逆イールド（マイナス）後の順イールド化が最も危険。
                    * **BAMLH0A0HYM2 (ハイイールド債スプレッド):** ジャンク債の金利上乗せ幅。企業の倒産リスク。これが広がり始めたら株は売られる。
                    * **PERMIT (住宅建築許可件数):** 家を建てる前の「許可」件数。木材価格や家電消費、雇用に波及するため、実体経済の最強の先行指標となる。
                    * **ICSA (新規失業保険申請件数):** 雇用統計を待たずに、毎週のレイオフ状況を最速で捉える。
                    * **AWHAEMAN (製造業平均労働時間):** 企業は「解雇」の前に「残業カット」を行うため、雇用悪化の超・先行シグナル。
                    
                    **【市場内包シグナル・アノマリー】**
                    * **Copper_Gold_Ratio (銅/金レシオ):** 景気に敏感な銅と、安全資産の金の比率。グローバル経済の体温計。米債利回りと強い正の相関を持つ。
                    * **Risk_Appetite_Ratio (XLY/XLP):** 裁量消費（攻め）と生活必需品（守り）の比率。機関投資家が現在どちらに資金をシフトしているかが分かる。
                    * **Presidential_Cycle (大統領選サイクル):** 0=選挙年、1=就任年、2=中間選挙年、3=選挙前年。米国株は「中間選挙年の秋に底を打ち、選挙前年に爆上げする」という強力なアノマリーを持つ。
                    """)

                # --- AI脳内可視化 ---
                st.subheader("1. AI Brain Visualization (AIの思考プロセスと異常値検知)")
                st.markdown("モデルが重視しているマクロ指標（棒の長さ）と、現在の値の過去3年平均からの乖離度（Z-Score / 色）です。")
                
                c_ai1, c_ai2 = st.columns([2, 1])
                with c_ai1:
                    fig_brain = px.bar(
                        imp_df.sort_values('Importance', ascending=True), 
                        x='Importance', y='Feature', orientation='h',
                        color='Current_Z_Score', 
                        color_continuous_scale='RdBu_r', 
                        range_color=[-3, 3],
                        title="Feature Importance × Current Abnormalities (Z-Score)"
                    )
                    fig_brain.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_brain, use_container_width=True)
                
                with c_ai2:
                    st.markdown("#### 🔍 クオンツ・インサイト")
                    st.markdown("* <span style='color:#f85149;'>**赤 (Hot):**</span> +2σ以上の異常値。")
                    st.markdown("* <span style='color:#58a6ff;'>**青 (Cold):**</span> -2σ以下の異常値。")
                    st.info(f"【現在のアノマリー状況】\n現在の年は大統領選サイクルにおいて **「{['選挙年', '就任年', '中間選挙年', '選挙前年'][df_ml['Presidential_Cycle'].iloc[-1]]}」** に該当します。\n※グラフ内で色が白い（0）の項目は、カテゴリ変数（アノマリー等）です。")

                # --- Market vs AI Gap Analysis ---
                st.markdown("---")
                st.subheader("2. Swing Trade Setup (理論値とのスプレッド)")
                c_g1, c_g2 = st.columns([2, 1])
                with c_g1:
                    td = df_ml.tail(90).copy()
                    td['AI_Fair'] = (model.predict(td[features]) + 1) * td[target_ticker]
                    fg = go.Figure()
                    fg.add_trace(go.Scatter(x=td.index, y=td[target_ticker], name="Market Price", line=dict(color='#ff4b4b', width=2)))
                    fg.add_trace(go.Scatter(x=td.index, y=td['AI_Fair'], name=f"AI Fair Value ({'Anomaly' if '反映' in anomaly_mode else 'Standard'})", line=dict(dash='dot', color='#58a6ff')))
                    st.plotly_chart(fg.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified"), use_container_width=True)
                with c_g2:
                    gap = ((df_ml[target_ticker].iloc[-1] - td['AI_Fair'].iloc[-1]) / td['AI_Fair'].iloc[-1]) * 100
                    st.metric("AI理論値からの乖離率 (Spread)", f"{gap:+.2f}%")
                    st.write("市場価格がAIフェアバリューから上方に大きく乖離している場合、トレンド・エグゾースチョン（買い疲れ）による反落リスクが高まります。")

                # --- 統合戦略判断 ---
                st.markdown("---")
                st.subheader("3. Executive Strategy Decision (1-Month Horizon)")
                cl1, cl2 = st.columns([2, 1])
                with cl1:
                    stress_val = df_ml['ANFCI'].iloc[-1] if 'ANFCI' in df_ml.columns else 0
                    score = pred_ret - (stress_val * 2) 
                    
                    if score > 2: status, color, icon = "積極的投資 (Aggressive Long)", "#00ff00", "🚀"
                    elif score > -1: status, color, icon = "部分的投資 (Cautious)", "#58a6ff", "⚖️"
                    else: status, color, icon = "防御的待機 (Risk Off)", "#f85149", "🛡️"
                    
                    st.markdown(f"<div class='insight-box' style='border-left: 5px solid {color};'><h3>AI判定: {icon} {status}</h3><p style='font-size:18px;'>AI 1ヶ月先 予測リターン: <b>{pred_ret:+.2f}%</b></p><p>ANFCI 金融ストレス水準: {stress_val:+.2f}</p></div>", unsafe_allow_html=True)
                with cl2:
                    vix_val = df_ml['^VIX'].iloc[-1] if '^VIX' in df_ml.columns else 20
                    risk = max(0, min(100, (1/vix_val)*1500))
                    if stress_val > 0: risk *= 0.8
                    
                    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=risk, title={'text':"Target Risk Exposure (%)"}, gauge={'axis':{'range':[0,100]},'bar':{'color':color}})).update_layout(template="plotly_dark", height=200, margin=dict(t=30,b=0)), use_container_width=True)

            except Exception as e: st.error(f"クオンツエンジン実行エラー: {e}")
except Exception as e:
    st.error(f"System Critical Error: {e}")
