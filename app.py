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
        # スイング相関
        st.markdown("### 🔗 Swing Correlation Matrix (1W/1H)")
        assets = {'SPY':'Stock', 'TLT':'Bond', 'GLD':'Gold', 'USO':'Oil', 'UUP':'Dollar', 'BTC-USD':'Crypto'}
        c_data = yf.download(list(assets.keys()), period="1wk", interval="1h", progress=False)['Close'].rename(columns=assets).corr()
        st.plotly_chart(px.imshow(c_data, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark", height=400), use_container_width=True)

        tabs_names = [t for t in settings_df['タブ名'].unique() if "ダッシュボード" not in t]
        tabs = st.tabs(tabs_names)
        for i, t_name in enumerate(tabs_names):
            with tabs[i]:
                t_df = settings_df[settings_df['タブ名'] == t_name]
                cols = st.columns(2)
                for g_idx, g_name in enumerate(t_df['グラフ名'].unique()):
                    with cols[g_idx % 2]:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        g_data, max_dt = t_df[t_df['グラフ名'] == g_name], None
                        for _, r in g_data.iterrows():
                            try:
                                d = None
                                if r['ソース'] == 'FRED': d = fred.get_series(r['ティッカー']).loc['2022-01-01':]
                                elif r['ソース'] == 'Yahoo': d = yf.Ticker(r['ティッカー']).history(start='2022-01-01')['Close']
                                elif r['ソース'] == 'DBnomics': d = fetch_series(r['ティッカー']).set_index('period')['value']
                                if d is not None:
                                    d.index = pd.to_datetime(d.index).tz_localize(None)
                                    if max_dt is None or d.index.max() > max_dt: max_dt = d.index.max()
                                    fig.add_trace(go.Scatter(x=d.index, y=d.values, name=r['データ名']), secondary_y=(r['軸']=='副軸'))
                            except: pass
                        if max_dt: fig.update_xaxes(range=[max_dt - pd.DateOffset(months=6), max_dt])
                        st.plotly_chart(fig.update_layout(title=g_name, height=300, template="plotly_dark"), use_container_width=True)

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
    # PAGE 4: Institutional Quant Engine
    # ==========================================
    elif page == "4. Investment Strategy (AI戦略)":
        st.title("🧠 Institutional Quant Strategy Engine")
        st.markdown("ガウス混合モデルによるレジーム判定、ボラティリティ・ターゲティング、そしてCTA/オプションの代替特徴量を用いた機関投資家レベルのクオンツ解析を実行します。")

        with st.spinner('高度な数学的モデリングとバックテストを実行中... (計算に時間がかかります)'):
            try:
                # --- 1. データのバルク取得とプロキシ生成 ---
                target_ticker = "SPY"
                
                # スプレッドシートからの動的取得に加えて、高度解析に必須なコアデータを強制取得
                core_tickers = [target_ticker, "^VIX", "^VIX3M", "DX-Y.NYB"]
                y_data = yf.download(core_tickers, period="10y", progress=False)['Close']
                y_data.index = pd.to_datetime(y_data.index).tz_localize(None).normalize()
                df_ml = y_data.ffill().dropna()

                # --- 2. Advanced Feature Engineering (プロのクオンツ指標) ---
                # A. CTA Trend Proxy (トレンドフォロワーのポジション推定)
                df_ml['SMA50'] = df_ml[target_ticker].rolling(50).mean()
                df_ml['SMA200'] = df_ml[target_ticker].rolling(200).mean()
                df_ml['CTA_Dist_200'] = (df_ml[target_ticker] / df_ml['SMA200']) - 1 # 200日線乖離
                df_ml['CTA_Cross'] = (df_ml['SMA50'] / df_ml['SMA200']) - 1          # ゴールデン/デッドクロス強度

                # B. Options Market Proxy (オプション市場のストレス構造)
                # VIX(1ヶ月) / VIX3M(3ヶ月) が1を超えるとバックワーデーション(プット需要の極端な急増)
                df_ml['Vol_Term_Structure'] = df_ml['^VIX'] / df_ml['^VIX3M'] 
                
                # C. Momentum & Volatility
                df_ml['Return_1M'] = df_ml[target_ticker].pct_change(21)
                df_ml['Vol_1M'] = df_ml[target_ticker].pct_change().rolling(21).std() * np.sqrt(252)

                # Target (21営業日後のリターン)
                df_ml['Target'] = df_ml[target_ticker].pct_change(21).shift(-21)
                
                # 欠損値処理
                df_ml = df_ml.dropna()
                
                # --- 3. Market Regime Detection (ガウス混合モデルによる相場環境認識) ---
                # リターンとボラティリティから、相場を2つの状態(レジーム)にクラスタリング
                regime_features = df_ml[['Return_1M', 'Vol_1M']]
                gmm = GaussianMixture(n_components=2, random_state=42)
                df_ml['Regime'] = gmm.fit_predict(regime_features)
                
                # ボラティリティが高い方を「Risk-Off (1)」、低い方を「Risk-On (0)」にラベル統一
                vol_regime0 = df_ml[df_ml['Regime'] == 0]['Vol_1M'].mean()
                vol_regime1 = df_ml[df_ml['Regime'] == 1]['Vol_1M'].mean()
                if vol_regime0 > vol_regime1:
                    df_ml['Regime'] = 1 - df_ml['Regime'] # 反転
                
                curr_regime = df_ml['Regime'].iloc[-1]
                regime_name = "🔴 Risk-Off Regime (高ボラティリティ・警戒相場)" if curr_regime == 1 else "🟢 Risk-On Regime (低ボラティリティ・安定相場)"

                # --- 4. Machine Learning (予測エンジン) ---
                features = ['^VIX', 'DX-Y.NYB', 'CTA_Dist_200', 'CTA_Cross', 'Vol_Term_Structure', 'Return_1M', 'Vol_1M']
                X, y = df_ml[features], df_ml['Target']
                
                # 直近重視の指数関数的ウェイト
                weights = np.exp(np.linspace(-2, 0, len(X)))
                model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
                model.fit(X, y, sample_weight=weights)
                
                latest_x = df_ml[features].iloc[-1:]
                pred_ret = model.predict(latest_x)[0] * 100

                # --- 5. Backtest Engine (プロ仕様のバックテストとリスク指標) ---
                # 過去のデータに対してモデルの予測値を計算し、取引シグナルを生成
                df_ml['Predicted_Ret'] = model.predict(X)
                df_ml['Signal'] = np.where(df_ml['Predicted_Ret'] > 0, 1, 0) # 予測がプラスならフルインベスト、マイナスなら現金
                
                # 翌日の日次リターンを計算 (バックテスト用)
                df_ml['Daily_Ret'] = df_ml[target_ticker].pct_change().shift(-1)
                df_ml['Strategy_Ret'] = df_ml['Signal'] * df_ml['Daily_Ret']
                
                # 直近3年間のバックテスト結果を抽出
                bt_df = df_ml.tail(252 * 3).dropna()
                bt_df['Equity_B&H'] = (1 + bt_df['Daily_Ret']).cumprod() * 100
                bt_df['Equity_AI'] = (1 + bt_df['Strategy_Ret']).cumprod() * 100
                
                # 統計指標の計算 (Sharpe Ratio, Max Drawdown)
                def calc_metrics(returns):
                    ann_ret = returns.mean() * 252
                    ann_vol = returns.std() * np.sqrt(252)
                    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
                    cum_ret = (1 + returns).cumprod()
                    drawdown = (cum_ret / cum_ret.cummax()) - 1
                    mdd = drawdown.min()
                    return sharpe, mdd

                sharpe_bh, mdd_bh = calc_metrics(bt_df['Daily_Ret'])
                sharpe_ai, mdd_ai = calc_metrics(bt_df['Strategy_Ret'])

                # ==========================================
                # UI Rendering (描画セクション)
                # ==========================================
                
                # --- セクション1: 相場環境とAI予測 ---
                st.subheader("1. Market Regime & AI Predictive Model")
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    st.markdown(f"""
                    <div class='insight-box'>
                        <h4>現在の相場環境 (GMM判定)</h4>
                        <h3 style="margin-top:0px;">{regime_name}</h3>
                        <p><b>AI予測 1ヶ月期待リターン:</b> <span style="font-size:24px; color:{'#00ff00' if pred_ret>0 else '#f85149'};">{pred_ret:+.2f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_r2:
                    st.markdown("**AI Feature Attribution (判断の根拠)**")
                    imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
                    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', template="plotly_dark", color='Importance', color_continuous_scale='Blues')
                    st.plotly_chart(fig_imp.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

                st.markdown("---")

                # --- セクション2: バックテストとリスク指標 ---
                st.subheader("2. Quantitative Backtest (直近3年間)")
                st.markdown("AIが「期待リターンがマイナス」と予測した日に現金を保有した場合の、リスク回避効果を測定します。")
                
                col_b1, col_b2 = st.columns([3, 1])
                with col_b1:
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Equity_AI'], name="AI Strategy", line=dict(color='#3fb950', width=3)))
                    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Equity_B&H'], name="Buy & Hold", line=dict(color='gray', dash='dot')))
                    
                    # 現金化した期間(Signal=0)を赤色でハイライト
                    cash_df = bt_df[bt_df['Signal'] == 0]
                    fig_bt.add_trace(go.Scatter(x=cash_df.index, y=cash_df['Equity_AI'], mode='markers', name="Cash Position", marker=dict(color='#f85149', size=4)))
                    
                    fig_bt.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified")
                    st.plotly_chart(fig_bt, use_container_width=True)
                
                with col_b2:
                    st.markdown("#### Risk Metrics")
                    st.markdown(f"**Sharpe Ratio**\n* AI: `{sharpe_ai:.2f}`\n* B&H: `{sharpe_bh:.2f}`")
                    st.markdown(f"**Max Drawdown**\n* AI: `<span style='color:#f85149;'>{mdd_ai*100:.1f}%</span>`\n* B&H: `{mdd_bh*100:.1f}%`", unsafe_allow_html=True)
                    st.caption("※シャープレシオは高いほど良く、ドローダウン(暴落率)はゼロに近いほど優れています。")

                st.markdown("---")

                # --- セクション3: プロの資金管理 (Volatility Targeting) ---
                st.subheader("3. Institutional Position Sizing (Target Volatility Framework)")
                
                # 機関投資家の標準的なターゲットボラティリティ(年率15%)に基づくポジションサイジング
                target_vol = 0.15 
                curr_vol = df_ml['Vol_1M'].iloc[-1]
                base_weight = target_vol / curr_vol if curr_vol > 0 else 0
                
                # AIの自信度とレジームによる調整
                adj_weight = base_weight
                if pred_ret < 0: adj_weight *= 0.2  # 弱気予測時は大幅減
                if curr_regime == 1: adj_weight *= 0.8 # リスクオフレジーム時はさらに2割減
                
                final_weight_pct = min(100, max(0, adj_weight * 100)) # 0-100%に丸める

                col_s1, col_s2 = st.columns([1, 2])
                with col_s1:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=final_weight_pct, number={"suffix": "%"}, title={'text':"最適ポジション比率"}, 
                        gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#58a6ff"}, 'steps':[{'range':[0,30],'color':"#f85149"}, {'range':[30,70],'color':"#e3b341"}]}
                    ))
                    st.plotly_chart(fig_g.update_layout(template="plotly_dark", height=250, margin=dict(t=40,b=0)), use_container_width=True)
                
                with col_s2:
                    st.markdown("#### アクション・ロジック (Why this size?)")
                    st.write(f"1. **市場の荒れ具合 (Current Volatility):** 現在の年率ボラティリティは `{curr_vol*100:.1f}%` です。目標リスク(15%)を維持するための基準ウェイトは `{base_weight*100:.0f}%` と計算されました。")
                    st.write(f"2. **レジーム調整:** 現在は `{regime_name}` のため、安全係数を掛けています。")
                    st.write(f"3. **AI方向性調整:** 1ヶ月予測が `{pred_ret:+.2f}%` であることを加味し、最終的な投下資本比率を **`{final_weight_pct:.1f}%`** に設定しました。残りは現金(Cash)または短期債券(SHV)で待機してください。")

            except Exception as e: 
                st.error(f"クオンツエンジン実行エラー: {e}\n(データの取得に失敗した可能性があります。時間をおいて再度お試しください)")

except Exception as e:
    st.error(f"System Critical Error: {e}")
