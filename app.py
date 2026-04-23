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
    # PAGE 4: Dynamic AI Strategy (スプレッドシート連動型)
    # ==========================================
    elif page == "4. Investment Strategy (AI戦略)":
        st.title("🧠 Dynamic AI Strategy & Gap Analysis")
        st.markdown("Page 1 & 2で定義されたすべてのマクロ指標を統合し、AIが最適な予測モデルを自動構築します。")

        with st.spinner('スプレッドシートから全指標を抽出し、AIを学習中... (約10〜20秒)'):
            try:
                # --- 1. スプレッドシートから動的にティッカーを抽出 ---
                target_ticker = "SPY" # 予測の目的変数
                
                # Yahoo系のティッカー抽出
                yahoo_tickers = settings_df[settings_df['ソース'] == 'Yahoo']['ティッカー'].unique().tolist()
                if target_ticker not in yahoo_tickers: yahoo_tickers.append(target_ticker)
                
                # FRED系のティッカー抽出
                fred_tickers = settings_df[settings_df['ソース'] == 'FRED']['ティッカー'].unique().tolist()

                # --- 2. データのバルク取得 ---
                series_list = []
                
                # Yahooデータの一括取得
                if yahoo_tickers:
                    y_data = yf.download(yahoo_tickers, period="5y", progress=False)['Close']
                    if len(yahoo_tickers) == 1: y_data = pd.DataFrame(y_data, columns=yahoo_tickers)
                    for col in y_data.columns:
                        s = y_data[col].dropna().rename(col)
                        series_list.append(s)
                
                # FREDデータの個別取得
                for tic in fred_tickers:
                    try:
                        s = fred.get_series(tic).loc['2019-01-01':].dropna().rename(tic)
                        series_list.append(s)
                    except: pass

                # --- 3. タイムゾーン正規化とデータの結合 ---
                for i in range(len(series_list)):
                    series_list[i].index = pd.to_datetime(series_list[i].index).tz_localize(None).normalize()
                
                # 月次データ等で生じるNaNを前方の値で埋める(ffill)
                df_ml = pd.concat(series_list, axis=1).ffill()
                
                # 特徴量エンジニアリング (目的変数 SPY のモメンタムとボラティリティを追加)
                df_ml['SPY_Mom_1M'] = df_ml[target_ticker].pct_change(21)
                df_ml['SPY_Vol_1M'] = df_ml[target_ticker].pct_change().rolling(21).std() * np.sqrt(252)
                
                # AIが予測するターゲット (21営業日後のSPYリターン)
                df_ml['Target'] = df_ml[target_ticker].pct_change(21).shift(-21)
                
                # 欠損値を含む行を削除し、学習可能な完全データセットを作成
                df_train = df_ml.dropna()
                
                # ターゲット変数以外のすべてを特徴量(X)とする
                features = [col for col in df_train.columns if col != 'Target']
                X, y = df_train[features], df_train['Target']
                
                # --- 4. 機械学習モデルの構築 ---
                # 直近のデータほど重みを大きくする
                weights = np.exp(np.linspace(-1, 0, len(X)))
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10).fit(X, y, sample_weight=weights)
                
                # 最新データで予測
                latest_x = df_ml[features].iloc[-1:]
                pred_ret = model.predict(latest_x)[0] * 100

                # --- 5. ダッシュボード表示: 乖離分析 ---
                st.subheader("1. Market vs AI Gap Analysis")
                c_g1, c_g2 = st.columns([2, 1])
                with c_g1:
                    td = df_ml.tail(60).copy()
                    td['AI_Fair'] = (model.predict(td[features]) + 1) * td[target_ticker]
                    fg = go.Figure()
                    fg.add_trace(go.Scatter(x=td.index, y=td[target_ticker], name="Market Price (SPY)", line=dict(color='#ff4b4b', width=2)))
                    fg.add_trace(go.Scatter(x=td.index, y=td['AI_Fair'], name="AI Theoretical Value", line=dict(dash='dot', color='#58a6ff')))
                    st.plotly_chart(fg.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=20,b=0)), use_container_width=True)
                with c_g2:
                    gap = ((df_ml[target_ticker].iloc[-1] - td['AI_Fair'].iloc[-1]) / td['AI_Fair'].iloc[-1]) * 100
                    st.metric("AI理論値との乖離率 (Spread)", f"{gap:+.2f}%")
                    if gap > 2: st.warning("⚠️ 理論値を上回る過熱状態")
                    elif gap < -2: st.success("✅ マクロ的に割安水準")
                    else: st.info("⚖️ 適正水準")

                # --- 6. ダッシュボード表示: 特徴量重要度 ---
                st.markdown("---")
                st.subheader("2. Model Insights: AIは何を見て判断したか？")
                st.markdown("あなたが設定したマクロ指標群の中で、**現在S&P500の予測に最も貢献している指標トップ10**です。")
                
                imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(10)
                fig_imp = px.bar(imp_df.sort_values('Importance', ascending=True), x='Importance', y='Feature', orientation='h', template="plotly_dark", color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig_imp.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

                # --- 7. 統合戦略判断 ---
                st.markdown("---")
                st.subheader("3. Executive Strategy Decision")
                cl1, cl2 = st.columns([2, 1])
                with cl1:
                    # 簡易的なボラティリティ(VIX)を安全弁としてスコア化
                    vix_val = df_ml['^VIX'].iloc[-1] if '^VIX' in df_ml.columns else 20
                    score = np.clip(pred_ret * 2, -5, 5) + (1 if vix_val < 20 else -1)
                    
                    if score > 2: status, color, icon = "積極的投資", "#00ff00", "🚀"
                    elif score > -1: status, color, icon = "部分的投資", "#58a6ff", "⚖️"
                    else: status, color, icon = "防御的待機", "#f85149", "🛡️"
                    
                    st.markdown(f"<div class='insight-box' style='border-left: 5px solid {color};'><h3>AI判定: {icon} {status}</h3><p style='font-size:18px;'>AI予測 1ヶ月リターン: <b>{pred_ret:+.2f}%</b></p></div>", unsafe_allow_html=True)
                with cl2:
                    risk = max(0, min(100, (1/vix_val)*1500))
                    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=risk, title={'text':"推奨リスク露出度"}, gauge={'axis':{'range':[0,100]},'bar':{'color':color}})).update_layout(template="plotly_dark", height=200, margin=dict(t=30,b=0)), use_container_width=True)

            except Exception as e: st.error(f"AI学習エラー: {e}\n(※データが不足しているティッカーがスプレッドシートにある可能性があります)")

    except Exception as e:
        st.error(f"System Critical Error: {e}")
   
except Exception as e:
    st.error(f"システムクリティカルエラー: {e}")
