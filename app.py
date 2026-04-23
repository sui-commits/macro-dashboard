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
    # PAGE 4: Institutional Quant Engine (LLM Integration)
    # ==========================================
    elif page == "4. Investment Strategy (AI戦略)":
        st.title("🧠 Institutional Quant Strategy Engine")
        st.markdown("マクロ経済、グローバル流動性、需給に特化した予測モデルを構築し、LLM解釈用のプロンプトを出力します。")

        with st.sidebar.expander("🛠️ モデル・パラメータ設定", expanded=True):
            exclude_other_stocks = st.checkbox("他の株式指数を除外する (推奨)", value=True, help="QQQ, DIAなどの他指数を排除し、マクロと需給のみで説明します。")
            include_anomaly = st.checkbox("大統領選サイクルを含める", value=False)
            lookback_years = st.slider("学習期間 (年)", 3, 10, 5)

        with st.spinner('クオンツ・フィルタリングおよびモデル構築中...'):
            try:
                target_ticker = "SPY"
                
                # --- 1 & 2. データの取得と結合 ---
                yahoo_tickers = settings_df[settings_df['ソース'] == 'Yahoo']['ティッカー'].unique().tolist()
                market_implied_tickers = ['HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                all_yahoo = list(set(yahoo_tickers + market_implied_tickers + [target_ticker]))

                pro_fred_tickers = ['ANFCI', 'STLFSI4', 'T10Y3M', 'BAMLH0A0HYM2', 'WALCL', 'M2SL', 'ICSA', 'AWHAEMAN', 'T5YIFR', 'PERMIT', 'UMCSENT']
                all_fred = list(set(settings_df[settings_df['ソース'] == 'FRED']['ティッカー'].unique().tolist() + pro_fred_tickers))

                series_list = []
                y_data = yf.download(all_yahoo, period=f"{lookback_years}y", progress=False)['Close']
                if isinstance(y_data, pd.Series): y_data = y_data.to_frame(all_yahoo[0])
                for col in y_data.columns:
                    series_list.append(y_data[col].dropna().rename(col))
                
                for tic in all_fred:
                    try:
                        s = fred.get_series(tic).loc[f"{2024-lookback_years}-01-01":].dropna().rename(tic)
                        series_list.append(s)
                    except: pass

                for i in range(len(series_list)):
                    series_list[i].index = pd.to_datetime(series_list[i].index).tz_localize(None).normalize()
                df_ml = pd.concat(series_list, axis=1).ffill()
                
                # --- 3. 特徴量エンジニアリング ---
                df_ml['CTA_200D_Bias'] = df_ml[target_ticker] / df_ml[target_ticker].rolling(200).mean() - 1
                df_ml['Vol_Term_Spread'] = df_ml['^VIX'] / df_ml['^VIX3M']
                df_ml['Copper_Gold_Ratio'] = df_ml['HG=F'] / df_ml['GC=F']
                df_ml['Presidential_Cycle'] = df_ml.index.year % 4
                df_ml['Target'] = df_ml[target_ticker].pct_change(21).shift(-21)
                df_train = df_ml.dropna()

                # --- 4. フィルタリング ---
                drop_cols = ['Target', 'HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                if exclude_other_stocks:
                    stock_indices = [t for t in yahoo_tickers if t != target_ticker]
                    drop_cols.extend(stock_indices)
                if not include_anomaly:
                    drop_cols.append('Presidential_Cycle')

                features = [col for col in df_train.columns if col not in drop_cols]
                X, y = df_train[features], df_train['Target']
                
                # 学習
                weights = np.exp(np.linspace(-1, 0, len(X)))
                model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X, y, sample_weight=weights)
                
                latest_x = df_ml[features].iloc[-1:]
                pred_ret = model.predict(latest_x)[0] * 100

                # Z-Score解析
                df_3y = df_ml.tail(252*3)
                macro_only = [f for f in features if f != 'Presidential_Cycle']
                z_scores = (latest_x[macro_only].iloc[0] - df_3y[macro_only].mean()) / df_3y[macro_only].std()
                
                imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                imp_df['Z_Score'] = imp_df['Feature'].map(z_scores).fillna(0)
                imp_df = imp_df.sort_values('Importance', ascending=False).head(12)

                # --- UI表示 ---
                st.subheader("1. AI Interpretation: Feature Importance")
                fig_brain = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h',
                                   color='Z_Score', color_continuous_scale='RdBu_r', range_color=[-3, 3], template="plotly_dark")
                st.plotly_chart(fig_brain.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0)), use_container_width=True)

                st.markdown("---")
                st.subheader("2. Market vs AI Fair Value")
                td = df_ml.tail(90).copy()
                td['AI_Fair'] = (model.predict(td[features]) + 1) * td[target_ticker]
                gap = ((df_ml[target_ticker].iloc[-1] - td['AI_Fair'].iloc[-1]) / td['AI_Fair'].iloc[-1]) * 100
                
                fig_gap = go.Figure()
                fig_gap.add_trace(go.Scatter(x=td.index, y=td[target_ticker], name="SPY Price", line=dict(color='#ff4b4b', width=2)))
                fig_gap.add_trace(go.Scatter(x=td.index, y=td['AI_Fair'], name="AI Fair Value", line=dict(dash='dot', color='#58a6ff')))
                st.plotly_chart(fig_gap.update_layout(template="plotly_dark", height=300, hovermode="x unified"), use_container_width=True)

                st.markdown("---")
                col_f1, col_f2 = st.columns([2, 1])
                with col_f1:
                    st.markdown(f"<div class='insight-box'><h4>AI 1ヶ月予測リターン</h4><h2>{pred_ret:+.2f}%</h2></div>", unsafe_allow_html=True)
                with col_f2:
                    vix_val = df_ml['^VIX'].iloc[-1]
                    risk = max(0, min(100, (1/vix_val)*1500))
                    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=risk, title={'text':"推奨リスク露出度"}, gauge={'bar':{'color':'#58a6ff'}})).update_layout(template="plotly_dark", height=200, margin=dict(t=30,b=0)), use_container_width=True)

                # ==========================================
                # 🤖 LLM解説プロンプトの動的生成
                # ==========================================
                st.markdown("---")
                st.subheader("🤖 LLM Analysis Prompt (言語化用プロンプト)")
                st.markdown("以下のテキストをコピーして、GeminiやChatGPTに貼り付けてください。現在のデータを元に、プロのクオンツ目線で相場環境を解説してくれます。")

                top_features_text = ""
                for _, row in imp_df.head(6).iterrows():
                    top_features_text += f"- {row['Feature']}: 重要度 {row['Importance']:.4f}, 現在のZ-Score {row['Z_Score']:+.2f}\\n"

                anfci_val = df_ml['ANFCI'].iloc[-1] if 'ANFCI' in df_ml.columns else "データなし"
                cycle_val = df_ml['Presidential_Cycle'].iloc[-1]
                cycle_dict = {0: "選挙年", 1: "就任年", 2: "中間選挙年", 3: "選挙前年"}

                llm_prompt = f"""あなたは世界トップクラスのマクロ・クオンツ・ファンドのシニア・ストラテジストです。
以下の自社開発AIモデル（Random Forest）が出力した最新の算出結果を元に、現在の相場環境で「どのマクロ指標が支配的か」を言語化し、プロ目線の市場解説レポートを作成してください。

【AIモデル分析結果 (対象: S&P500, スイングホライズン: 1ヶ月)】
・AI予測 1ヶ月リターン: {pred_ret:+.2f}%
・現在価格とAI適正価格の乖離率: {gap:+.2f}% (プラスなら相場が割高・過熱、マイナスなら割安)
・現在のVIX (恐怖指数): {vix_val:.2f}
・現在のANFCI (シカゴ連銀 金融環境指数): {anfci_val}
・現在の大統領選サイクル: {cycle_dict.get(cycle_val, "不明")}

【AIが重視している特徴量上位と、現在値のZ-Score (直近3年比の異常度)】
{top_features_text}
※Z-Scoreが+2以上、または-2以下のものは、過去3年平均から大きく逸脱した「異常値」として相場を強く牽引・圧迫しています。

【出力要件】
1. 現在の市場を牽引（または圧迫）している主要なマクロ要因は何か？（特徴量のZ-Scoreの異常値を元に因果関係を考察してください）
2. AIの適正価格との乖離(Spread)や、金融ストレス(ANFCI)を考慮した、現在の市場の「歪み」やリスク。
3. クオンツ目線での、今後1ヶ月の具体的な投資スタンス（強気/弱気/中立）とアクションプラン。
出力は、機関投資家のポートフォリオ・マネージャー宛てのレポートのような、論理的で洗練されたトーンにしてください。
"""
                st.code(llm_prompt, language="text")

            except Exception as e: 
                st.error(f"分析エラー: {e}")

# ===== ココが重要！システム全体を囲む try の except =====
except Exception as e:
    st.error(f"System Critical Error: {e}")
