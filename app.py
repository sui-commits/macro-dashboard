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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
import scipy.optimize as sco
import numpy.linalg as la

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
.zscore-hot { color: #f85149; font-weight: bold; }
.zscore-cold { color: #58a6ff; font-weight: bold; }
.zscore-neutral { color: #8b949e; }
.kpi-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.kpi-title { color: #8b949e; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.kpi-value { color: #c9d1d9; font-size: 28px; font-weight: 700; margin-bottom: 0px; }
.kpi-sub { font-size: 13px; margin-top: 5px; }
.model-breakdown { font-size: 12px; color: #8b949e; display: flex; justify-content: space-between; border-top: 1px solid #30363d; margin-top: 10px; padding-top: 5px;}
</style>
""", unsafe_allow_html=True)

# --- 2. 認証 & キャッシュ設定（高速化） ---
# 🚨 注意: GitHubに直接APIキーを書かないでください。Streamlit CloudのSecrets機能を使います。
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

@st.cache_data(ttl=3600)
def fetch_market_data(tickers, period=None, start=None):
    try:
        if start:
            data = yf.download(tickers, start=start, progress=False)['Close']
        else:
            data = yf.download(tickers, period=period if period else "5y", progress=False)['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0] if isinstance(tickers, list) else tickers)
        return data
    except Exception as e:
        st.toast(f"データ取得エラー: {e}")
        return pd.DataFrame()

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
        "4. Investment Strategy (ハイブリッドAI戦略)",
        "5. Headline Reverse-Engineering (イベント逆引き)", 
        "6. Portfolio Optimization (アロケーション)",
        "7. Macro Data Explorer (マクロ生データ確認)",
        "8. Hybrid AI Regime Strategy (SOTAモデル)"
    ])

    # ==========================================
    # PAGE 1: Institutional Market Dynamics
    # ==========================================
    if page == "1. Market Dynamics (現在)":
        st.title("📈 Institutional Market Dynamics & Flows")
        st.markdown("ボラティリティ構造、セクター・ローテーション、およびオプション需給から、足元の市場の「歪み」と「資金の逃避先」を特定します。")

        # --- 1. VIX Term Structure ---
        st.subheader("1. Volatility Term Structure (恐怖の構造とイールドカーブ)")
        with st.spinner("Analyzing Volatility Surface..."):
            try:
                vix_tickers = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]
                vix_data = fetch_market_data(vix_tickers, period="5d").iloc[-1]
                
                c_vix1, c_vix2 = st.columns([1.5, 1])
                with c_vix1:
                    terms = ['9 Days', '1 Month', '3 Months', '6 Months']
                    vix_values = [vix_data['^VIX9D'], vix_data['^VIX'], vix_data['^VIX3M'], vix_data['^VIX6M']]
                    
                    fig_vix = go.Figure()
                    fig_vix.add_trace(go.Scatter(x=terms, y=vix_values, mode='lines+markers', 
                                                 marker=dict(size=10, color='#f85149' if vix_values[0] > vix_values[2] else '#3fb950'),
                                                 line=dict(width=3)))
                    fig_vix.update_layout(title="VIX Term Structure Curve", template="plotly_dark", height=250, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_vix, use_container_width=True)
                
                with c_vix2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if vix_values[1] > vix_values[2]: 
                        st.markdown("<div class='alert-box-red'><h4>⚠️ バックワーデーション</h4>短期的な恐怖が中期を上回っています。暴落の警戒レベルが最大です。</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='alert-box-green'><h4>✅ コンタンゴ</h4>恐怖の構造は正常（右肩上がり）です。押し目買いが機能しやすい環境です。</div>", unsafe_allow_html=True)
            except Exception as e: st.warning("VIXデータの取得に失敗しました。")

        st.markdown("---")

        # --- 2. Sector Rotation ---
        st.subheader("2. Sector Rotation (資金循環のクアドラント分析)")
        with st.spinner("Calculating Sector Momentum..."):
            try:
                sectors = {'XLK':'Technology', 'XLF':'Financials', 'XLV':'Health Care', 'XLY':'Consumer Disc', 
                           'XLC':'Communication', 'XLI':'Industrials', 'XLP':'Consumer Staples', 'XLE':'Energy', 
                           'XLU':'Utilities', 'XLRE':'Real Estate', 'XLB':'Materials'}
                
                sec_df = fetch_market_data(list(sectors.keys()), period="1mo")
                
                rot_data = []
                for tic, name in sectors.items():
                    ret_1w = ((sec_df[tic].iloc[-1] / sec_df[tic].iloc[-6]) - 1) * 100
                    ret_1m = ((sec_df[tic].iloc[-1] / sec_df[tic].iloc[0]) - 1) * 100
                    rot_data.append({"Sector": name, "1W_Return": ret_1w, "1M_Return": ret_1m})
                    
                df_rot = pd.DataFrame(rot_data)
                fig_rot = px.scatter(df_rot, x='1W_Return', y='1M_Return', text='Sector', 
                                     color='1M_Return', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
                
                fig_rot.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='White')))
                fig_rot.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
                fig_rot.add_vline(x=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
                
                fig_rot.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Short-term Momentum (1-Week %)", yaxis_title="Medium-term Trend (1-Month %)")
                st.plotly_chart(fig_rot, use_container_width=True)
            except Exception as e: st.warning(f"セクターデータの取得に失敗しました: {e}")

        st.markdown("---")

        # --- 3. Positioning & Options Wall ---
        st.subheader("3. Institutional Positioning & Options Wall")
        target_ticker = st.selectbox("🎯 分析銘柄の選択", ["SPY", "QQQ", "NVDA", "AVGO", "SOXL", "TSLA"])
        c_cta, c_opt = st.columns(2)
        
        with c_cta:
            st.markdown(f"#### 🤖 {target_ticker} CTA Trend Proxy")
            try:
                tkr_data = yf.Ticker(target_ticker).history(period="1y")
                tkr_data.index = pd.to_datetime(tkr_data.index).tz_localize(None)
                tkr_data['SMA50'] = tkr_data['Close'].rolling(50).mean()
                tkr_data['SMA200'] = tkr_data['Close'].rolling(200).mean()
                curr = tkr_data['Close'].iloc[-1]
                dist_200 = ((curr - tkr_data['SMA200'].iloc[-1]) / tkr_data['SMA200'].iloc[-1]) * 100
                
                fig_cta = go.Figure()
                fig_cta.add_trace(go.Scatter(x=tkr_data.index, y=tkr_data['Close'], name='Price', line=dict(color='#c9d1d9')))
                fig_cta.add_trace(go.Scatter(x=tkr_data.index, y=tkr_data['SMA50'], name='50 SMA', line=dict(color='#58a6ff', dash='dot')))
                fig_cta.add_trace(go.Scatter(x=tkr_data.index, y=tkr_data['SMA200'], name='200 SMA', line=dict(color='#ff4b4b', dash='dash')))
                fig_cta.add_trace(go.Scatter(x=tkr_data.index, y=np.where(tkr_data['SMA50']>tkr_data['SMA200'], tkr_data['SMA50'], tkr_data['SMA200']), fill='tonexty', fillcolor='rgba(63,185,80,0.1)', line=dict(width=0), showlegend=False))
                
                fig_cta.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified")
                st.plotly_chart(fig_cta, use_container_width=True)
                st.info(f"**CTA Bias:** 価格は200日線から `{dist_200:+.1f}%` 乖離。現在 **{'強気 (Long)' if dist_200 > 0 else '弱気 (Short)'}** ポジションです。")
            except: pass

        with c_opt:
            st.markdown(f"#### 🎯 {target_ticker} Options Max Pain Magnet")
            try:
                tkr_opt = yf.Ticker(target_ticker)
                opt_dates = tkr_opt.options
                if len(opt_dates) == 0:
                    st.warning("現在、Yahoo Financeからオプションデータが配信されていません。")
                else:
                    exp = opt_dates[0]
                    c, p = tkr_opt.option_chain(exp).calls, tkr_opt.option_chain(exp).puts
                    strikes = sorted(list(set(c['strike']).union(set(p['strike']))))
                    mp, min_l = 0, float('inf')
                    for s in strikes:
                        l = c[c['strike']<s].apply(lambda x:(s-x['strike'])*x['openInterest'], axis=1).sum() + p[p['strike']>s].apply(lambda x:(x['strike']-s)*x['openInterest'], axis=1).sum()
                        if l < min_l: min_l, mp = l, s
                        
                    curr_price = tkr_opt.history(period="1d")['Close'].iloc[-1]
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = curr_price,
                        delta = {'reference': mp, 'position': "top", 'formatter': "{+g} points to Wall"},
                        title = {'text': f"Target: ${mp:.0f} (Exp: {exp})"},
                        gauge = {
                            'axis': {'range': [mp * 0.9, mp * 1.1]},
                            'bar': {'color': "#58a6ff"},
                            'threshold': {'line': {'color': "yellow", 'width': 4}, 'thickness': 0.75, 'value': mp}
                        }
                    ))
                    fig_gauge.update_layout(template="plotly_dark", height=250, margin=dict(t=40, b=0, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
            except Exception as e: st.error(f"オプションデータ処理エラー: {e}")

    # ==========================================
    # PAGE 2 & 3: Asset Class & Historical Analog (要約表示)
    # ==========================================
    elif page in ["2. Asset Class Macro (アセット別分析)", "3. Historical Analysis (過去比較)"]:
        st.info("GitHub移行用のテンプレートコードです。PAGE 2とPAGE 3のロジックは先ほどの完全版からコピペして利用可能です（文字数制限のため短縮しています）。")

    # ==========================================
    # PAGE 4: Institutional Quant Engine (Hybrid SOTA Version)
    # ==========================================
    elif page == "4. Investment Strategy (ハイブリッドAI戦略)":
        st.title("🧠 Regime-Conditioned Hybrid AI Strategy")
        st.markdown("【SOTAアップデート】GMM（混合ガウスモデル）で現在のマクロレジームを特定し、**現在と同じ相場環境の過去データのみ**を用いてアンサンブルAIを学習させます。")

        with st.sidebar.expander("⚙️ Model Architecture Settings", expanded=True):
            indicator_mode = st.radio("Macro Mode", ["Leading (先行指標特化)", "Full Macro (遅行指標含む)"])
            exclude_other_stocks = st.checkbox("Cross-Asset Exclusion (株価指数の除外)", value=True)
            include_anomaly = st.checkbox("Presidential Cycle (アノマリー追加)", value=False)
            lookback_years = st.slider("Lookback Window (学習期間)", 3, 10, 5)

        with st.spinner('Initializing GMM & Tri-Model Ensemble Engine...'):
            try:
                target_ticker = "SPY"
                yahoo_tickers = settings_df[settings_df['ソース'] == 'Yahoo']['ティッカー'].unique().tolist()
                market_implied_tickers = ['HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                all_yahoo = list(set(yahoo_tickers + market_implied_tickers + [target_ticker]))

                base_fred = ['ANFCI', 'STLFSI4', 'T10Y3M', 'BAMLH0A0HYM2', 'WALCL', 'RRPONTSYD', 'M2SL', 'ICSA', 'AWHAEMAN', 'T5YIFR']
                if "Full Macro" in indicator_mode: base_fred.extend(['CPIAUCSL', 'UNRATE', 'PAYEMS', 'INDPRO'])
                all_fred = list(set(settings_df[settings_df['ソース'] == 'FRED']['ティッカー'].unique().tolist() + base_fred))

                series_list = []
                y_data = fetch_market_data(all_yahoo, period=f"{lookback_years}y")
                for col in y_data.columns: series_list.append(y_data[col].dropna().rename(col))
                
                for tic in all_fred:
                    try: series_list.append(fred.get_series(tic).loc[f"{2024-lookback_years}-01-01":].dropna().rename(tic))
                    except: pass

                for i in range(len(series_list)): series_list[i].index = pd.to_datetime(series_list[i].index).tz_localize(None).normalize()
                df_ml = pd.concat(series_list, axis=1).ffill()
                
                # --- STEP 1: GMM Regime Detection (SOTA Hybrid) ---
                st.markdown("#### 🔬 Step 1: Latent Regime Detection (GMM)")
                gmm_feats = ['^VIX', 'T10Y3M', 'BAMLH0A0HYM2']
                for f in gmm_feats:
                    if f not in df_ml.columns: df_ml[f] = 0
                
                df_gmm = df_ml[gmm_feats].ffill().bfill()
                scaler_gmm = StandardScaler()
                X_gmm = scaler_gmm.fit_transform(df_gmm)
                
                gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
                df_ml['Regime'] = gmm.fit_predict(X_gmm)
                
                regime_means = df_ml.groupby('Regime')['^VIX'].mean().sort_values()
                regime_map = {regime_means.index[0]: '🟢 Normal (Risk-On)', 
                              regime_means.index[1]: '🟡 Transition (Caution)', 
                              regime_means.index[2]: '🔴 Crisis (Risk-Off)'}
                
                curr_regime_id = df_ml['Regime'].iloc[-1]
                curr_regime_name = regime_map[curr_regime_id]
                st.info(f"**Current Market Regime:** AIは現在の市場を **{curr_regime_name}** と判定しました。この環境下で機能する特徴量のみを抽出し、推論を行います。")

                # --- STEP 2: Feature Engineering ---
                df_ml['CTA_200D_Bias'] = df_ml[target_ticker] / df_ml[target_ticker].rolling(200).mean() - 1
                df_ml['Vol_Term_Spread'] = df_ml['^VIX'] / df_ml['^VIX3M']
                df_ml['Copper_Gold_Ratio'] = df_ml['HG=F'] / df_ml['GC=F']
                df_ml['Presidential_Cycle'] = df_ml.index.year % 4
                df_ml['Target'] = df_ml[target_ticker].pct_change(21).shift(-21)

                drop_cols = ['Target', 'Regime', 'HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                if exclude_other_stocks: drop_cols.extend([t for t in yahoo_tickers if t != target_ticker])
                if not include_anomaly: drop_cols.append('Presidential_Cycle')

                features = [col for col in df_ml.columns if col not in drop_cols]
                latest_x = df_ml[features].iloc[-1:]
                
                # --- STEP 3: Regime-Conditioned Data Filtering ---
                df_train_full = df_ml.dropna(subset=['Target'] + features)
                df_train = df_train_full[df_train_full['Regime'] == curr_regime_id]
                
                if len(df_train) < 50:
                    st.warning("現在のレジームに該当する学習データが少なすぎるため、全期間データで学習をフォールバックします。")
                    df_train = df_train_full

                X, y = df_train[features], df_train['Target']
                
                # --- STEP 4: Tri-Model Ensemble Training ---
                st.markdown(f"#### 🧠 Step 2: Tri-Model Ensemble Engine (Trained on {len(df_train)} contextual days)")
                weights = np.exp(np.linspace(-1, 0, len(X)))
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                latest_x_scaled = scaler.transform(latest_x)

                rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
                rf_model.fit(X, y, sample_weight=weights)
                
                gb_model = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
                gb_model.fit(X, y, sample_weight=weights)
                
                en_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
                en_model.fit(X_scaled, y, sample_weight=weights)
                
                pred_rf = rf_model.predict(latest_x)[0] * 100
                pred_gb = gb_model.predict(latest_x)[0] * 100
                pred_en = en_model.predict(latest_x_scaled)[0] * 100
                
                pred_ret = np.mean([pred_rf, pred_gb, pred_en])
                pred_std = np.std([pred_rf, pred_gb, pred_en])
                ci_lower = pred_ret - (1.96 * pred_std)
                ci_upper = pred_ret + (1.96 * pred_std)

                # --- STEP 5: Rendering Results ---
                c1, c2, c3 = st.columns(3)
                color_ret = "#3fb950" if pred_ret > 0 else "#f85149"
                c1.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Ensemble 1M Expected</div>
                    <div class='kpi-value' style='color:{color_ret}'>{pred_ret:+.2f}%</div>
                    <div class='model-breakdown'><span>RF: {pred_rf:+.1f}%</span><span>GB: {pred_gb:+.1f}%</span><span>EN: {pred_en:+.1f}%</span></div>
                </div>""", unsafe_allow_html=True)
                
                vix_dec = df_ml['^VIX'].iloc[-1] / 100
                market_var = vix_dec ** 2
                kelly_f = (pred_ret / 100) / market_var if market_var > 0 else 0
                optimal_weight = max(0, min(100, kelly_f * 100 * 0.5))
                
                c2.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Target Exposure (Kelly)</div>
                    <div class='kpi-value' style='color:#e3b341'>{optimal_weight:.0f}%</div>
                    <div class='kpi-sub'>Cash Recommendation: {100-optimal_weight:.0f}%</div>
                </div>""", unsafe_allow_html=True)
                
                conf_score = max(0, 100 - (pred_std * 15))
                c3.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Model Consensus</div>
                    <div class='kpi-value' style='color:#a371f7'>{conf_score:.1f}/100</div>
                    <div class='kpi-sub'>Agreement across engines</div>
                </div>""", unsafe_allow_html=True)

                # 特徴量の重要度抽出 (Regimeに依存)
                combined_imp = (rf_model.feature_importances_ + gb_model.feature_importances_) / 2
                imp_df = pd.DataFrame({'Feature': features, 'Importance': combined_imp})
                
                df_3y = df_ml.tail(252*3)
                macro_only = [f for f in features if f != 'Presidential_Cycle']
                z_scores = (latest_x[macro_only].iloc[0] - df_3y[macro_only].mean()) / df_3y[macro_only].std()
                imp_df['Z_Score'] = imp_df['Feature'].map(z_scores).fillna(0)
                imp_df = imp_df.sort_values('Importance', ascending=False).head(10)

                st.markdown("---")
                st.markdown(f"#### 🔍 Key Drivers under {curr_regime_name} Regime")
                fig_brain = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h', color='Z_Score', color_continuous_scale='RdBu_r', range_color=[-3, 3], template="plotly_dark")
                fig_brain.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), coloraxis_colorbar=dict(title="Z-Score"))
                st.plotly_chart(fig_brain, use_container_width=True)
                
                with st.expander("📝 Generate Quantitative Report Prompt (LLM用プロンプト)", expanded=False):
                    top_features_text = "".join([f"- {row['Feature']}: 重要度 {row['Importance']:.4f}, 現在のZ-Score {row['Z_Score']:+.2f}\n" for _, row in imp_df.head(8).iterrows()])
                    llm_prompt = f"""あなたはトップ・クオンツファンドのシニア・ポートフォリオマネージャーです。以下のデータに基づき、投資委員会向けの市場解説およびアロケーション戦略レポートを作成してください。

【GMM Regime Detection】
・現在の市場レジーム: {curr_regime_name}

【アンサンブルAI予測メトリクス (ホライズン: 1ヶ月)】
・統合予測リターン: {pred_ret:+.2f}% 
・モデルコンセンサス度: {conf_score:.1f}/100
・最適ポジション露出度 (ハーフ・ケリー基準): {optimal_weight:.0f}%

【AI決定要因: 当該レジーム下における上位特徴量とZ-Score異常値】
{top_features_text}
"""
                    st.code(llm_prompt, language="text")

            except Exception as e: st.error(f"分析モデル・エラー: {e}")

    # ==========================================
    # PAGE 5 ~ 8 (要約表示・コード省略)
    # ==========================================
    elif page in ["5. Headline Reverse-Engineering (イベント逆引き)", "6. Portfolio Optimization (アロケーション)", "7. Macro Data Explorer (マクロ生データ確認)", "8. Hybrid AI Regime Strategy (SOTAモデル)"]:
        st.info("GitHub移行用のテンプレートコードです。これらのページのロジックも以前のコードからそのまま貼り付けてご利用いただけます。")

# ==========================================
# グローバル・エラーハンドリング
# ==========================================
except Exception as e:
    st.error(f"System Critical Error: {e}")
