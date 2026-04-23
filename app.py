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
        
@st.cache_data(ttl=1800) # 30分間キャッシュしてYahooのブロックを回避
def get_options_max_pain(ticker_symbol):
    try:
        tkr = yf.Ticker(ticker_symbol)
        opt_dates = tkr.options
        if len(opt_dates) == 0:
            return None, None, None
            
        exp = opt_dates[0]
        opt_chain = tkr.option_chain(exp)
        c, p = opt_chain.calls, opt_chain.puts
        
        strikes = sorted(list(set(c['strike']).union(set(p['strike']))))
        mp, min_l = 0, float('inf')
        for s in strikes:
            l = c[c['strike']<s].apply(lambda x:(s-x['strike'])*x['openInterest'], axis=1).sum() + p[p['strike']>s].apply(lambda x:(x['strike']-s)*x['openInterest'], axis=1).sum()
            if l < min_l: min_l, mp = l, s
            
        curr_price = tkr.history(period="1d")['Close'].iloc[-1]
        return exp, mp, curr_price
    except Exception as e:
        raise e

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
                # キャッシュ化された安全な関数を呼び出す
                exp, mp, curr_price = get_options_max_pain(target_ticker)
                
                if exp is None:
                    st.warning("現在、Yahoo Financeからオプションデータが制限されています。")
                else:
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
            
            except Exception as e: 
                st.error(f"現在Yahoo APIのアクセス制限(Rate Limit)を受けています。")
                st.info("💡 約30分〜1時間ほど放置すると自動的に制限が解除されます。")
                
    # ==========================================
    # PAGE 2: Institutional Asset Class Macro
    # ==========================================
    elif page == "2. Asset Class Macro (アセット別分析)":
        st.title("🏦 Institutional Asset Class Macro")
        st.markdown("ローリング相関、Z-Score標準化、およびマクロ乖離シグナルを用いて、グローバルアセットの根源的な力学を解析します。")

        st.subheader("1. Cross-Asset Regime Monitor (Rolling Correlation)")
        with st.spinner("Calculating Rolling Correlations..."):
            try:
                assets = {'SPY':'Stocks (S&P500)', 'TLT':'Bonds (20Y+)', 'GLD':'Gold', 'USO':'Oil', 'UUP':'US Dollar'}
                close_data = fetch_market_data(list(assets.keys()), period="2y").rename(columns=assets)
                
                roll_corr_bond = close_data['Stocks (S&P500)'].rolling(60).corr(close_data['Bonds (20Y+)'])
                roll_corr_usd = close_data['Stocks (S&P500)'].rolling(60).corr(close_data['US Dollar'])
                roll_corr_gold = close_data['US Dollar'].rolling(60).corr(close_data['Gold'])

                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=roll_corr_bond.index, y=roll_corr_bond, name='Stocks vs Bonds (Risk On/Off)', line=dict(color='#58a6ff')))
                fig_roll.add_trace(go.Scatter(x=roll_corr_usd.index, y=roll_corr_usd, name='Stocks vs USD (Liquidity)', line=dict(color='#3fb950')))
                fig_roll.add_trace(go.Scatter(x=roll_corr_gold.index, y=roll_corr_gold, name='USD vs Gold (Real Yield)', line=dict(color='#e3b341')))
                
                fig_roll.add_hline(y=0, line_dash="dash", line_color="#8b949e")
                fig_roll.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0), hovermode="x unified")
                st.plotly_chart(fig_roll, use_container_width=True)
                
                curr_sb_corr = roll_corr_bond.iloc[-1]
                if curr_sb_corr > 0.5:
                    st.warning("⚠️ **Regime Alert:** 株と債券が強い正の相関状態です。伝統的な分散効果が機能していません。")
                elif curr_sb_corr < -0.5:
                    st.success("✅ **Normal Regime:** 株と債券が逆相関です。リスクオフ時に債券がクッションとして機能します。")
            except Exception as e: st.warning("相関データの取得をスキップしました。")

        st.markdown("---")
        st.subheader("2. Macro Fundamentals Z-Score Dashboard")
        
        tabs_names = [t for t in settings_df['タブ名'].unique() if "ダッシュボード" not in t]
        if tabs_names:
            tabs = st.tabs(tabs_names)
            for i, t_name in enumerate(tabs_names):
                with tabs[i]:
                    t_df = settings_df[settings_df['タブ名'] == t_name]
                    cols = st.columns(2)
                    
                    for g_idx, g_name in enumerate(t_df['グラフ名'].unique()):
                        with cols[g_idx % 2]:
                            g_data = t_df[t_df['グラフ名'] == g_name]
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            latest_z_scores = []
                            max_dt = None
                            
                            if len(g_data) == 1 and g_data.iloc[0]['ソース'] == 'CTA':
                                pass 
                            elif len(g_data) == 1 and g_data.iloc[0]['ソース'] == 'Options':
                                pass 
                            else:
                                for _, r in g_data.iterrows():
                                    try:
                                        d = None
                                        if r['ソース'] == 'FRED': d = fred.get_series(r['ティッカー']).loc['2020-01-01':]
                                        elif r['ソース'] == 'Yahoo': d = fetch_market_data([r['ティッカー']], start='2020-01-01')[r['ティッカー']]
                                        elif r['ソース'] == 'DBnomics': 
                                            db_df = fetch_series(r['ティッカー'])
                                            if not db_df.empty: d = db_df[['period', 'value']].dropna().set_index('period')['value']
                                        
                                        if d is not None and not d.empty:
                                            d.index = pd.to_datetime(d.index).tz_localize(None)
                                            if max_dt is None or d.index.max() > max_dt: max_dt = d.index.max()
                                            
                                            fig.add_trace(go.Scatter(x=d.index, y=d.values, name=r['データ名']), secondary_y=(r['軸']=='副軸'))
                                            
                                            d_3y = d.last('3Y')
                                            if len(d_3y) > 30: 
                                                z = (d.iloc[-1] - d_3y.mean()) / d_3y.std()
                                                latest_z_scores.append((r['データ名'], z))
                                    except: pass
                                
                                if max_dt: fig.update_xaxes(range=[max_dt - pd.DateOffset(years=2), max_dt + pd.DateOffset(days=10)])
                                fig.update_layout(title=g_name, height=300, template="plotly_dark", hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0))
                            
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                if latest_z_scores:
                                    z_html = "<div style='font-size:13px; text-align:right; margin-top:-15px; margin-bottom:15px;'>"
                                    for name, z in latest_z_scores:
                                        if z > 1.5: z_class = "zscore-hot"
                                        elif z < -1.5: z_class = "zscore-cold"
                                        else: z_class = "zscore-neutral"
                                        z_html += f"<span class='{z_class}'>{name} Z: {z:+.1f}σ</span> &nbsp;&nbsp;"
                                    z_html += "</div>"
                                    st.markdown(z_html, unsafe_allow_html=True)
    # ==========================================
    # PAGE 3: Institutional Historical Analog
    # ==========================================
    elif page == "3. Historical Analysis (過去比較)":
        st.title("🕰️ Historical Analog & Regime Projection")
        st.markdown("現在の市場軌跡（T-N日）と過去の金融危機の軌跡を統計的にマッチングし、今後の「ゴーストパス（未来の軌跡）」を投影します。")

        c_set1, c_set2, c_set3, c_set4 = st.columns(4)
        target_idx = c_set1.selectbox("🎯 分析対象", ["^GSPC (S&P 500)", "^NDX (Nasdaq 100)", "^TNX (10年金利)"])
        ticker = target_idx.split(" ")[0]
        
        regimes = {
            "2022 Inflation Shock": "2022-01-03",
            "2020 COVID Crash": "2020-02-19",
            "2018 Volmageddon": "2018-01-26",
            "2008 Lehman Brothers": "2008-09-15",
            "2000 Dot-com Bubble": "2000-03-24",
            "1987 Black Monday": "1987-10-19"
        }
        past_ev_name = c_set2.selectbox("📜 比較する過去のレジーム", list(regimes.keys()))
        past_date_str = regimes[past_ev_name]
        
        curr_date = pd.to_datetime(c_set3.date_input("📍 現在の起点 (T=0)", pd.to_datetime("today") - pd.offsets.BDay(1))).tz_localize(None)
        lookback = c_set4.slider("過去の照合期間 (T-日数)", 20, 120, 60)
        lookforward = 120 

        with st.spinner('Historical Analog Data processing...'):
            try:
                data = fetch_market_data([ticker, "^VIX"], start="1980-01-01")
                data.index = pd.to_datetime(data.index).tz_localize(None)
                
                p_idx = data.index.get_indexer([pd.to_datetime(past_date_str)], method='nearest')[0]
                c_idx = data.index.get_indexer([curr_date], method='nearest')[0]
                
                p_slice = data.iloc[p_idx - lookback : p_idx + lookforward + 1].copy()
                p_slice['T'] = np.arange(-lookback, len(p_slice) - lookback)
                
                c_slice = data.iloc[c_idx - lookback : c_idx + 1].copy()
                c_slice['T'] = np.arange(-lookback, 1)
                
                p_base_price = p_slice[ticker].loc[p_slice['T'] == 0].values[0]
                c_base_price = c_slice[ticker].loc[c_slice['T'] == 0].values[0]
                
                p_slice['Price_Norm'] = (p_slice[ticker] / p_base_price) * 100
                c_slice['Price_Norm'] = (c_slice[ticker] / c_base_price) * 100

                p_compare = p_slice[p_slice['T'] <= 0]['Price_Norm'].values
                c_compare = c_slice['Price_Norm'].values
                min_len = min(len(p_compare), len(c_compare))
                corr = np.corrcoef(p_compare[-min_len:], c_compare[-min_len:])[0, 1]
                
                st.markdown("---")
                k1, k2, k3 = st.columns(3)
                
                corr_color = "#3fb950" if corr > 0.8 else "#e3b341" if corr > 0.5 else "#f85149"
                k1.markdown(f"""<div class='metric-box'>
                    <h4 style='color:#8b949e; margin-bottom:5px;'>Statistical Similarity (シンクロ率)</h4>
                    <h2 style='color:{corr_color}; margin-top:0px;'>{corr*100:.1f}%</h2>
                </div>""", unsafe_allow_html=True)
                
                p_drawdown = (p_slice['Price_Norm'].min() - 100)
                k2.markdown(f"""<div class='metric-box'>
                    <h4 style='color:#8b949e; margin-bottom:5px;'>Historical Drawdown (T+以降)</h4>
                    <h2 style='color:#f85149; margin-top:0px;'>{p_drawdown:.1f}%</h2>
                </div>""", unsafe_allow_html=True)
                
                c_ret = ((c_slice[ticker].iloc[-1] / c_slice[ticker].iloc[0]) - 1) * 100
                k3.markdown(f"""<div class='metric-box'>
                    <h4 style='color:#8b949e; margin-bottom:5px;'>Current Momentum</h4>
                    <h2 style='color:{"#3fb950" if c_ret>0 else "#f85149"}; margin-top:0px;'>{c_ret:+.1f}%</h2>
                </div>""", unsafe_allow_html=True)

                fig_analog = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3], subplot_titles=("Price Analog", "Volatility Regime (VIX)"))
                
                p_past = p_slice[p_slice['T'] <= 0]
                p_future = p_slice[p_slice['T'] >= 0]
                
                fig_analog.add_trace(go.Scatter(x=p_past['T'], y=p_past['Price_Norm'], name=f"{past_ev_name} (Up to T=0)", line=dict(color='gray', width=2)), row=1, col=1)
                fig_analog.add_trace(go.Scatter(x=p_future['T'], y=p_future['Price_Norm'], name=f"Ghost Path (Future Projection)", line=dict(color='gray', width=2, dash='dot')), row=1, col=1)
                fig_analog.add_trace(go.Scatter(x=c_slice['T'], y=c_slice['Price_Norm'], name="Current Market", line=dict(color='#ff4b4b', width=3)), row=1, col=1)

                if "^VIX" in p_slice.columns and "^VIX" in c_slice.columns:
                    fig_analog.add_trace(go.Scatter(x=p_slice['T'], y=p_slice['^VIX'], name=f"VIX: {past_ev_name}", line=dict(color='rgba(128,128,128,0.5)', width=1)), row=2, col=1)
                    fig_analog.add_trace(go.Scatter(x=c_slice['T'], y=c_slice['^VIX'], name="VIX: Current", line=dict(color='#58a6ff', width=2)), row=2, col=1)

                fig_analog.add_vline(x=0, line_dash="solid", line_color="#8b949e", line_width=1, row='all', col=1)
                fig_analog.update_layout(template="plotly_dark", height=600, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_analog, use_container_width=True)

            except Exception as e:
                st.error(f"アナログ分析データの処理中にエラーが発生しました: {e}")
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
    # PAGE 5: Headline Reverse-Engineering
    # ==========================================
    elif page == "5. Headline Reverse-Engineering (イベント逆引き)":
        st.title("🗞️ Volatility Anomaly & Reverse-Engineering")
        st.markdown("SPYの下落とVIXの異常な急騰（クラッシュ・シグナル）が同時に発生した「真の金融ショック日」を特定し、背景のニュースを逆引きします。")

        lookback_months = st.slider("分析期間 (ヶ月)", 1, 24, 12)
        
        with st.spinner('市場のクラッシュ日を検出中...'):
            try:
                market_data = fetch_market_data(["SPY", "^VIX"], period=f"{lookback_months}mo")
                
                spy_data = market_data['SPY']
                vix_data = market_data['^VIX']
                
                spy_ret = spy_data.pct_change().dropna() * 100
                vix_ret = vix_data.pct_change().dropna() * 100
                
                vix_z = (vix_ret - vix_ret.mean()) / vix_ret.std()
                
                crash_days = spy_ret[(spy_ret < 0) & (vix_z > 1.5)]
                
                if len(crash_days) == 0:
                    st.success("指定された期間内に、強烈なクラッシュ（異常変動）は検出されませんでした。")
                else:
                    top_moves = crash_days.nsmallest(min(5, len(crash_days))).index
                    
                    st.subheader("📊 統計的クラッシュ日の検出 (VIX Spike + SPY Drop)")
                    fig_events = go.Figure()
                    fig_events.add_trace(go.Scatter(x=spy_data.index, y=spy_data.values, name="SPY Price", line=dict(color='#8b949e', width=2)))
                    
                    for date in top_moves:
                        ret_val = spy_ret.loc[date]
                        vix_spike = vix_ret.loc[date]
                        fig_events.add_trace(go.Scatter(
                            x=[date], y=[spy_data.loc[date]], mode='markers+text',
                            marker=dict(color='#f85149', size=14, symbol='star'),
                            text=[f"SPY: {ret_val:.1f}%<br>VIX: +{vix_spike:.1f}%"], textposition="top center", name=f"{date.strftime('%Y-%m-%d')}"
                        ))
                    fig_events.update_layout(template="plotly_dark", height=400, showlegend=False)
                    st.plotly_chart(fig_events, use_container_width=True)

                    st.markdown("---")
                    with st.expander("🤖 ニュース逆引きプロンプトの生成", expanded=True):
                        events_text = "".join([f"- {d.strftime('%Y-%m-%d')}: SPY {spy_ret.loc[d]:+.2f}%, VIX {vix_ret.loc[d]:+.2f}%\n" for d in top_moves])
                        llm_event_prompt = f"""あなたはマクロ経済の歴史に精通したアナリストです。
以下の{len(top_moves)}つの日付は、S&P500が下落し、同時にVIX（恐怖指数）が統計的に異常な急騰を見せた「金融ショック日」です。
そのトリガーとなったニュースや経済指標発表をウェブ検索等で特定し、解説してください。
【クラッシュ発生日リスト】
{events_text}"""
                        st.code(llm_event_prompt, language="text")
            except Exception as e: 
                st.error(f"イベント特定エラー: {e}")
    # ==========================================
    # PAGE 7: Macro Data Explorer
    # ==========================================
    elif page == "7. Macro Data Explorer (マクロ生データ確認)":
        st.title("🗄️ Macro Data Explorer")
        st.markdown("FREDおよびYahoo FinanceからAPI経由で取得・結合したマクロ指標の生データと相関構造を確認します。")
        
        with st.spinner("Fetching underlying macro dataset..."):
            try:
                base_fred = ['ANFCI', 'T10Y3M', 'BAMLH0A0HYM2', 'WALCL', 'UMCSENT']
                fred_data = []
                for tic in base_fred:
                    try: fred_data.append(fred.get_series(tic).loc["2015-01-01":].rename(tic))
                    except: pass
                
                yahoo_tickers = ['SPY', '^VIX', 'HG=F', 'GC=F']
                yahoo_data = fetch_market_data(yahoo_tickers, start="2015-01-01")
                
                df_fred = pd.concat(fred_data, axis=1) if fred_data else pd.DataFrame()
                df_raw = pd.concat([yahoo_data, df_fred], axis=1).ffill().dropna()
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader("Raw Data Table (Downloadable)")
                    st.dataframe(df_raw.sort_index(ascending=False), height=400)
                
                with c2:
                    st.subheader("Cross-Indicator Correlation Matrix")
                    corr = df_raw.corr()
                    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto", template="plotly_dark")
                    fig_corr.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"データエクスプローラーの読み込みエラー: {e}")

    # ==========================================
    # PAGE 8: Model-First Hybrid AI (GMM Regime Mapping)
    # ==========================================
    elif page == "8. Hybrid AI Regime Strategy (SOTAモデル)":
        st.title("🔬 Market Regime Map (GMM Latent States)")
        st.markdown("S&P500の過去の軌跡に対し、GMMが事後的に割り当てた「隠れた相場状態」をマッピングします。")

        with st.spinner("Training Gaussian Mixture Model for Regime Detection..."):
            try:
                spy = fetch_market_data(["SPY", "^VIX"], start="2010-01-01")
                fred_series = []
                for tic in ['T10Y3M', 'BAMLH0A0HYM2']: 
                    try: fred_series.append(fred.get_series(tic).loc["2010-01-01":].rename(tic))
                    except: pass
                
                df_hmm = pd.concat([spy, pd.concat(fred_series, axis=1)], axis=1).ffill().dropna()
                
                features_gmm = ['^VIX', 'T10Y3M', 'BAMLH0A0HYM2']
                scaler_gmm = StandardScaler()
                X_gmm = scaler_gmm.fit_transform(df_hmm[features_gmm])
                
                gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
                df_hmm['Regime'] = gmm.fit_predict(X_gmm)
                
                regime_means = df_hmm.groupby('Regime')['^VIX'].mean().sort_values()
                regime_map = {regime_means.index[0]: 'Normal (Risk-On)', 
                              regime_means.index[1]: 'Transition (Caution)', 
                              regime_means.index[2]: 'Crisis (Risk-Off)'}
                df_hmm['Regime_Name'] = df_hmm['Regime'].map(regime_map)

                st.subheader("Historical Regime Map (S&P 500)")
                fig_regime = go.Figure()
                fig_regime.add_trace(go.Scatter(x=df_hmm.index, y=df_hmm['SPY'], mode='lines', line=dict(color='#c9d1d9', width=1), name='SPY Price'))
                
                colors = {'Normal (Risk-On)': 'rgba(63, 185, 80, 0.5)', 'Transition (Caution)': 'rgba(227, 179, 65, 0.5)', 'Crisis (Risk-Off)': 'rgba(248, 81, 73, 0.8)'}
                
                df_plot = df_hmm.tail(252*5)
                for reg in colors.keys():
                    mask = df_plot['Regime_Name'] == reg
                    if mask.any():
                        fig_regime.add_trace(go.Scatter(x=df_plot.index[mask], y=df_plot['SPY'][mask], mode='markers', 
                                                        marker=dict(color=colors[reg], size=5), name=reg))

                fig_regime.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_regime, use_container_width=True)
                
            except Exception as e:
                st.error(f"レジームマップの描画エラー: {e}")

# ==========================================
# グローバル・エラーハンドリング
# ==========================================
except Exception as e:
    st.error(f"System Critical Error: {e}")
