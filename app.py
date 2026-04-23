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
# サイドバーのメニュー項目を全ページ分に拡張します
    st.sidebar.title("💎 Macro Navigation")
    page = st.sidebar.radio("機能を選択", [
        "1. Market Dynamics (現在)", 
        "2. Asset Class Macro (アセット別分析)", 
        "3. Historical Analysis (過去比較)",
        "4. Investment Strategy (AI戦略)",
        "5. Headline Reverse-Engineering (イベント逆引き)", # これを追加
        "6. Portfolio Optimization (アロケーション)"     # これを追加
    ])

    # ==========================================
    # PAGE 1: Institutional Market Dynamics
    # ==========================================
    if page == "1. Market Dynamics (現在)":
        st.markdown("""
        <style>
        .metric-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
        .alert-box-red { background-color: #2d1111; border-left: 5px solid #f85149; padding: 15px; border-radius: 5px; }
        .alert-box-green { background-color: #112d11; border-left: 5px solid #3fb950; padding: 15px; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)

        st.title("📈 Institutional Market Dynamics & Flows")
        st.markdown("ボラティリティ構造、セクター・ローテーション、およびオプション需給から、足元の市場の「歪み」と「資金の逃避先」を特定します。")

        # --- 1. VIX Term Structure (恐怖の期間構造) ---
        st.subheader("1. Volatility Term Structure (恐怖の構造とイールドカーブ)")
        with st.spinner("Analyzing Volatility Surface..."):
            try:
                # VIX関連指数の取得 (9日, 30日, 3ヶ月, 6ヶ月)
                vix_tickers = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]
                vix_data = yf.download(vix_tickers, period="5d", progress=False)['Close'].iloc[-1]
                
                c_vix1, c_vix2 = st.columns([1.5, 1])
                
                with c_vix1:
                    # 期間構造のカーブ描画
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
                    if vix_values[1] > vix_values[2]: # VIX(1M) > VIX(3M)
                        st.markdown("<div class='alert-box-red'><h4>⚠️ バックワーデーション (Panic)</h4>短期的な恐怖が中期を上回っています。市場はプットオプションをパニック買いしており、暴落の警戒レベルが最大です。現金比率を高めてください。</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='alert-box-green'><h4>✅ コンタンゴ (Normal)</h4>恐怖の構造は正常（右肩上がり）です。構造的なパニックは起きておらず、押し目買いが機能しやすい環境です。</div>", unsafe_allow_html=True)
            except Exception as e: st.warning("VIXデータの取得に失敗しました。")

        st.markdown("---")

        # --- 2. Sector Rotation (Quadrant Analysis) ---
        st.subheader("2. Sector Rotation (資金循環のクアドラント分析)")
        st.markdown("過去1週間（短期）と過去1ヶ月（中期）のリターンを比較し、資金がどのセクターに逃げているかを特定します。")
        with st.spinner("Calculating Sector Momentum..."):
            try:
                sectors = {'XLK':'Technology', 'XLF':'Financials', 'XLV':'Health Care', 'XLY':'Consumer Disc', 
                           'XLC':'Communication', 'XLI':'Industrials', 'XLP':'Consumer Staples', 'XLE':'Energy', 
                           'XLU':'Utilities', 'XLRE':'Real Estate', 'XLB':'Materials'}
                
                # 1ヶ月分のデータを取得
                sec_df = yf.download(list(sectors.keys()), period="1mo", progress=False)['Close']
                
                rot_data = []
                for tic, name in sectors.items():
                    ret_1w = ((sec_df[tic].iloc[-1] / sec_df[tic].iloc[-6]) - 1) * 100 # 約1週間(5営業日)
                    ret_1m = ((sec_df[tic].iloc[-1] / sec_df[tic].iloc[0]) - 1) * 100  # 約1ヶ月(21営業日)
                    rot_data.append({"Sector": name, "1W_Return": ret_1w, "1M_Return": ret_1m})
                    
                df_rot = pd.DataFrame(rot_data)
                
                # 散布図 (Quadrant) の作成
                fig_rot = px.scatter(df_rot, x='1W_Return', y='1M_Return', text='Sector', 
                                     color='1M_Return', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
                
                fig_rot.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='White')))
                # 十字線を追加
                fig_rot.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
                fig_rot.add_vline(x=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
                
                # クアドラントの注釈
                fig_rot.add_annotation(x=df_rot['1W_Return'].max(), y=df_rot['1M_Return'].max(), text="🔥 リーダー (強い)", showarrow=False, opacity=0.5)
                fig_rot.add_annotation(x=df_rot['1W_Return'].min(), y=df_rot['1M_Return'].min(), text="🧊 出遅れ/弱気", showarrow=False, opacity=0.5)
                fig_rot.add_annotation(x=df_rot['1W_Return'].max(), y=df_rot['1M_Return'].min(), text="🚀 反発開始 (注目)", showarrow=False, opacity=0.5)
                
                fig_rot.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Short-term Momentum (1-Week %)", yaxis_title="Medium-term Trend (1-Month %)")
                st.plotly_chart(fig_rot, use_container_width=True)
            except Exception as e: st.warning(f"セクターデータの取得に失敗しました: {e}")

        st.markdown("---")

        # --- 3. Positioning & Options Wall ---
        st.subheader("3. Institutional Positioning & Options Wall")
        c_cta, c_opt = st.columns(2)
        
        with c_cta:
            st.markdown("#### 🤖 S&P500 CTA Trend Proxy")
            try:
                spy = yf.Ticker("SPY").history(period="1y")
                spy.index = pd.to_datetime(spy.index).tz_localize(None)
                spy['SMA50'] = spy['Close'].rolling(50).mean()
                spy['SMA200'] = spy['Close'].rolling(200).mean()
                curr = spy['Close'].iloc[-1]
                dist_200 = ((curr - spy['SMA200'].iloc[-1]) / spy['SMA200'].iloc[-1]) * 100
                
                fig_cta = go.Figure()
                fig_cta.add_trace(go.Scatter(x=spy.index, y=spy['Close'], name='SPY Price', line=dict(color='#c9d1d9')))
                fig_cta.add_trace(go.Scatter(x=spy.index, y=spy['SMA50'], name='50 SMA', line=dict(color='#58a6ff', dash='dot')))
                fig_cta.add_trace(go.Scatter(x=spy.index, y=spy['SMA200'], name='200 SMA', line=dict(color='#ff4b4b', dash='dash')))
                
                # 50SMAと200SMAのクロスエリアを塗りつぶし (モメンタムの強さ)
                fig_cta.add_trace(go.Scatter(x=spy.index, y=np.where(spy['SMA50']>spy['SMA200'], spy['SMA50'], spy['SMA200']), fill='tonexty', fillcolor='rgba(63,185,80,0.1)', line=dict(width=0), showlegend=False))
                
                fig_cta.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified")
                st.plotly_chart(fig_cta, use_container_width=True)
                st.info(f"**CTA Bias:** 価格は200日線から `{dist_200:+.1f}%` 乖離。トレンドフォロワーは現在 **{'強気 (Long)' if dist_200 > 0 else '弱気 (Short)'}** ポジションに偏っています。")
            except: pass

        with c_opt:
            st.markdown("#### 🎯 SPY Options Max Pain Magnet")
            try:
                spy_opt = yf.Ticker("SPY")
                exp = spy_opt.options[0] # 直近の満期日
                c, p = spy_opt.option_chain(exp).calls, spy_opt.option_chain(exp).puts
                
                strikes = sorted(list(set(c['strike']).union(set(p['strike']))))
                mp, min_l = 0, float('inf')
                for s in strikes:
                    l = c[c['strike']<s].apply(lambda x:(s-x['strike'])*x['openInterest'], axis=1).sum() + p[p['strike']>s].apply(lambda x:(x['strike']-s)*x['openInterest'], axis=1).sum()
                    if l < min_l: min_l, mp = l, s
                    
                curr_price = spy['Close'].iloc[-1]
                diff_mp = ((curr_price - mp) / mp) * 100
                
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
                st.write(f"市場のディーラーが最も利益を得る（＝オプションの買い手が最も損をする）価格帯（黄色い線）です。満期に向けてこの価格に引き寄せられるマグネット効果に注意してください。")
            except Exception as e: st.write("Options data unavailable")

    # ==========================================
    # PAGE 2: Institutional Asset Class Macro
    # ==========================================
    elif page == "2. Asset Class Macro (アセット別分析)":
        st.markdown("""
        <style>
        .zscore-hot { color: #f85149; font-weight: bold; }
        .zscore-cold { color: #58a6ff; font-weight: bold; }
        .zscore-neutral { color: #8b949e; }
        </style>
        """, unsafe_allow_html=True)

        st.title("🏦 Institutional Asset Class Macro")
        st.markdown("ローリング相関、Z-Score標準化、およびマクロ乖離シグナルを用いて、グローバルアセットの根源的な力学を解析します。")

        # --- 1. Cross-Asset Rolling Correlation (動的相関の崩壊検知) ---
        st.subheader("1. Cross-Asset Regime Monitor (Rolling Correlation)")
        with st.spinner("Calculating Rolling Correlations..."):
            try:
                assets = {'SPY':'Stocks (S&P500)', 'TLT':'Bonds (20Y+)', 'GLD':'Gold', 'USO':'Oil', 'UUP':'US Dollar'}
                close_data = yf.download(list(assets.keys()), period="2y", progress=False)['Close'].rename(columns=assets)
                
                # 60日ローリング相関の計算
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
                
                # 相関の異常検知インサイト
                curr_sb_corr = roll_corr_bond.iloc[-1]
                if curr_sb_corr > 0.5:
                    st.warning("⚠️ **Regime Alert:** 株と債券が強い正の相関（同時に売買されている）状態です。インフレ主導の相場であり、伝統的な60/40ポートフォリオの分散効果が機能していません。")
                elif curr_sb_corr < -0.5:
                    st.success("✅ **Normal Regime:** 株と債券が逆相関です。リスクオフ時に債券がクッションとして機能する健全な環境です。")
            except Exception as e: st.warning("相関データの取得をスキップしました。")

        st.markdown("---")

        # --- 2. Macro Indicator Z-Score Normalization (異常値のフラット比較) ---
        st.subheader("2. Macro Fundamentals Z-Score Dashboard")
        st.markdown("各アセットの先行指標が、過去3年の平均からどれだけ逸脱しているか（標準偏差）を可視化します。")
        
        tabs_names = [t for t in settings_df['タブ名'].unique() if "ダッシュボード" not in t]
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
                        
                        # 特殊グラフ（CTA/Options）の処理
                        if len(g_data) == 1 and g_data.iloc[0]['ソース'] == 'CTA':
                            r = g_data.iloc[0]
                            try:
                                d = yf.Ticker(r['ティッカー']).history(period="2y")
                                d.index = pd.to_datetime(d.index).tz_localize(None)
                                sma200 = d['Close'].rolling(200).mean()
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=d.index, y=d['Close'], name='Price', line=dict(color='#c9d1d9')))
                                fig.add_trace(go.Scatter(x=d.index, y=sma200, name='200 SMA', line=dict(color='#ff4b4b', dash='dot')))
                                fig.update_layout(title=f"🤖 {g_name} (CTA Trend)", height=300, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                            except: pass
                            
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

                        # 通常のマクロ指標の処理 (Z-Score計算付き)
                        else:
                            for _, r in g_data.iterrows():
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
                                        
                                        # グラフ描画
                                        fig.add_trace(go.Scatter(x=d.index, y=d.values, name=r['データ名']), secondary_y=(r['軸']=='副軸'))
                                        
                                        # Z-Scoreの計算 (直近3年ベース)
                                        d_3y = d.last('3Y')
                                        if len(d_3y) > 30: # データが十分にある場合のみ
                                            z = (d.iloc[-1] - d_3y.mean()) / d_3y.std()
                                            latest_z_scores.append((r['データ名'], z))
                                except: pass
                            
                            if max_dt: fig.update_xaxes(range=[max_dt - pd.DateOffset(years=2), max_dt + pd.DateOffset(days=10)])
                            fig.update_layout(title=g_name, height=300, template="plotly_dark", hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0))
                        
                        # グラフとZ-Score異常値の表示
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Z-Scoreに基づくマクロ乖離シグナルの生成
                            if latest_z_scores:
                                z_html = "<div style='font-size:13px; text-align:right; margin-top:-15px; margin-bottom:15px;'>"
                                for name, z in latest_z_scores:
                                    if z > 1.5: z_class = "zscore-hot"
                                    elif z < -1.5: z_class = "zscore-cold"
                                    else: z_class = "zscore-neutral"
                                    z_html += f"<span class='{z_class}'>{name} Z: {z:+.1f}σ</span> &nbsp;&nbsp;"
                                z_html += "</div>"
                                st.markdown(z_html, unsafe_allow_html=True)
                                
                                # 2つの指標が同じグラフにある場合、Z-Scoreの乖離(Divergence)を検知
                                if len(latest_z_scores) == 2:
                                    diff = abs(latest_z_scores[0][1] - latest_z_scores[1][1])
                                    if diff > 2.5:
                                        st.markdown(f"<div style='background-color:#2d1b19; padding:5px; border-radius:3px; font-size:12px; color:#f85149;'>⚠️ **Divergence Alert:** {latest_z_scores[0][0]} と {latest_z_scores[1][0]} の動きに過去の相関関係から大きく逸脱した歪み（{diff:.1f}σ差）が発生しています。</div>", unsafe_allow_html=True)

   # ==========================================
    # PAGE 3: Institutional Historical Analog
    # ==========================================
    elif page == "3. Historical Analysis (過去比較)":
        st.markdown("""
        <style>
        .metric-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
        </style>
        """, unsafe_allow_html=True)

        st.title("🕰️ Historical Analog & Regime Projection")
        st.markdown("現在の市場軌跡（T-N日）と過去の金融危機の軌跡を統計的にマッチングし、今後の「ゴーストパス（未来の軌跡）」を投影します。")

        # --- 設定パネル ---
        c_set1, c_set2, c_set3, c_set4 = st.columns(4)
        target_idx = c_set1.selectbox("🎯 分析対象", ["^GSPC (S&P 500)", "^NDX (Nasdaq 100)", "^TNX (10年金利)"])
        ticker = target_idx.split(" ")[0]
        
        # 過去の主要なレジーム（※SPY等ではなく^GSPCを使うことで1980年代のデータも取得可能）
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
        lookforward = 120 # 未来への投影日数 (固定)

        with st.spinner('Historical Analog Data processing...'):
            try:
                # データの取得（対象資産 + VIX）
                data = yf.download([ticker, "^VIX"], start="1980-01-01", progress=False)['Close']
                data.index = pd.to_datetime(data.index).tz_localize(None)
                
                # インデックスの取得（Nearest: 最も近い営業日）
                p_idx = data.index.get_indexer([pd.to_datetime(past_date_str)], method='nearest')[0]
                c_idx = data.index.get_indexer([curr_date], method='nearest')[0]
                
                # --- T=0を基準としたデータの切り出し ---
                # 過去の軌跡 (T-lookback から T+lookforward)
                p_slice = data.iloc[p_idx - lookback : p_idx + lookforward + 1].copy()
                p_slice['T'] = np.arange(-lookback, len(p_slice) - lookback)
                
                # 現在の軌跡 (T-lookback から T=0まで)
                c_slice = data.iloc[c_idx - lookback : c_idx + 1].copy()
                c_slice['T'] = np.arange(-lookback, 1)
                
                # --- 正規化 (T=0を100とする) ---
                p_base_price = p_slice[ticker].loc[p_slice['T'] == 0].values[0]
                c_base_price = c_slice[ticker].loc[c_slice['T'] == 0].values[0]
                
                p_slice['Price_Norm'] = (p_slice[ticker] / p_base_price) * 100
                c_slice['Price_Norm'] = (c_slice[ticker] / c_base_price) * 100

                # --- 統計的類似度 (Similarity Score) の計算 ---
                # T-lookback から T=0 までの両者の相関を計算
                p_compare = p_slice[p_slice['T'] <= 0]['Price_Norm'].values
                c_compare = c_slice['Price_Norm'].values
                
                # データ長がわずかにずれる場合のフェイルセーフ
                min_len = min(len(p_compare), len(c_compare))
                corr = np.corrcoef(p_compare[-min_len:], c_compare[-min_len:])[0, 1]
                
                # --- KPI表示 ---
                st.markdown("---")
                k1, k2, k3 = st.columns(3)
                
                corr_color = "#3fb950" if corr > 0.8 else "#e3b341" if corr > 0.5 else "#f85149"
                k1.markdown(f"""<div class='metric-box'>
                    <h4 style='color:#8b949e; margin-bottom:5px;'>Statistical Similarity (シンクロ率)</h4>
                    <h2 style='color:{corr_color}; margin-top:0px;'>{corr*100:.1f}%</h2>
                    <p style='font-size:12px; color:#8b949e;'>Pearson Correlation (T-{lookback} to T=0)</p>
                </div>""", unsafe_allow_html=True)
                
                p_drawdown = (p_slice['Price_Norm'].min() - 100)
                k2.markdown(f"""<div class='metric-box'>
                    <h4 style='color:#8b949e; margin-bottom:5px;'>Historical Drawdown (T+以降)</h4>
                    <h2 style='color:#f85149; margin-top:0px;'>{p_drawdown:.1f}%</h2>
                    <p style='font-size:12px; color:#8b949e;'>過去レジームでの最大下落率</p>
                </div>""", unsafe_allow_html=True)
                
                c_ret = ((c_slice[ticker].iloc[-1] / c_slice[ticker].iloc[0]) - 1) * 100
                k3.markdown(f"""<div class='metric-box'>
                    <h4 style='color:#8b949e; margin-bottom:5px;'>Current Momentum (T-{lookback} to T=0)</h4>
                    <h2 style='color:{"#3fb950" if c_ret>0 else "#f85149"}; margin-top:0px;'>{c_ret:+.1f}%</h2>
                    <p style='font-size:12px; color:#8b949e;'>現在の期間リターン</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- グラフ描画 (Plotly Subplots) ---
                fig_analog = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                           vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                           subplot_titles=("Price Analog (Rebased to 100 at T=0)", "Volatility Regime (VIX)"))

                # 1. Price Analog
                # 過去の軌跡 (T=0までとT=0以降で線のスタイルを変える)
                p_past = p_slice[p_slice['T'] <= 0]
                p_future = p_slice[p_slice['T'] >= 0]
                
                fig_analog.add_trace(go.Scatter(x=p_past['T'], y=p_past['Price_Norm'], name=f"{past_ev_name} (Up to T=0)", line=dict(color='gray', width=2)), row=1, col=1)
                fig_analog.add_trace(go.Scatter(x=p_future['T'], y=p_future['Price_Norm'], name=f"Ghost Path (Future Projection)", line=dict(color='gray', width=2, dash='dot')), row=1, col=1)
                
                # 現在の軌跡
                fig_analog.add_trace(go.Scatter(x=c_slice['T'], y=c_slice['Price_Norm'], name="Current Market", line=dict(color='#ff4b4b', width=3)), row=1, col=1)

                # 2. VIX Analog
                if "^VIX" in p_slice.columns and "^VIX" in c_slice.columns:
                    fig_analog.add_trace(go.Scatter(x=p_slice['T'], y=p_slice['^VIX'], name=f"VIX: {past_ev_name}", line=dict(color='rgba(128,128,128,0.5)', width=1)), row=2, col=1)
                    fig_analog.add_trace(go.Scatter(x=c_slice['T'], y=c_slice['^VIX'], name="VIX: Current", line=dict(color='#58a6ff', width=2)), row=2, col=1)

                # レイアウト調整
                fig_analog.add_vline(x=0, line_dash="solid", line_color="#8b949e", line_width=1, row='all', col=1) # T=0の縦線
                
                fig_analog.update_layout(template="plotly_dark", height=600, hovermode="x unified",
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig_analog.update_xaxes(title_text="Trading Days relative to T=0", row=2, col=1)
                
                st.plotly_chart(fig_analog, use_container_width=True)

                # ==========================================
                # 🤖 LLM Prompt Generator (Analog Analysis)
                # ==========================================
                with st.expander("📝 Generate Analog Analysis Prompt (LLM用プロンプト)", expanded=False):
                    llm_analog_prompt = f"""あなたはマクロ経済の歴史とクオンツ分析に精通したシニア・ストラテジストです。
現在、当ファンドのアナログ分析エンジンを用いて、現在の市場軌跡と過去の歴史的イベントの軌跡を比較しています。

【アナログ分析結果】
・対象資産: {ticker}
・比較対象の歴史的レジーム: {past_ev_name} (起点: {past_date_str})
・直近{lookback}日間の軌跡の統計的類似度 (Pearson Correlation): {corr*100:.1f}%
・現在の期間リターン: {c_ret:+.1f}%
・もし歴史が全く同じように繰り返した場合の、T=0以降の最大ドローダウンリスク: {p_drawdown:.1f}%

【出力要件】
1. なぜ現在のマクロ環境が「{past_ev_name}」と比較され得るのか、ファンダメンタルズ的な類似点と相違点を考察してください（金利サイクル、インフレ、流動性など）。
2. シンクロ率が{corr*100:.1f}%であることを踏まえ、クオンツの観点から、この「Ghost Path（未来への投影）」をどこまで信用してポジション管理に組み込むべきか、リスクを評価してください。
"""
                    st.code(llm_analog_prompt, language="text")

            except Exception as e:
                st.error(f"アナログ分析データの処理中にエラーが発生しました: {e}")
    # ==========================================
    # PAGE 4: Institutional Quant Engine (Tri-Model Ensemble)
    # ==========================================
    elif page == "4. Investment Strategy (AI戦略)":
        st.markdown("""
        <style>
        .kpi-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .kpi-title { color: #8b949e; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .kpi-value { color: #c9d1d9; font-size: 28px; font-weight: 700; margin-bottom: 0px; }
        .kpi-sub { font-size: 13px; margin-top: 5px; }
        .model-breakdown { font-size: 12px; color: #8b949e; display: flex; justify-content: space-between; border-top: 1px solid #30363d; margin-top: 10px; padding-top: 5px;}
        </style>
        """, unsafe_allow_html=True)

        st.title("🧠 Tri-Model Ensemble Quant Strategy")
        st.markdown("ランダムフォレスト、勾配ブースティング、Elastic Netの3つの独立したアルゴリズムを統合し、非線形なモメンタムと線形なマクロ構造を同時に解析します。")

        # --- サイドバー設定 ---
        with st.sidebar.expander("⚙️ Model Architecture Settings", expanded=True):
            indicator_mode = st.radio("Macro Mode", ["Leading (先行指標特化)", "Full Macro (遅行指標含む)"])
            exclude_other_stocks = st.checkbox("Cross-Asset Exclusion (株価指数の除外)", value=True)
            include_anomaly = st.checkbox("Presidential Cycle (アノマリー追加)", value=False)
            lookback_years = st.slider("Lookback Window (学習期間)", 3, 10, 5)

        with st.spinner('Initializing Tri-Model Ensemble Engine... (ElasticNet, GradientBoosting, RandomForest)'):
            try:
                target_ticker = "SPY"
                
                # --- 1. データパイプライン ---
                yahoo_tickers = settings_df[settings_df['ソース'] == 'Yahoo']['ティッカー'].unique().tolist()
                market_implied_tickers = ['HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                all_yahoo = list(set(yahoo_tickers + market_implied_tickers + [target_ticker]))

                base_fred = ['ANFCI', 'STLFSI4', 'T10Y3M', 'BAMLH0A0HYM2', 'WALCL', 'M2SL', 'ICSA', 'AWHAEMAN', 'T5YIFR', 'PERMIT', 'UMCSENT']
                if "Full Macro" in indicator_mode: base_fred.extend(['CPIAUCSL', 'UNRATE', 'PAYEMS', 'INDPRO'])
                all_fred = list(set(settings_df[settings_df['ソース'] == 'FRED']['ティッカー'].unique().tolist() + base_fred))

                series_list = []
                y_data = yf.download(all_yahoo, period=f"{lookback_years}y", progress=False)['Close']
                if isinstance(y_data, pd.Series): y_data = y_data.to_frame(all_yahoo[0])
                for col in y_data.columns: series_list.append(y_data[col].dropna().rename(col))
                
                for tic in all_fred:
                    try: series_list.append(fred.get_series(tic).loc[f"{2024-lookback_years}-01-01":].dropna().rename(tic))
                    except: pass

                for i in range(len(series_list)): series_list[i].index = pd.to_datetime(series_list[i].index).tz_localize(None).normalize()
                df_ml = pd.concat(series_list, axis=1).ffill()
                
                # --- 2. 特徴量エンジニアリング ---
                df_ml['CTA_200D_Bias'] = df_ml[target_ticker] / df_ml[target_ticker].rolling(200).mean() - 1
                df_ml['Vol_Term_Spread'] = df_ml['^VIX'] / df_ml['^VIX3M']
                df_ml['Copper_Gold_Ratio'] = df_ml['HG=F'] / df_ml['GC=F']
                df_ml['Presidential_Cycle'] = df_ml.index.year % 4
                df_ml['Target'] = df_ml[target_ticker].pct_change(21).shift(-21)
                df_train = df_ml.dropna()

                drop_cols = ['Target', 'HG=F', 'GC=F', 'XLY', 'XLP', '^VIX', '^VIX3M']
                if exclude_other_stocks: drop_cols.extend([t for t in yahoo_tickers if t != target_ticker])
                if not include_anomaly: drop_cols.append('Presidential_Cycle')

                features = [col for col in df_train.columns if col not in drop_cols]
                X, y = df_train[features], df_train['Target']
                
                # --- 3. アンサンブル学習 (Tri-Model Integration) ---
                weights = np.exp(np.linspace(-1, 0, len(X)))
                
                # 線形モデル(ElasticNet)用にデータを標準化
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                latest_x = df_ml[features].iloc[-1:]
                latest_x_scaled = scaler.transform(latest_x)

                # モデル1: Random Forest (分散低減)
                rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
                rf_model.fit(X, y, sample_weight=weights)
                
                # モデル2: Gradient Boosting (バイアス低減・モメンタム捕捉)
                gb_model = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
                gb_model.fit(X, y, sample_weight=weights)
                
                # モデル3: Elastic Net (線形外挿・多重共線性抑制)
                en_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
                en_model.fit(X_scaled, y, sample_weight=weights)
                
                # 個別予測の取得
                pred_rf = rf_model.predict(latest_x)[0] * 100
                pred_gb = gb_model.predict(latest_x)[0] * 100
                pred_en = en_model.predict(latest_x_scaled)[0] * 100
                
                # 最終予測 (3モデルの平均)
                pred_ret = np.mean([pred_rf, pred_gb, pred_en])
                pred_std = np.std([pred_rf, pred_gb, pred_en]) # モデル間の意見の割れ具合
                
                ci_lower = pred_ret - (1.96 * pred_std)
                ci_upper = pred_ret + (1.96 * pred_std)

                # --- 4. Z-Score 解析 & 特徴量重要度 (RFとGBの統合) ---
                df_3y = df_ml.tail(252*3)
                macro_only = [f for f in features if f != 'Presidential_Cycle']
                z_scores = (latest_x[macro_only].iloc[0] - df_3y[macro_only].mean()) / df_3y[macro_only].std()
                
                # RFとGBの重要度を平均化してアンサンブル重要度を算出
                combined_imp = (rf_model.feature_importances_ + gb_model.feature_importances_) / 2
                imp_df = pd.DataFrame({'Feature': features, 'Importance': combined_imp})
                imp_df['Z_Score'] = imp_df['Feature'].map(z_scores).fillna(0)
                imp_df = imp_df.sort_values('Importance', ascending=False).head(10)

                # --- 5. ヒストリカル予測 (Fair Value Band) ---
                td = df_ml.tail(120).copy()
                td_X_scaled = scaler.transform(td[features])
                
                td_pred_rf = rf_model.predict(td[features])
                td_pred_gb = gb_model.predict(td[features])
                td_pred_en = en_model.predict(td_X_scaled)
                
                td_pred_mean = np.mean([td_pred_rf, td_pred_gb, td_pred_en], axis=0)
                td_pred_std = np.std([td_pred_rf, td_pred_gb, td_pred_en], axis=0)
                
                td['AI_Fair'] = (td_pred_mean + 1) * td[target_ticker]
                td['Fair_Upper'] = (td_pred_mean + 1.96 * td_pred_std + 1) * td[target_ticker]
                td['Fair_Lower'] = (td_pred_mean - 1.96 * td_pred_std + 1) * td[target_ticker]
                
                gap = ((df_ml[target_ticker].iloc[-1] - td['AI_Fair'].iloc[-1]) / td['AI_Fair'].iloc[-1]) * 100
                
                vix_dec = df_ml['^VIX'].iloc[-1] / 100
                market_var = vix_dec ** 2
                kelly_f = (pred_ret / 100) / market_var if market_var > 0 else 0
                optimal_weight = max(0, min(100, kelly_f * 100 * 0.5))

                # ==========================================
                # UI Rendering (Dashboard)
                # ==========================================
                
                # --- KPI Cards ---
                c1, c2, c3, c4 = st.columns(4)
                
                color_ret = "#3fb950" if pred_ret > 0 else "#f85149"
                c1.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Ensemble 1M Expected</div>
                    <div class='kpi-value' style='color:{color_ret}'>{pred_ret:+.2f}%</div>
                    <div class='kpi-sub'>95% CI: [{ci_lower:+.1f}%, {ci_upper:+.1f}%]</div>
                    <div class='model-breakdown'>
                        <span>RF: {pred_rf:+.1f}%</span><span>GB: {pred_gb:+.1f}%</span><span>EN: {pred_en:+.1f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)
                
                color_gap = "#f85149" if gap > 2 else "#3fb950" if gap < -2 else "#58a6ff"
                c2.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Fair Value Spread</div>
                    <div class='kpi-value' style='color:{color_gap}'>{gap:+.2f}%</div>
                    <div class='kpi-sub'>Market vs AI Pricing</div>
                </div>""", unsafe_allow_html=True)
                
                c3.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Target Exposure (Kelly)</div>
                    <div class='kpi-value' style='color:#e3b341'>{optimal_weight:.0f}%</div>
                    <div class='kpi-sub'>Cash Recommendation: {100-optimal_weight:.0f}%</div>
                </div>""", unsafe_allow_html=True)
                
                conf_score = max(0, 100 - (pred_std * 15))
                c4.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Model Consensus</div>
                    <div class='kpi-value' style='color:#a371f7'>{conf_score:.1f}/100</div>
                    <div class='kpi-sub'>Agreement across 3 engines</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- Charts Row ---
                col_chart1, col_chart2 = st.columns([1.2, 1])
                
                with col_chart1:
                    st.markdown("#### 📉 Market Price vs Tri-Model Fair Value Band")
                    fig_gap = go.Figure()
                    fig_gap.add_trace(go.Scatter(x=td.index.tolist() + td.index[::-1].tolist(), y=td['Fair_Upper'].tolist() + td['Fair_Lower'][::-1].tolist(), fill='toself', fillcolor='rgba(88, 166, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='Model Consensus Band'))
                    fig_gap.add_trace(go.Scatter(x=td.index, y=td['AI_Fair'], name="Ensemble Fair Value", line=dict(dash='dot', color='#58a6ff', width=2)))
                    fig_gap.add_trace(go.Scatter(x=td.index, y=td[target_ticker], name="SPY Price", line=dict(color='#ff4b4b', width=2)))
                    fig_gap.update_layout(template="plotly_dark", height=380, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig_gap, use_container_width=True)

                with col_chart2:
                    st.markdown("#### 🔍 Integrated Feature Attribution")
                    fig_brain = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h', color='Z_Score', color_continuous_scale='RdBu_r', range_color=[-3, 3], template="plotly_dark")
                    fig_brain.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), coloraxis_colorbar=dict(title="Z-Score"))
                    st.plotly_chart(fig_brain, use_container_width=True)

                # ==========================================
                # 🤖 LLM Prompt Generator
                # ==========================================
                with st.expander("📝 Generate Quantitative Report Prompt (LLM用プロンプト)", expanded=False):
                    top_features_text = "".join([f"- {row['Feature']}: 重要度 {row['Importance']:.4f}, 現在のZ-Score {row['Z_Score']:+.2f}\n" for _, row in imp_df.head(8).iterrows()])
                    anfci_val = df_ml['ANFCI'].iloc[-1] if 'ANFCI' in df_ml.columns else 0.0

                    llm_prompt = f"""あなたはトップ・クオンツファンドのシニア・ポートフォリオマネージャーです。
自社開発のTri-Model Ensemble AI（Random Forest, Gradient Boosting, Elastic Netの統合モデル）の推論結果を元に、投資委員会向けの市場解説およびアロケーション戦略レポートを作成してください。

【アンサンブルAI予測メトリクス (対象: 米国株, ホライズン: 1ヶ月)】
・統合予測リターン: {pred_ret:+.2f}% 
  (内訳: Random Forest {pred_rf:+.2f}%, Gradient Boosting {pred_gb:+.2f}%, Elastic Net {pred_en:+.2f}%)
・モデルコンセンサス度: {conf_score:.1f}/100 (3つのモデルの意見一致度)
・AI適正価格との乖離(Spread): {gap:+.2f}% (プラス=市場が割高)
・VIX: {df_ml['^VIX'].iloc[-1]:.2f} / 金融ストレス(ANFCI): {anfci_val:.2f}
・最適ポジション露出度 (ハーフ・ケリー基準): {optimal_weight:.0f}%

【AI決定要因: 上位特徴量とZ-Score異常値】
{top_features_text}

【出力要件】
1. マクロ環境の総括: 特徴量のZ-Score異常値を起点とし、なぜAIがこのリターンを算出したか因果関係を解説してください。
2. モデル解析: 3つのモデル間で予測値に「差」がある場合、それはなぜ生じているか（非線形な勢い vs 線形なファンダメンタルズのどちらが強いか）を考察してください。
3. クロスアセット・ランキング: 上記環境下において、[株, 債券, 現金, 原油, 金] の中で、今後1ヶ月でアウトパフォームする順に1〜5位のランキングと、クオンツ的根拠を提示してください。
"""
                    st.code(llm_prompt, language="text")

            except Exception as e: st.error(f"分析モデル・エラー: {e}\n(追加モジュール scikit-learn が必要です。)")

# ==========================================
    # PAGE 5: Headline Reverse-Engineering (イベント逆引き)
    # ==========================================
    elif page == "5. Headline Reverse-Engineering (イベント逆引き)":
        st.title("🗞️ Volatility Anomaly & Reverse-Engineering")
        st.markdown("株価が統計的な異常反応（3シグマ級の変動）を示した日を数学的に特定し、LLMを使ってその背景にあるニュースヘッドラインを逆引きします。")

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

            except Exception as e: 
                st.error(f"イベント特定エラー: {e}")


    # ==========================================
    # PAGE 6: Institutional Portfolio Optimizer (Black-Litterman)
    # ==========================================
    elif page == "6. Portfolio Optimization (アロケーション)":
        st.markdown("""
        <style>
        .kpi-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .kpi-title { color: #8b949e; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .kpi-value { color: #c9d1d9; font-size: 28px; font-weight: 700; margin-bottom: 0px; }
        </style>
        """, unsafe_allow_html=True)

        st.title("⚖️ Black-Litterman Portfolio Optimizer")
        st.markdown("市場の均衡リターンと、AIエンジンの独自予測（Views）をベイズ推定で統合し、Max Sharpe（最大投資効率）となる最適ポートフォリオウェイトを算出します。")

        with st.spinner('Calculating Covariance Matrix & Black-Litterman Posteriors...'):
            try:
                # --- 1. アセットデータの取得 ---
                assets = {'SPY': 'Equities', 'TLT': 'Bonds', 'GLD': 'Gold', 'USO': 'Commodities'}
                tickers = list(assets.keys())
                
                prices = yf.download(tickers, period="5y", progress=False)['Close']
                returns = prices.pct_change().dropna()
                
                # --- 2. 基礎統計量の計算 ---
                cov_matrix = returns.cov() * 252
                mkt_weights = np.array([0.60, 0.20, 0.10, 0.10])
                
                risk_free_rate = 0.04 
                spy_ret = returns['SPY'].mean() * 252
                spy_var = returns['SPY'].var() * 252
                risk_aversion = (spy_ret - risk_free_rate) / spy_var
                
                pi = risk_aversion * np.dot(cov_matrix, mkt_weights)
                
                # --- 3. AI Views の生成 ---
                views = []
                confidences = []
                for tic in tickers:
                    ret_3m = (prices[tic].iloc[-1] / prices[tic].iloc[-63]) - 1
                    vol_1m = returns[tic].tail(21).std() * np.sqrt(252)
                    view_ret = (ret_3m * 4) * 0.5 + (pi[tickers.index(tic)] * 0.5) 
                    views.append(view_ret)
                    confidences.append(1.0 / (vol_1m + 0.01))

                Q = np.array(views)
                P = np.eye(len(tickers)) 
                tau = 0.05 
                omega = np.diag([(tau * cov_matrix.iloc[i, i]) / confidences[i] for i in range(len(tickers))])

                # --- 4. Black-Litterman 事後期待リターンの計算 ---
                tau_cov_inv = la.inv(tau * cov_matrix)
                omega_inv = la.inv(omega)
                
                bl_expected_returns = np.dot(
                    la.inv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P)),
                    np.dot(tau_cov_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
                )

                # --- 5. Mean-Variance Optimization ---
                def get_ret_vol_sr(weights, exp_returns):
                    weights = np.array(weights)
                    port_ret = np.sum(exp_returns * weights)
                    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sr = (port_ret - risk_free_rate) / port_vol
                    return np.array([port_ret, port_vol, sr])

                def neg_sharpe(weights, exp_returns): return -get_ret_vol_sr(weights, exp_returns)[2]
                def check_sum(weights): return np.sum(weights) - 1

                constraints = ({'type': 'eq', 'fun': check_sum})
                bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))
                init_guess = len(tickers) * [1.0 / len(tickers)]

                opt_results_bl = sco.minimize(neg_sharpe, init_guess, args=(bl_expected_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
                opt_weights_bl = opt_results_bl.x

                # --- 6. UI Rendering ---
                c1, c2, c3 = st.columns(3)
                bl_stats = get_ret_vol_sr(opt_weights_bl, bl_expected_returns)
                mkt_stats = get_ret_vol_sr(mkt_weights, pi)

                c1.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Expected Portfolio Return</div>
                    <div class='kpi-value' style='color:#3fb950'>{bl_stats[0]*100:.1f}%</div>
                    <div class='kpi-sub'>Market Baseline: {mkt_stats[0]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                c2.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Portfolio Volatility (Risk)</div>
                    <div class='kpi-value' style='color:#f85149'>{bl_stats[1]*100:.1f}%</div>
                    <div class='kpi-sub'>Market Baseline: {mkt_stats[1]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                c3.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-title'>Ex-Ante Sharpe Ratio</div>
                    <div class='kpi-value' style='color:#a371f7'>{bl_stats[2]:.2f}</div>
                    <div class='kpi-sub'>Market Baseline: {mkt_stats[2]:.2f}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- アロケーション比較チャート ---
                st.subheader("Asset Allocation: Market Equilibrium vs AI Black-Litterman")
                df_weights = pd.DataFrame({
                    'Asset': [assets[t] for t in tickers],
                    'Market Neutral (60/40 basis)': mkt_weights * 100,
                    'AI Optimized (Max Sharpe)': opt_weights_bl * 100
                }).melt(id_vars='Asset', var_name='Strategy', value_name='Weight (%)')

                fig_weights = px.bar(df_weights, x='Asset', y='Weight (%)', color='Strategy', barmode='group', 
                                     color_discrete_sequence=['#8b949e', '#58a6ff'], template="plotly_dark")
                fig_weights.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_weights, use_container_width=True)

                # --- 事前・事後リターンのシフト ---
                st.markdown("---")
                st.subheader("Expected Return Shift (AI Views Impact)")
                
                df_ret_shift = pd.DataFrame({
                    'Asset': [assets[t] for t in tickers],
                    'Implied (Market)': pi * 100,
                    'AI View (Raw Prediction)': Q * 100,
                    'Black-Litterman Posterior': bl_expected_returns * 100
                })
                
                fig_shift = go.Figure()
                fig_shift.add_trace(go.Scatter(x=df_ret_shift['Asset'], y=df_ret_shift['Implied (Market)'], name='Market Implied', mode='markers', marker=dict(size=12, symbol='circle-open', color='gray')))
                fig_shift.add_trace(go.Scatter(x=df_ret_shift['Asset'], y=df_ret_shift['AI View (Raw Prediction)'], name='AI Raw View', mode='markers', marker=dict(size=12, symbol='x', color='#f85149')))
                fig_shift.add_trace(go.Scatter(x=df_ret_shift['Asset'], y=df_ret_shift['Black-Litterman Posterior'], name='BL Posterior (Final)', mode='markers', marker=dict(size=16, symbol='star', color='#3fb950')))
                
                for i in range(len(tickers)):
                    fig_shift.add_annotation(x=df_ret_shift['Asset'][i], y=df_ret_shift['Black-Litterman Posterior'][i],
                                             ax=df_ret_shift['Asset'][i], ay=df_ret_shift['Implied (Market)'][i],
                                             xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#8b949e')

                fig_shift.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Expected Return (Annualized %)")
                st.plotly_chart(fig_shift, use_container_width=True)

            except Exception as e:
                st.error(f"最適化エンジンのエラー: {e}")

# ==========================================
# グローバル・エラーハンドリング (システムの終端)
# ==========================================
except Exception as e:
    st.error(f"System Critical Error: {e}")
