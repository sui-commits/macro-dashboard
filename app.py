import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from dbnomics import fetch_series

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

@st.cache_data(ttl=3600) # 読み込みが重くなるためキャッシュ時間を1時間に延長
def load_settings():
    df = pd.read_csv(SHEET_URL)
    for col in df.columns: df[col] = df[col].astype(str).str.strip()
    return df

try:
    settings_df = load_settings()
    
    st.sidebar.title("💎 Macro Navigation")
    page = st.sidebar.radio("機能を選択", ["1. Market Dynamics (現在)", "2. Macro Economics (深層)", "3. Historical Analysis (過去比較)"])

    # ==========================================
    # PAGE 1: Market Dynamics
    # ==========================================
    if page == "1. Market Dynamics (現在)":
        st.title("📈 Market Dynamics & Terminal Board")
        
        st.markdown("### ⚡ Real-time Market Board")
        board_cols = st.columns(5)
        count = 0
        for _, row in settings_df.iterrows():
            if row['ソース'] in ['FRED', 'Yahoo'] and count < 30:
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
        st.title("🏛️ Macro Economic Indicators (PhD Level)")
        
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
                                if r['ソース'] == 'FRED':
                                    d = fred.get_series(r['ティッカー']).loc['2020-01-01':]
                                elif r['ソース'] == 'Yahoo':
                                    d = yf.Ticker(r['ティッカー']).history(start='2020-01-01')['Close']
                                elif r['ソース'] == 'DBnomics':
                                    db_df = fetch_series(r['ティッカー'])
                                    if not db_df.empty:
                                        d = db_df[['period', 'value']].dropna().set_index('period')['value']
                                
                                if d is not None and not d.empty:
                                    d.index = pd.to_datetime(d.index).tz_localize(None)
                                    current_dt = d.index.max()
                                    if max_dt is None or current_dt > max_dt: max_dt = current_dt
                                    fig.add_trace(go.Scatter(x=d.index, y=d.values, name=r['データ名']), secondary_y=(r['軸']=='副軸'))
                            except Exception as e:
                                pass # エラーはスキップして次の指標へ
                        
                        # ★ 直近半年間のグラフ算出（X軸の強制ズーム設定）
                        if max_dt: 
                            fig.update_xaxes(range=[max_dt - pd.DateOffset(months=6), max_dt + pd.DateOffset(days=5)])
                            
                        fig.update_layout(title=g_name, height=300, template="plotly_dark", hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PAGE 3: Historical Analysis (結論算出エンジン)
    # ==========================================
    elif page == "3. Historical Analysis (過去比較)":
        st.title("🕰️ Historical Regime & Forward Return Analysis")
        st.markdown("過去の危機と現在の軌跡を正規化(T=0=100)して比較し、統計的結論を導きます。")
        
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("分析指数", ["^GSPC (S&P 500)", "^DJI (Dow Jones)", "^NDX (Nasdaq 100)"])
        ticker = target.split(" ")[0]
        regimes = {
            "2022 Inflation Shock": "2022-01-03", 
            "2008 Lehman Shock": "2008-09-15", 
            "2000 Dot-com Bubble": "2000-03-24", 
            "1987 Black Monday": "1987-10-19",
            "1973 Oil Crisis": "1973-10-01"
        }
        past_ev = c2.selectbox("過去のレジーム(T=0)", list(regimes.keys()))
        curr_ev = pd.to_datetime(c3.date_input("現在の比較起点(T=0)", pd.to_datetime("2024-01-01"))).tz_localize(None)
        days_to_track = st.slider("比較する営業日数 (Trading Days)", 50, 500, 250)

        if st.button("結論を算出 (Run Quantitative Analysis)"):
            with st.spinner('軌跡と統計的結論を算出中...'):
                try:
                    d = yf.Ticker(ticker).history(start="1970-01-01")['Close']
                    d.index = pd.to_datetime(d.index).tz_localize(None)
                    
                    p_idx = d.index.get_indexer([pd.to_datetime(regimes[past_ev])], method='nearest')[0]
                    c_idx = d.index.get_indexer([curr_ev], method='nearest')[0]
                    
                    offset = int(days_to_track / 4)
                    p_slice = (d.iloc[max(0, p_idx - offset) : p_idx + days_to_track] / d.iloc[p_idx]) * 100
                    c_slice = (d.iloc[max(0, c_idx - offset) : c_idx + days_to_track] / d.iloc[c_idx]) * 100
                    
                    p_x = np.arange(-len(p_slice[:p_idx - max(0, p_idx - offset)]), len(p_slice) - len(p_slice[:p_idx - max(0, p_idx - offset)]))
                    c_x = np.arange(-len(c_slice[:c_idx - max(0, c_idx - offset)]), len(c_slice) - len(c_slice[:c_idx - max(0, c_idx - offset)]))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=p_x, y=p_slice.values, name=f"Past: {past_ev}", line=dict(dash='dash', color='gray', width=2)))
                    fig.add_trace(go.Scatter(x=c_x, y=c_slice.values, name=f"Current: {curr_ev.strftime('%Y-%m-%d')}", line=dict(width=3, color='#ff4b4b')))
                    fig.add_vline(x=0, line_dash="solid", line_color="rgba(255,255,255,0.5)")
                    fig.update_layout(title=f"{ticker} Regime Alignment (T=0=100)", height=450, template="plotly_dark", xaxis_title="Trading Days from T=0")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ★ 統計的結論の計算（エラー修正済）
                    st.markdown("### 📊 Statistical Conclusion (Forward Returns)")
                    
                    def get_ret(idx, days):
                        if idx + days < len(d): return ((d.iloc[idx+days]/d.iloc[idx])-1)*100
                        return np.nan
                    
                    def get_mdd(idx, days):
                        if idx + days < len(d):
                            w = d.iloc[idx : idx+days]
                            return ((w - w.cummax()) / w.cummax()).min() * 100
                        return np.nan

                    res = pd.DataFrame({
                        "Timeframe": ["+1 Month (21d)", "+3 Months (63d)", "+6 Months (126d)", "+1 Year (252d)"],
                        f"Past Return ({past_ev})": [
                            f"{get_ret(p_idx, 21):.2f}%", f"{get_ret(p_idx, 63):.2f}%", 
                            f"{get_ret(p_idx, 126):.2f}%", f"{get_ret(p_idx, 252):.2f}%"
                        ],
                        "Current Return (So far)": [
                            f"{get_ret(c_idx, 21):.2f}%" if not np.isnan(get_ret(c_idx, 21)) else "N/A",
                            f"{get_ret(c_idx, 63):.2f}%" if not np.isnan(get_ret(c_idx, 63)) else "N/A",
                            f"{get_ret(c_idx, 126):.2f}%" if not np.isnan(get_ret(c_idx, 126)) else "N/A",
                            f"{get_ret(c_idx, 252):.2f}%" if not np.isnan(get_ret(c_idx, 252)) else "N/A"
                        ]
                    })
                    st.table(res)
                    
                    mdd = get_mdd(p_idx, 252)
                    st.error(f"⚠️ **【ストレステスト・リスク結論】** T=0から1年以内の **最大下落幅 (Max Drawdown) は {mdd:.2f}%** でした。過去の軌跡をなぞる場合、現在位置からこの水準までの下落リスクをポートフォリオに組み込んでください。")

                except Exception as e:
                    st.error(f"解析・計算エラー: 詳細を確認してください ({e})")

except Exception as e:
    st.error(f"System Error: {e}")
