import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np

# --- 初期設定 ---
st.set_page_config(page_title="Macro Cockpit", layout="wide")
st.title("🌐 Ultimate Macro Cockpit (225225 Style)")

API_KEY = st.secrets["FRED_API_KEY"]
SHEET_URL = st.secrets["SHEET_URL"]
fred = Fred(api_key=API_KEY)

@st.cache_data(ttl=600)
def load_settings():
    df = pd.read_csv(SHEET_URL)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df

try:
    settings_df = load_settings()
    # 既存のタブの前に、225225風の「マーケットボード」タブを強制追加
    tabs_list = ["🔥 225225ボード"] + settings_df['タブ名'].unique().tolist()
    tabs = st.tabs(tabs_list)

    # ==========================================
    # ① 225225風 マーケットボード（全指標のタイル表示）
    # ==========================================
    with tabs[0]:
        st.subheader("リアルタイム・マクロボード")
        
        # 4列のグリッドを作成して敷き詰める（スマホでも見やすいレイアウト）
        board_cols = st.columns(4)
        count = 0
        
        for index, row in settings_df.iterrows():
            if row['ソース'] in ['FRED', 'Yahoo']:
                ticker = row['ティッカー']
                name = row['データ名']
                try:
                    if row['ソース'] == 'FRED':
                        data = fred.get_series(ticker).dropna()
                        if len(data) >= 2:
                            current = data.iloc[-1]
                            previous = data.iloc[-2]
                            delta = current - previous
                            # FRED指標は小数点2桁
                            board_cols[count % 4].metric(label=name, value=f"{current:.2f}", delta=f"{delta:.2f}")
                            count += 1
                    elif row['ソース'] == 'Yahoo':
                        data = yf.Ticker(ticker).history(period="5d")['Close'].dropna()
                        if len(data) >= 2:
                            current = data.iloc[-1]
                            previous = data.iloc[-2]
                            delta = current - previous
                            delta_pct = (delta / previous) * 100
                            # 株価等は値とパーセントを表示
                            board_cols[count % 4].metric(label=name, value=f"{current:.2f}", delta=f"{delta:.2f} ({delta_pct:.2f}%)")
                            count += 1
                except Exception:
                    pass
        st.markdown("---")

    # ==========================================
    # ② 各種マクロチャート・CTA・オプションの描画
    # ==========================================
    for i, tab_name in enumerate(tabs_list[1:], start=1):
        with tabs[i]:
            tab_df = settings_df[settings_df['タブ名'] == tab_name]
            graph_names = tab_df['グラフ名'].unique().tolist()
            
            chart_cols = st.columns(2) # グラフは2列でコンパクトに
            
            for g_idx, graph_name in enumerate(graph_names):
                graph_df = tab_df[tab_df['グラフ名'] == graph_name]
                first_source = graph_df.iloc[0]['ソース']
                
                with chart_cols[g_idx % 2]:
                    if first_source == 'Options':
                        for index, row in graph_df.iterrows():
                            ticker = row['ティッカー']
                            data_name = row['データ名']
                            try:
                                stock = yf.Ticker(ticker)
                                dates = stock.options
                                if not dates: continue
                                expiry = dates[0]
                                calls, puts = stock.option_chain(expiry).calls, stock.option_chain(expiry).puts
                                
                                strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
                                max_pain, min_loss = 0, float('inf')
                                for strike in strikes:
                                    loss = calls[calls['strike'] < strike].apply(lambda x: (strike - x['strike']) * x['openInterest'], axis=1).sum() + \
                                           puts[puts['strike'] > strike].apply(lambda x: (x['strike'] - strike) * x['openInterest'], axis=1).sum()
                                    if loss < min_loss: min_loss, max_pain = loss, strike
                                
                                current_price = stock.history(period='1d')['Close'].iloc[-1]
                                fig = go.Figure()
                                fig.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], name='Call', marker_color='blue'))
                                fig.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'], name='Put', marker_color='red'))
                                fig.add_vline(x=max_pain, line_dash="dash", line_color="yellow", annotation_text=f"Max Pain: {max_pain}", annotation_position="top left")
                                fig.add_vline(x=current_price, line_dash="solid", line_color="green", annotation_text=f"Price: {current_price:.2f}", annotation_position="bottom right")
                                fig.update_layout(title=f"{data_name} (満期: {expiry})", height=280, margin=dict(l=0,r=0,t=30,b=0), dragmode='pan')
                                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                            except Exception: pass
                                
                    elif first_source == 'CTA':
                        for index, row in graph_df.iterrows():
                            try:
                                data = yf.Ticker(row['ティッカー']).history(period="1y")
                                data['SMA20'], data['SMA200'] = data['Close'].rolling(20).mean(), data['Close'].rolling(200).mean()
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='価格', line=dict(color='white')))
                                fig.add_trace(go.Scatter(x=data.index, y=np.where(data['SMA20'] > data['SMA200'], data['Close'].max(), data['Close'].min()), name='CTA Trend', fill='tozeroy', line=dict(width=0), fillcolor='rgba(0, 255, 0, 0.1)'))
                                fig.update_layout(title=f"{row['データ名']}", height=280, margin=dict(l=0,r=0,t=30,b=0), dragmode='pan')
                                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                            except Exception: pass

                    else:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        for index, row in graph_df.iterrows():
                            try:
                                if row['ソース'] == 'FRED':
                                    data = fred.get_series(row['ティッカー']).loc['2020-01-01':]
                                elif row['ソース'] == 'Yahoo':
                                    data = yf.Ticker(row['ティッカー']).history(start='2020-01-01')['Close']
                                fig.add_trace(go.Scatter(x=data.index, y=data.values, name=row['データ名']), secondary_y=(row['軸']=='副軸'))
                            except Exception: pass
                        fig.update_layout(title=graph_name, hovermode="x unified", height=280, margin=dict(l=0,r=0,t=30,b=0), dragmode='pan')
                        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

except Exception as e:
    st.error(f"システムエラー: {e}")
