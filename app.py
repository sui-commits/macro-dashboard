import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np

# --- 初期設定 ---
st.set_page_config(page_title="My Macro Tracker", layout="wide")
st.title("🌐 My Macro Tracker & Market Dashboard")

API_KEY = st.secrets["FRED_API_KEY"]
SHEET_URL = st.secrets["SHEET_URL"]
fred = Fred(api_key=API_KEY)

@st.cache_data(ttl=600) # 10分間隔で更新
def load_settings():
    df = pd.read_csv(SHEET_URL)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df

try:
    settings_df = load_settings()
    tabs_list = settings_df['タブ名'].unique().tolist()
    tabs = st.tabs(tabs_list)

    for i, tab_name in enumerate(tabs_list):
        with tabs[i]:
            st.header(f"📊 {tab_name}")
            tab_df = settings_df[settings_df['タブ名'] == tab_name]
            graph_names = tab_df['グラフ名'].unique().tolist()
            
            for graph_name in graph_names:
                graph_df = tab_df[tab_df['グラフ名'] == graph_name]
                first_source = graph_df.iloc[0]['ソース']
                
                # ==========================================
                # ① オプション市場（Max Pain）の処理
                # ==========================================
                if first_source == 'Options':
                    cols = st.columns(min(len(graph_df), 3)) # 横並びで表示
                    for index, row in graph_df.iterrows():
                        ticker = row['ティッカー']
                        data_name = row['データ名']
                        
                        with cols[index % len(cols)]:
                            try:
                                stock = yf.Ticker(ticker)
                                dates = stock.options
                                if not dates:
                                    st.warning(f"{ticker} のオプションデータなし")
                                    continue
                                    
                                expiry = dates[0]
                                chain = stock.option_chain(expiry)
                                calls, puts = chain.calls, chain.puts
                                
                                strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
                                max_pain, min_loss = 0, float('inf')
                                
                                for strike in strikes:
                                    call_loss = calls[calls['strike'] < strike].apply(lambda x: (strike - x['strike']) * x['openInterest'], axis=1).sum()
                                    put_loss = puts[puts['strike'] > strike].apply(lambda x: (x['strike'] - strike) * x['openInterest'], axis=1).sum()
                                    if (call_loss + put_loss) < min_loss:
                                        min_loss = call_loss + put_loss
                                        max_pain = strike
                                
                                current_price = stock.history(period='1d')['Close'].iloc[-1]
                                
                                fig = go.Figure()
                                fig.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], name='Call', marker_color='rgba(0, 0, 255, 0.6)'))
                                fig.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'], name='Put', marker_color='rgba(255, 0, 0, 0.6)'))
                                fig.add_vline(x=max_pain, line_dash="dash", line_color="yellow", annotation_text=f"Max Pain: {max_pain}")
                                fig.add_vline(x=current_price, line_dash="solid", line_color="green", annotation_text=f"Price: {current_price:.2f}")
                                
                                fig.update_layout(title=f"{data_name} (満期: {expiry})", barmode='group', hovermode="x unified", dragmode='pan')
                                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                            except Exception as e:
                                st.error(f"{data_name} エラー: {e}")
                                
                # ==========================================
                # ② CTAトレンド（疑似システムトレード）の処理
                # ==========================================
                elif first_source == 'CTA':
                    for index, row in graph_df.iterrows():
                        ticker = row['ティッカー']
                        data_name = row['データ名']
                        try:
                            data = yf.Ticker(ticker).history(period="1y")
                            data['SMA20'] = data['Close'].rolling(window=20).mean()
                            data['SMA200'] = data['Close'].rolling(window=200).mean()
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='価格', line=dict(color='white', width=2)))
                            fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='短期(20日)', line=dict(color='orange', width=1)))
                            fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='長期(200日)', line=dict(color='blue', width=1)))
                            
                            # 背景色でトレンド（Long/Short）を示す
                            fig.add_trace(go.Scatter(x=data.index, y=np.where(data['SMA20'] > data['SMA200'], data['Close'].max(), data['Close'].min()), 
                                                     name='CTA Trend', mode='lines', fill='tozeroy', line=dict(width=0), fillcolor='rgba(0, 255, 0, 0.1)'))
                            
                            fig.update_layout(title=f"{data_name} - CTAポジショニング動向", hovermode="x unified", dragmode='pan')
                            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                        except Exception as e:
                            st.error(f"{data_name} CTA計算エラー: {e}")

                # ==========================================
                # ③ マクロ指標・時系列データの重ね合わせ処理
                # ==========================================
                else:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    for index, row in graph_df.iterrows():
                        data_name = row['データ名']
                        ticker = row['ティッカー']
                        source = row['ソース']
                        is_secondary = True if row['軸'] == '副軸' else False
                        
                        try:
                            if source == 'FRED':
                                data = fred.get_series(ticker).loc['2020-01-01':]
                                dates, values = data.index, data.values
                            elif source == 'Yahoo':
                                data = yf.Ticker(ticker).history(start='2020-01-01')
                                dates, values = data.index, data['Close'].values
                            else:
                                continue
                                
                            fig.add_trace(go.Scatter(x=dates, y=values, name=data_name, mode='lines', line=dict(width=2)), secondary_y=is_secondary)
                        except Exception as e:
                            st.warning(f"{data_name} の取得に失敗: {e}")
                    
                    fig.update_layout(title=graph_name, hovermode="x unified", height=500, dragmode='pan')
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                    
                st.markdown("---")
except Exception as e:
    st.error(f"システムエラー: {e}")
