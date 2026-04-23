import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go

# ページの設定
st.set_page_config(page_title="My Macro Tracker", layout="wide")
st.title("📊 My Macro Tracker")

# Streamlitの「シークレット機能」からカギとURLを安全に読み込む
API_KEY = st.secrets["FRED_API_KEY"]
SHEET_URL = st.secrets["SHEET_URL"]

fred = Fred(api_key=API_KEY)

# スプレッドシートのデータを読み込む（1時間キャッシュして高速化）
@st.cache_data(ttl=3600)
def load_settings():
    return pd.read_csv(SHEET_URL)

try:
    settings_df = load_settings()
    
    # スプレッドシートの「グループ」ごとにタブを自動生成
    groups = settings_df['グループ'].unique()
    tabs = st.tabs(groups)

    # 各タブの中にグラフを描画
    for i, group in enumerate(groups):
        with tabs[i]:
            st.header(f"{group} の動向")
            group_df = settings_df[settings_df['グループ'] == group]
            
            for index, row in group_df.iterrows():
                name = row['指標名']
                ticker = row['ティッカー']
                
                try:
                    # データの取得とグラフ化
                    data = fred.get_series(ticker)
                    data = data.loc['2020-01-01':] # 2020年以降に絞る
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data.values, name=name, mode='lines', line=dict(width=2)))
                    fig.update_layout(title=name, hovermode="x unified", height=350, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"{name} のデータが取得できませんでした。")
                    
except Exception as e:
    st.error(f"エラーの詳細: {e}")
