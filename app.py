import streamlit as st

# アプリ全体の設定（必ず一番上に書きます）
st.set_page_config(page_title="Macro Quant Terminal Pro", layout="wide", initial_sidebar_state="expanded")

# 共通デザイン（CSS）
st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Consolas', monospace; }
</style>
""", unsafe_allow_html=True)

# トップページの表示内容
st.title("💎 Macro Quant Terminal Pro")
st.markdown("左側のサイドバーから、実行したい分析メニューを選択してください。")
st.info("✅ システムのマルチページ最適化に成功しました！")
