import os
import streamlit as st

def get_gemini_api_key(user_key: str = None) -> str:
    """Auto-detect API Key from Sidebar, Secrets, or Env Vars"""
    if user_key and user_key.strip():
        return user_key.strip()
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY")
    return None