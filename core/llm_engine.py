import google.generativeai as genai
from core.config import get_gemini_api_key

def generate_ai_response(user_query: str, file_context: str) -> str:
    # Key backend environment ya Streamlit secrets se auto-fetch hogi
    api_key = get_gemini_api_key()
    
    if not api_key:
        return "❌ **API Key Missing:** Streamlit Cloud Secrets mein `GEMINI_API_KEY` set karein."

    try:
        genai.configure(api_key=api_key)
        
        prompt = f"""
You are DocuSphere, an intelligent multimodal AI assistant.

SYSTEM INSTRUCTIONS:
1. Detect the user's input language and respond in the EXACT same language naturally.
2. If document context is present, cite specific page/data details accurately.
3. If no document is needed, act as a direct, helpful AI.

DOCUMENT CONTEXT:
{file_context}

USER QUERY:
{user_query}
"""
        # Updated models endpoint
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"❌ AI Engine Error: {str(e)}"
