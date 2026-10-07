import google.generativeai as genai
from core.config import get_gemini_api_key

def generate_ai_response(user_query: str, file_context: str, user_api_key: str = None) -> str:
    api_key = get_gemini_api_key(user_api_key)
    if not api_key:
        return f"**Query Received:** *'{user_query}'*\n\n{file_context}\n\n*(Please enter a valid Gemini API Key in sidebar or st.secrets to get live AI responses)*"

    try:
        genai.configure(api_key=api_key)
        prompt = f"""
You are DocuSphere, an intelligent multimodal AI assistant.

SYSTEM INSTRUCTIONS:
1. Detect the language of user input (Urdu, Roman Urdu, English, Arabic, etc.) and respond in the EXACT same language naturally.
2. If document context is relevant, give clear, accurate references.
3. If user is asking general questions, answer directly using your AI intelligence.

DOCUMENT CONTEXT:
{file_context}

USER QUERY:
{user_query}
"""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text

    except Exception as e:
        return f"❌ Gemini API Engine Error: {str(e)}"