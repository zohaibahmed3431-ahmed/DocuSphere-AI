import os
from openai import OpenAI
from config.settings import OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL
from ai.prompts import SYSTEM_ACCOUNTANT_PROMPT

def query_openrouter_ai(user_prompt, data_context="", api_key=None):
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return "⚠️ OpenRouter API Key is missing. Please enter your valid API Key in the sidebar configuration."
    
    try:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
        )
        
        full_content = f"{SYSTEM_ACCOUNTANT_PROMPT}\n\n[FINANCIAL DATASET CONTEXT]:\n{data_context}\n\n[USER INSTRUCTION]:\n{user_prompt}"
        
        response = client.chat.completions.create(
            model=DEFAULT_OPENROUTER_MODEL,
            messages=[
                {"role": "user", "content": full_content}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenRouter API Communication Error: {str(e)}"
