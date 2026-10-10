import json

import streamlit as st

from ai.prompts import SYSTEM_PROMPT
from core.utils import dataframe_to_context


def build_context(
    bank_df,
    ledger_df,
    summary,
    uploaded_files_info,
):
    context_parts = []

    context_parts.append("### Uploaded Files")
    for file_info in uploaded_files_info:
        context_parts.append(
            f"- {file_info['name']} ({file_info['type']})"
        )

    if summary:
        context_parts.append("\n### Reconciliation Summary")
        context_parts.append(json.dumps(summary, indent=2, default=str))

    if bank_df is not None:
        context_parts.append("\n### Bank Transactions Sample")
        context_parts.append(dataframe_to_context(bank_df))

    if ledger_df is not None:
        context_parts.append("\n### Ledger Transactions Sample")
        context_parts.append(dataframe_to_context(ledger_df))

    return "\n".join(context_parts)


def ask_ai(
    provider,
    api_key,
    model,
    question,
    context,
):
    if provider == "OpenAI":
        return ask_openai(api_key, model, question, context)

    if provider == "Gemini":
        return ask_gemini(api_key, model, question, context)

    return "Please select a valid AI provider."


def ask_openai(api_key, model, question, context):
    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI package install nahi hai. 'pip install openai' chalayein."

    if not api_key:
        return "Please enter your OpenAI API key in the sidebar."

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{context}\n\n### User Question\n{question}",
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def ask_gemini(api_key, model, question, context):
    try:
        import google.generativeai as genai
    except ImportError:
        return (
            "Gemini package install nahi hai. "
            "'pip install google-generativeai' chalayein."
        )

    if not api_key:
        return "Please enter your Gemini API key in the sidebar."

    genai.configure(api_key=api_key)

    model_instance = genai.GenerativeModel(model)

    prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\n### User Question\n{question}"

    response = model_instance.generate_content(prompt)

    return response.text