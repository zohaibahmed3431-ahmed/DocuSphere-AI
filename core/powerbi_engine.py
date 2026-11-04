import pandas as pd
import plotly.express as px

def generate_powerbi_dashboard(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    rev_col = next((c for c in numeric_cols if any(k in c.lower() for k in ['sale', 'revenue', 'income', 'credit', 'amount'])), None)
    exp_col = next((c for c in numeric_cols if any(k in c.lower() for k in ['expense', 'cost', 'debit'])), None)
    cat_col = next((c for c in text_cols if any(k in c.lower() for k in ['category', 'product', 'type', 'source', 'direction'])), None)
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    
    total_rev = float(df[rev_col].sum()) if rev_col else (float(df[numeric_cols[0]].sum()) if numeric_cols else 0.0)
    total_exp = float(df[exp_col].sum()) if exp_col else 0.0
    net_profit = total_rev - total_exp
    
    kpis = {
        "Total Revenue / Inflow": round(total_rev, 2),
        "Total Expense / Outflow": round(total_exp, 2),
        "Net Profit / Variance": round(net_profit, 2),
        "Total Records": len(df)
    }
    
    charts = {}
    if rev_col and cat_col:
        fig_bar = px.bar(
            df.groupby(cat_col)[rev_col].sum().reset_index(),
            x=cat_col, y=rev_col,
            title=f"Distribution of {rev_col} by {cat_col}",
            template="plotly_dark",
            color=cat_col
        )
        charts["Category Bar Chart"] = fig_bar
        
    if rev_col and date_col:
        df_sorted = df.copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')
        df_sorted = df_sorted.dropna(subset=[date_col]).sort_values(by=date_col)
        fig_line = px.line(
            df_sorted.groupby(date_col)[rev_col].sum().reset_index(),
            x=date_col, y=rev_col,
            title=f"Trend Over Time ({rev_col})",
            template="plotly_dark"
        )
        charts["Time Series Line Chart"] = fig_line

    if numeric_cols:
        fig_hist = px.histogram(
            df, x=numeric_cols[0],
            title=f"Distribution Frequency ({numeric_cols[0]})",
            template="plotly_dark"
        )
        charts["Histogram Frequency"] = fig_hist
        
    return kpis, charts
