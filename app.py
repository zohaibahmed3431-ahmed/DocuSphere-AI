from __future__ import annotations

import re
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_uploaded_file
from src.retrieval import HybridRetriever
from src.llm import GeminiAssistant
from src.security import sanitize_question
from src.citations import format_sources
from src.csv_analytics import analyze_csv, dataframe_to_csv_bytes, is_full_details_request, full_file_report
from src.tabular import read_tabular, dataframe_context
from src.utils import is_code_file
from src.pdf_export import build_pdf_report
from src.reconciliation import looks_like_reconciliation_request, auto_select_two_ledgers, reconcile_tables, reconciliation_excel
from src.router import route, is_graph_picture_request, is_image_generation, should_use_vision

load_dotenv()

st.set_page_config(page_title="DocuSphere AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.5rem;padding-bottom:5rem}
[data-testid="stSidebar"]{min-width:300px;max-width:340px}
.hero{padding:1.35rem 1.5rem;border:1px solid rgba(128,128,128,.20);border-radius:20px;background:linear-gradient(135deg,rgba(90,80,180,.12),rgba(30,120,180,.08));margin-bottom:1rem}
.hero h1{margin:0 0 .35rem 0;font-size:2.1rem}.hero p{margin:0;opacity:.72}
.answer-card{padding:.4rem 0}
.small-muted{opacity:.68;font-size:.88rem}
[data-testid="stChatMessage"]{border-radius:16px}
</style>
""", unsafe_allow_html=True)

SUPPORTED=["pdf","docx","pptx","xlsx","xls","csv","txt","md","log","py","java","cpp","c","h","hpp","js","jsx","ts","tsx","html","css","scss","sql","json","xml","yaml","yml","toml","ini","cfg","conf","sh","bat","ps1","php","go","rs","rb","kt","kts","swift","r","m","vue","svelte","tex","jpg","jpeg","png","webp","gif","bmp","tif","tiff","avif"]

if "records" not in st.session_state: st.session_state.records=[]
if "chat" not in st.session_state: st.session_state.chat=[]
if "retriever" not in st.session_state: st.session_state.retriever=HybridRetriever()
if "assistant" not in st.session_state: st.session_state.assistant=GeminiAssistant()
if "tabular" not in st.session_state: st.session_state.tabular={}

st.markdown('<div class="hero"><h1>🧠 DocuSphere AI</h1><p>Your AI workspace for conversation, documents, data, charts, code and current information.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Workspace")
    st.caption("AI automatically decides whether your question needs file knowledge, exact data analysis, general AI, or web information.")
    uploads=st.file_uploader("Add files", type=SUPPORTED, accept_multiple_files=True)
    processed_names = {r["source"] for r in st.session_state.records}
    pending_uploads = [u for u in uploads if u.name not in processed_names] if uploads else []
    if pending_uploads:
        st.warning(
            f"⚠️ **{len(pending_uploads)} file(s) uploaded but not processed.** "
            "Click **Process files** so DocuSphere can read, index, and use them."
        )
    if st.button("➕ Process files", use_container_width=True, disabled=not pending_uploads):
        progress=st.progress(0); status=st.empty(); added=[]; tab={}
        for i,u in enumerate(pending_uploads):
            status.write(f"Processing `{u.name}`…")
            try:
                added.extend(ingest_uploaded_file(u))
                tab.update(read_tabular(u))
            except Exception as e:
                st.error(f"{u.name}: {e}")
            progress.progress((i+1)/len(pending_uploads))
        if added:
            names={r["source"] for r in added}
            st.session_state.records=[r for r in st.session_state.records if r["source"] not in names]+added
            st.session_state.tabular.update(tab)
            st.session_state.retriever.build(st.session_state.records)
            st.rerun()

    if st.session_state.records:
        st.divider(); st.subheader("📚 Connected files")
        counts={}
        for r in st.session_state.records: counts[r["source"]]=counts.get(r["source"],0)+1
        for name,count in counts.items(): st.write(f"📄 **{name}** · {count} chunks")
        for label,df in st.session_state.tabular.items():
            st.caption(f"📊 `{label}` · {len(df):,} rows × {len(df.columns)} columns")
        if st.button("🗑️ Remove all files", use_container_width=True):
            st.session_state.records=[]; st.session_state.tabular={}; st.session_state.retriever.clear(); st.rerun()

    st.divider()
    st.subheader("⚙️ AI behavior")
    st.caption("No mode selection is required. Ask naturally; DocuSphere chooses the appropriate tool automatically.")
    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.chat=[]; st.rerun()

if uploads:
    processed_names = {r["source"] for r in st.session_state.records}
    pending_names = [u.name for u in uploads if u.name not in processed_names]
    if pending_names:
        st.warning(
            "⚠️ **File processing reminder:** " + ", ".join(f"`{n}`" for n in pending_names) +
            " — click **Process files** in the sidebar before asking questions about these files."
        )

if not st.session_state.records:
    st.info("No file is required. Start a normal AI conversation below, or add files when you want file-specific answers.")
else:
    st.success(f"{len({r['source'] for r in st.session_state.records})} file(s) connected. You can still ask completely general questions.")

for i,item in enumerate(st.session_state.chat):
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("chart") is not None: st.plotly_chart(item["chart"], use_container_width=True, key=f"history-chart-{i}")
        if item.get("table") is not None and not item["table"].empty: st.dataframe(item["table"], use_container_width=True, hide_index=True)
        if item.get("download_data") is not None:
            st.download_button("⬇️ Download file", item["download_data"], item.get("download_name","docusphere_export.csv"), item.get("download_mime", "text/csv"), key=f"history-download-{i}")
        if item.get("sources"): st.markdown(format_sources(item["sources"]))


def is_code_question(q: str) -> bool:
    ql = re.sub(r"\s+", " ", (q or "").lower().strip())
    return any(x in ql for x in (
        "code", "program", "script", "function", "class", "method", "variable", "array",
        "bug", "error", "exception", "debug", "fix this", "modify this", "change this code",
        "run this", "compile", "import", "syntax", "algorithm", "python", "java", "c++", "javascript",
    ))

def build_code_context(results):
    if not any(is_code_file(r.get("source","")) for r in st.session_state.records): return ""
    by={}
    for r in st.session_state.records:
        if is_code_file(r.get("source","")): by.setdefault(r["source"],[]).append(r["text"])
    return "\n\n".join(f"[FULL CODE FILE: {src}]\n"+"\n\n".join(parts)[:60000] for src,parts in by.items())



def is_pdf_export_request(q: str) -> bool:
    ql = re.sub(r"\s+", " ", q.lower().strip())
    pdf_words = ("pdf", "pdf file", "pdf report", "pdf bana", "pdf banao", "pdf bana do", "pdf mein", "pdf me")
    export_words = ("make", "create", "generate", "give", "send", "download", "export", "save", "bana", "banao", "chahiye", "do", "de do", "nikal")
    return any(x in ql for x in pdf_words) and (any(x in ql for x in export_words) or "as pdf" in ql or "in pdf" in ql or "pdf report" in ql)

def should_use_web(q: str) -> bool:
    ql = q.lower()
    triggers = (
        "latest", "today", "current", "recent", "news", "search web", "search online",
        "internet", "price now", "right now", "as of today", "what happened",
        "who is the current", "live update", "currently", "this morning", "this week",
    )
    return any(x in ql for x in triggers)


def should_generate_image(q: str) -> bool:
    """Detect natural-language requests for an actual generated image."""
    ql = re.sub(r"\s+", " ", q.lower().strip())
    # Explicit image/picture/visual creation requests.
    if re.search(r"\b(generate|create|make|draw|render|design|produce)\b.{0,60}\b(image|picture|photo|illustration|graphic|poster|artwork|visual|tasveer|pic)\b", ql):
        return True
    if re.search(r"\b(image|picture|photo|illustration|graphic|poster|artwork|visual|tasveer|pic)\b.{0,50}\b(banao|bana do|generate|create|make|draw|dikhao|do)\b", ql):
        return True
    return any(x in ql for x in (
        "tasveer banao", "tasveer bana do", "tasveer dikhao", "tasveer chahiye",
        "image banao", "image bana do", "image chahiye",
        "picture banao", "picture bana do", "picture chahiye",
        "photo banao", "photo bana do", "photo chahiye",
        "pic banao", "pic bana do", "pic chahiye",
        "image generate karo", "picture generate karo", "photo generate karo",
        "ek image banao", "ek picture banao", "mujhe ek picture chahiye",
        "mujhe ek pic chahiye", "mujhe picture chahiye", "mujhe image chahiye",
        "show me a picture", "show me an image", "show me a photo",
    ))

def is_graph_picture_request(q: str) -> bool:
    """Detect requests for a graph/chart as an image/visual."""
    ql = re.sub(r"\s+", " ", q.lower().strip())
    has_graph = any(x in ql for x in ("graph", "chart", "plot", "bar chart", "line chart", "pie chart"))
    has_picture = any(x in ql for x in ("picture", "image", "photo", "pic", "tasveer", "visual"))
    has_create = any(x in ql for x in ("banao", "bana do", "generate", "create", "make", "draw", "render", "dikhao"))
    return has_graph and (has_picture or has_create)


def _tabular_target(query: str, frames: dict) -> list[tuple[str, pd.DataFrame]]:
    """Rank tables by explicit filename/column mentions, then by relevance."""
    if not frames:
        return []
    q = query.lower()
    scored = []
    for label, df in frames.items():
        score = 0
        ll = label.lower()
        if ll in q:
            score += 1000
        stem = re.sub(r"\.[^.]+$", "", ll)
        if stem and stem in q:
            score += 800
        for col in df.columns:
            c = str(col).lower()
            if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", q):
                score += 80
        scored.append((score, label, df))
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    # If the user explicitly names a file/column, use that target first. Otherwise
    # let each table try deterministic analysis; this avoids silently choosing a random file.
    return [(label, df) for _, label, df in scored]

def chart_to_png(fig) -> bytes | None:
    """Render the same Plotly trace data as a standalone PNG without a browser dependency."""
    try:
        out = BytesIO()
        mpl = plt.figure(figsize=(14, 8), dpi=160)
        ax = mpl.gca()
        traces = getattr(fig, "data", ()) or ()
        title = getattr(getattr(fig, "layout", None), "title", None)
        title_text = getattr(title, "text", "") if title else ""
        for trace in traces:
            kind = str(getattr(trace, "type", ""))
            name = getattr(trace, "name", "") or ""
            x_raw = getattr(trace, "x", ())
            y_raw = getattr(trace, "y", ())
            x = list(x_raw) if x_raw is not None else []
            y = list(y_raw) if y_raw is not None else []
            if kind == "pie":
                labels_raw = getattr(trace, "labels", ())
                values_raw = getattr(trace, "values", ())
                labels = list(labels_raw) if labels_raw is not None else []
                values = list(values_raw) if values_raw is not None else []
                if labels and values:
                    ax.pie(values, labels=labels, autopct="%1.1f%%")
                    ax.axis("equal")
            elif kind == "bar":
                positions = list(range(len(x)))
                width = 0.8 / max(len(traces), 1)
                offset = (list(traces).index(trace) - (len(traces)-1)/2) * width
                ax.bar([v + offset for v in positions], y, width=width, label=name or None)
                ax.set_xticks(positions)
                ax.set_xticklabels([str(v) for v in x], rotation=30, ha="right")
            elif kind in {"scatter", "scattergl"}:
                mode = str(getattr(trace, "mode", "lines"))
                if "markers" in mode and "lines" not in mode:
                    ax.scatter(x, y, label=name or None)
                else:
                    ax.plot(x, y, marker="o" if "markers" in mode else None, label=name or None)
            else:
                # Generic fallback for numeric x/y traces.
                if x and y:
                    ax.plot(x, y, label=name or None)
        if title_text:
            ax.set_title(str(title_text), loc="left", pad=14)
        layout = getattr(fig, "layout", None)
        xaxis = getattr(layout, "xaxis", None) if layout else None
        yaxis = getattr(layout, "yaxis", None) if layout else None
        if xaxis and getattr(xaxis, "title", None):
            ax.set_xlabel(str(getattr(xaxis.title, "text", "") or ""))
        if yaxis and getattr(yaxis, "title", None):
            ax.set_ylabel(str(getattr(yaxis.title, "text", "") or ""))
        if any(getattr(t, "name", "") for t in traces):
            ax.legend(loc="best")
        ax.grid(True, alpha=0.2)
        mpl.tight_layout()
        mpl.savefig(out, format="png", bbox_inches="tight")
        plt.close(mpl)
        return out.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def local_greeting(q: str):
    """Handle simple greetings without spending an API request."""
    ql = re.sub(r"\s+", " ", q.lower().strip())
    greetings = {
        "hi", "hello", "hey", "hiya", "salam", "assalam o alaikum",
        "assalamualaikum", "aoa", "good morning", "good afternoon", "good evening",
    }
    return ql in greetings

def local_small_talk(q: str):
    ql = re.sub(r"\s+", " ", q.lower().strip())
    if ql in {"thanks", "thank you", "thx", "shukriya", "thanks bro", "thank you bro"}:
        return "You're welcome! 😊"
    if ql in {"how are you", "how are you?", "how r u", "kaise ho", "kese ho"}:
        return "Main theek hoon 😊 — batao, kis kaam mein help chahiye?"
    if ql in {"who are you", "what are you", "tum kon ho", "aap kon ho"}:
        return "Main DocuSphere AI hoon — files, data, bank reconciliation, graphs, coding, current information aur general questions handle karne ke liye bana hoon."
    if ql in {"bye", "goodbye", "allah hafiz", "khuda hafiz"}:
        return "Allah Hafiz! 👋 Jab bhi zarurat ho, wapas aa jana."
    if ql in {"fuck you", "f u", "fuck u", "shut up", "bakwas", "pagal ho"}:
        return "Koi baat nahi 😄. Batao kya kaam karna hai, main help karta hoon."
    return None


question=st.chat_input("Ask anything — files, CSV, graphs, coding, projects, current topics, or general questions…")

if question:
    clean=sanitize_question(question)
    if not clean: st.stop()
    st.session_state.chat.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)

    with st.chat_message("assistant"):
        # Simple greetings do not need Gemini, files, retrieval, or quota.
        request_route = route(clean, bool(st.session_state.records), bool(st.session_state.tabular))
        if request_route.kind == "greeting":
            answer = "Hi! 👋 I’m DocuSphere AI. Ask me anything, or upload a file and I’ll analyze it."
            st.markdown(answer)
            st.session_state.chat.append({"role":"assistant","content":answer,"sources":[],"chart":None,"table":None,"download_data":None,"download_name":"docusphere_export.csv","download_mime":"text/csv"})
            st.stop()

        local_answer = local_small_talk(clean)
        if local_answer:
            st.markdown(local_answer)
            st.session_state.chat.append({"role":"assistant","content":local_answer,"sources":[],"chart":None,"table":None,"download_data":None,"download_name":"docusphere_export.csv","download_mime":"text/csv"})
            st.stop()

        # Handle a pure image request before retrieval/OCR/RAG. This prevents an
        # unrelated uploaded document from making a simple image request feel slow.
        if request_route.kind == "image":
            with st.spinner("🎨 Generating your image…"):
                try:
                    image_bytes = st.session_state.assistant.generate_image(clean)
                    st.image(image_bytes, caption="Generated by Gemini", use_container_width=True)
                    st.download_button(
                        "⬇️ Download image", image_bytes, "docusphere_generated.png",
                        "image/png", key=f"generated-image-{len(st.session_state.chat)}"
                    )
                    answer = "I generated the requested image above."
                    st.session_state.chat.append({"role":"assistant","content":answer,"sources":[],"chart":None,"table":None,"download_data":None,"download_name":"docusphere_generated.png"})
                    st.stop()
                except Exception as exc:
                    st.error(f"Image generation failed: {exc}")
                    st.info("The request was stopped safely; DocuSphere will not keep loading indefinitely.")
                    st.stop()

        with st.spinner("Analyzing your request…"):
            # Do not inject unrelated file chunks into ordinary AI questions.
            results=st.session_state.retriever.search(clean, top_k=12) if request_route.kind in {"file", "data", "full_file", "pdf", "reconciliation", "graph_picture"} and st.session_state.records else []
            analysis=None; source_name=None; full_report=None; full_reports=[]
            targets = _tabular_target(clean, st.session_state.tabular) if request_route.kind in {"data", "full_file", "pdf", "graph_picture", "reconciliation"} else []

            # Deterministic bank reconciliation takes priority over generic CSV analysis.
            if request_route.kind == "reconciliation" and looks_like_reconciliation_request(clean, st.session_state.tabular):
                pair = auto_select_two_ledgers(st.session_state.tabular)
                if pair is None:
                    st.warning("I found ledger-like data, but could not safely identify exactly one bank/statement file and one software ledger. No matching was performed to avoid false results.")
                    st.stop()
                (bank_name, bank_df), (software_name, software_df) = pair
                try:
                    reconciliation_result = reconcile_tables(bank_df, software_df, bank_name, software_name)
                except Exception as exc:
                    st.error(f"Reconciliation stopped safely: {exc}")
                    st.stop()
                sm = reconciliation_result.summary
                st.markdown("### 🏦 Bank Reconciliation")
                st.caption(f"Bank/statement: `{bank_name}`  •  Software ledger: `{software_name}`")
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Matched", f"{sm['matched_rows']:,}")
                c2.metric("Bank only", f"{sm['bank_unmatched']:,}")
                c3.metric("Software only", f"{sm['software_unmatched']:,}")
                c4.metric("Total compared", f"{sm['bank_rows'] + sm['software_rows']:,}")
                st.info("Matching is deterministic: source serial/reference values do not have to be equal. Amount is required; unique amount + date + explicit debit/credit direction can safely match, while reference/description and the conservative date window provide stronger identity evidence. Ambiguous duplicates are left unmatched instead of guessed.")
                if not reconciliation_result.matched.empty:
                    st.markdown("#### ✅ Matched transactions — shared reconciliation ID (original ledger serials preserved)")
                    st.dataframe(reconciliation_result.matched, use_container_width=True, hide_index=True)
                else:
                    st.warning("No transactions could be safely matched.")
                if not reconciliation_result.unmatched.empty:
                    st.markdown("#### ⚠️ Unmatched transactions")
                    st.dataframe(reconciliation_result.unmatched, use_container_width=True, hide_index=True)
                # Always provide complete CSVs because CSV has no Excel worksheet-row limit.
                matched_csv = reconciliation_result.matched.to_csv(index=False).encode("utf-8-sig")
                unmatched_csv = reconciliation_result.unmatched.to_csv(index=False).encode("utf-8-sig")
                st.markdown("#### ⬇️ Reconciliation files")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download MATCHED transactions (CSV)", matched_csv,
                        "bank_reconciliation_matched.csv", "text/csv",
                        key=f"recon-matched-csv-{len(st.session_state.chat)}", use_container_width=True
                    )
                with c2:
                    st.download_button(
                        "Download UNMATCHED transactions (CSV)", unmatched_csv,
                        "bank_reconciliation_unmatched.csv", "text/csv",
                        key=f"recon-unmatched-csv-{len(st.session_state.chat)}", use_container_width=True
                    )

                # Also provide separate Excel files when each result fits Excel's hard row limit.
                try:
                    matched_xlsx = reconciliation_excel(reconciliation_result, "matched")
                    unmatched_xlsx = reconciliation_excel(reconciliation_result, "unmatched")
                    c3, c4 = st.columns(2)
                    with c3:
                        st.download_button(
                            "Download MATCHED transactions (Excel)", matched_xlsx,
                            "bank_reconciliation_matched.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"recon-matched-xlsx-{len(st.session_state.chat)}", use_container_width=True
                        )
                    with c4:
                        st.download_button(
                            "Download UNMATCHED transactions (Excel)", unmatched_xlsx,
                            "bank_reconciliation_unmatched.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"recon-unmatched-xlsx-{len(st.session_state.chat)}", use_container_width=True
                        )
                    stored_download = matched_xlsx
                    stored_name = "bank_reconciliation_matched.xlsx"
                    stored_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                except ValueError:
                    st.info("The result is larger than Excel's worksheet limit. The complete CSV downloads above contain all transactions without truncation.")
                    stored_download = None
                    stored_name = "bank_reconciliation_matched.csv"
                    stored_mime = "text/csv"

                answer = (
                    f"I reconciled `{bank_name}` against `{software_name}`. **{sm['matched_rows']:,}** transaction(s) were safely matched. "
                    f"Every matched pair receives a new shared reconciliation serial; the original Bank and Software serial/reference values are preserved separately. "
                    f"**{sm['bank_unmatched']:,}** bank transaction(s) and **{sm['software_unmatched']:,}** software transaction(s) remain unmatched. "
                    "Ambiguous transactions were left unmatched rather than guessed."
                )
                st.markdown(answer)
                st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":None,"table":reconciliation_result.matched,"download_data":stored_download,"download_name":stored_name,"download_mime":stored_mime})
                st.stop()
            if request_route.kind == "full_file":
                for label, df in targets:
                    full_reports.append(full_file_report(df, label))
                if len(full_reports) == 1:
                    full_report = full_reports[0]; source_name = full_report["filename"]
            else:
                for label, df in targets:
                    candidate = analyze_csv(df, clean)
                    if candidate is not None:
                        analysis = candidate; source_name = label; break

            chart=None; table=None; download_data=None; download_name="docusphere_export.csv"

            # PDF export is a first-class output for ANY uploaded file, including CSV/XLSX.
            # It uses deterministic extracted/file data instead of asking Gemini to fabricate a report.
            if request_route.kind == "pdf" and st.session_state.records:
                target_name = source_name or (targets[0][0] if targets else st.session_state.records[0]["source"])
                target_df = st.session_state.tabular.get(target_name)
                target_records = [r for r in st.session_state.records if r.get("source") == target_name]
                report = full_file_report(target_df, target_name) if target_df is not None else None
                pdf_bytes = build_pdf_report(
                    target_name, dataframe=target_df, records=target_records,
                    title=f"DocuSphere AI — {target_name}",
                    summary=(report or {}).get("summary") if report else None,
                    metrics=(report or {}).get("metrics") if report else None,
                    profile=(report or {}).get("profile") if report else None,
                    result_table=(analysis.table if analysis is not None else None),
                )
                safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", target_name.rsplit(".",1)[0]).strip("_") or "docusphere_report"
                st.success(f"PDF report created for `{target_name}`.")
                st.download_button("⬇️ Download PDF report", pdf_bytes, f"{safe_name}_report.pdf", "application/pdf", key=f"pdf-export-{len(st.session_state.chat)}")
                answer = f"I created a PDF report for `{target_name}` using the uploaded file data. The PDF contains the available details, profile, and requested results without inventing missing values."
                st.markdown(answer)
                st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":table,"download_data":pdf_bytes,"download_name":f"{safe_name}_report.pdf","download_mime":"application/pdf"})
                st.stop()

            # A request for a graph picture should use the real uploaded data when a
            # suitable structured table exists. This prevents an AI-drawn decorative
            # graph from replacing an exact business/data chart.
            if request_route.kind == "graph_picture" and analysis and analysis.chart is not None:
                st.markdown("#### 📈 Graph")
                st.plotly_chart(analysis.chart, use_container_width=True)
                chart = analysis.chart
                graph_png = chart_to_png(analysis.chart)
                if graph_png is None:
                    st.warning("The interactive graph is ready, but the standalone PNG renderer failed. The interactive graph remains available above.")
                if graph_png:
                    st.download_button(
                        "⬇️ Download graph as PNG", graph_png, "docusphere_graph.png", "image/png",
                        key=f"graph-png-{len(st.session_state.chat)}"
                    )
                st.success("This graph was generated directly from the uploaded data.")
                answer = "I generated the graph from the uploaded data. The interactive graph and PNG download are above."
                st.markdown(answer)
                st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":table,"download_data":graph_png,"download_name":"docusphere_graph.png","download_mime":"image/png"})
                if results: st.markdown(format_sources(results))
                st.stop()

            # If the user explicitly asks for a graph as a picture but there is no
            # structured-data chart available, route it to image generation rather than
            # pretending a text answer is a graph image.
            if request_route.kind == "graph_picture" and (analysis is None or analysis.chart is None) and not full_reports:
                with st.spinner("🎨 Generating your graph image…"):
                    try:
                        image_bytes = st.session_state.assistant.generate_image(clean)
                        st.image(image_bytes, caption="Generated by Gemini", use_container_width=True)
                        st.download_button(
                            "⬇️ Download image", image_bytes, "docusphere_graph.png",
                            "image/png", key=f"generated-graph-{len(st.session_state.chat)}"
                        )
                        answer = "I generated the requested graph image above."
                        st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":None,"table":None,"download_data":None,"download_name":"docusphere_graph.png"})
                        st.stop()
                    except Exception as exc:
                        st.error(f"Graph image generation failed: {exc}")
                        st.info("The request was stopped safely; DocuSphere will not keep loading indefinitely.")
                        st.stop()

            # Full-file dashboards are deterministic and never depend on Gemini.
            if full_reports:
                for report_idx, report in enumerate(full_reports):
                    st.markdown(f"### 📊 Complete file analysis — `{report['filename']}`")
                    st.info(report.get("summary", "Complete file analysis calculated directly from the uploaded data."))
                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("Rows", f"{report['rows']:,}")
                    m2.metric("Columns", f"{report['columns']:,}")
                    m3.metric("Numeric fields", f"{len(report['numeric_columns']):,}")
                    m4.metric("Date field", report['date_column'] or "Not detected")
                    if not report['metrics'].empty:
                        st.markdown("#### Financial / numeric overview")
                        st.dataframe(report['metrics'], use_container_width=True, hide_index=True)
                        table=report['metrics']
                    for idx,fig in enumerate(report['charts']):
                        st.plotly_chart(fig, use_container_width=True, key=f"full-report-chart-{report_idx}-{idx}-{len(st.session_state.chat)}")
                        if chart is None: chart=fig
                    st.markdown("#### Column profile")
                    st.dataframe(report['profile'], use_container_width=True, hide_index=True)
                    table=report['profile']
                    export_df = st.session_state.tabular[report['filename']]
                    export_data = export_df.to_csv(index=False).encode("utf-8-sig")
                    safe_name=re.sub(r'[^a-zA-Z0-9]+','_',report['filename'].rsplit('.',1)[0]).strip('_')
                    st.download_button("⬇️ Download complete CSV", export_data, f"{safe_name}_full_data.csv", "text/csv", key=f"full-export-{report_idx}-{len(st.session_state.chat)}")
                source_name = ", ".join(r['filename'] for r in full_reports)
                st.success("Complete file details were calculated directly from the uploaded tables. No LLM-generated numbers were used.")

            data_context=""
            if st.session_state.tabular:
                data_context += "\nSTRUCTURED FILE SCHEMAS:\n" + dataframe_context(st.session_state.tabular, max_rows=12, max_cols=35)

            if analysis:
                st.markdown(f"#### {analysis.title}")
                st.markdown(analysis.summary)
                if analysis.chart is not None:
                    st.plotly_chart(analysis.chart, use_container_width=True)
                    chart=analysis.chart
                if analysis.table is not None and not analysis.table.empty:
                    st.dataframe(analysis.table, use_container_width=True, hide_index=True)
                    table=analysis.table
                export_df=analysis.records if analysis.records is not None and not analysis.records.empty else None
                if export_df is not None and (analysis.kind=="export" or "csv" in clean.lower() or "download" in clean.lower()):
                    download_data = export_df.to_csv(index=False).encode("utf-8-sig")
                    safe=re.sub(r"[^a-zA-Z0-9]+","_",analysis.title.lower()).strip("_")[:60]
                    download_name=f"docusphere_{safe or 'export'}.csv"
                    st.download_button(f"⬇️ Download {len(export_df):,} record(s) as CSV", download_data, download_name, "text/csv", key=f"export-{len(st.session_state.chat)}")
                data_context += f"\nEXACT ANALYTICS RESULT FROM {source_name}:\nTitle: {analysis.title}\nSummary: {analysis.summary}\nPeriod: {analysis.period}\nValue column: {analysis.value_column}\nDate column: {analysis.date_column}\nRESULT:\n{analysis.table.head(150).to_csv(index=False) if analysis.table is not None and not analysis.table.empty else '(no aggregate table)'}"

            history=st.session_state.chat[-16:]
            if full_reports:
                answer = (
                    f"I analyzed {len(full_reports)} uploaded table file(s) directly. "
                    "The numeric summary, column profile, professional charts, and CSV downloads are shown above."
                )
            else:
                answer=st.session_state.assistant.answer(
                    question=clean,
                    retrieved=results,
                    conversation=history,
                    web_search=should_use_web(clean) and analysis is None and not full_reports,
                    data_context=data_context,
                    code_context=build_code_context(results) if is_code_question(clean) else "",
                    vision_images=(
                        [(r["mime_type"], r["image_bytes"], r["source"]) for r in st.session_state.records if r.get("image_bytes") and r.get("mime_type")]
                        if should_use_vision(clean)
                        else []
                    ),
                )

        # If exact analytics already gave the requested result, Gemini adds concise interpretation rather than replacing it.
        st.markdown(answer)
        if results: st.markdown(format_sources(results))

    st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":table,"download_data":download_data,"download_name":download_name,"download_mime":"text/csv"})
