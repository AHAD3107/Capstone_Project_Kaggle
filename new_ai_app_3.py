import os
import streamlit as st
import sqlite3
import json
import pandas as pd
from typing import TypedDict, Annotated, List, Dict, Any
from datetime import datetime
from PIL import Image
# --- CLOUD LLM IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
# --- LOCAL LLM IMPORTS ---
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import base64
from io import BytesIO
import requests
import re  # Import regex for robust parsing

# ---  CONFIGURATION  ---
# Paste your Google Gemini API Key here to bypass the sidebar input.
HARDCODED_KEY = "" 

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Enterprise Invoice Agent", layout="wide", page_icon="🧾")

# --- DATABASE ---
DB_NAME = "invoice_memory.db"

def init_db():
    """
    Initializes the database and handles schema migrations safely.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. Create base table
    c.execute('''CREATE TABLE IF NOT EXISTS invoices
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  vendor TEXT,
                  date TEXT,
                  amount REAL,
                  currency TEXT,
                  converted_amount REAL,
                  base_currency TEXT,
                  category TEXT,
                  anomaly_flag BOOLEAN,
                  processed_at TIMESTAMP)''')
    
    # 2. Schema Migration (Add columns if missing)
    columns_to_add = [
        ("currency", "TEXT"),
        ("converted_amount", "REAL"),
        ("base_currency", "TEXT")
    ]
    
    for col_name, col_type in columns_to_add:
        try:
            c.execute(f"ALTER TABLE invoices ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass # Column already exists
    
    conn.commit()
    conn.close()

init_db()

# ---  CLOUD LLM (Gemini - Vision) ---
def get_llm():
    if HARDCODED_KEY:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=HARDCODED_KEY)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Google API Key missing for Vision tasks.")
        st.stop()
    
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=api_key)

# ---  LOCAL LLM (Gemma - Reasoning) ---
def get_local_llm():
    # Make sure 'ollama run gemma3:1b' is active
    return Ollama(model="gemma3:1b", temperature=0)

# ---  LIVE FX ENGINE ---
@st.cache_data(ttl=3600) 
def fetch_exchange_rates(base_currency: str):
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url)
        data = response.json()
        return data.get("rates", {})
    except Exception:
        # Fallback rates
        return {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.50}

def calculate_fx(amount: float, from_curr: str, rates_map: Dict[str, float]) -> float:
    from_curr = from_curr.upper()
    # Simple direct multiplication if rate exists (Target = Amount * Rate)
    if from_curr in rates_map:
        return round(amount * rates_map[from_curr], 2)
    return amount

# ---  UI THEME ---
def apply_cyber_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        .stApp { background-color: #0E0E12; font-family: 'Inter', sans-serif; }
        section[data-testid="stSidebar"] { background-color: #15151C; border-right: 1px solid rgba(255, 255, 255, 0.05); }
        .cyber-card {
            background: rgba(30, 30, 40, 0.6);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }
        h1, h2, h3 { color: #FFFFFF; }
        h1 span { background: linear-gradient(90deg, #00F0FF, #6C5DD3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status-pill { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
        .status-success { background: rgba(0, 240, 255, 0.15); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.3); }
        .status-danger { background: rgba(255, 0, 85, 0.15); color: #FF0055; border: 1px solid rgba(255, 0, 85, 0.3); }
        div.stButton > button {
            background: linear-gradient(90deg, #6C5DD3, #8B5CF6);
            border: none; color: white; font-weight: 600;
            padding: 0.6rem 1.5rem; border-radius: 8px;
        }
        .stJson, .stDataFrame { background: #1A1A23 !important; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); }
    </style>
    """, unsafe_allow_html=True)

apply_cyber_theme()

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    image_data: Any              
    extracted_data: Dict         
    validation_logs: List[str]   
    category: str                
    anomaly_detected: bool       
    confidence_score: float      
    email_draft: str             
    human_review_needed: bool    
    anomaly_threshold: float
    base_currency: str          
    converted_amount: float     
    fx_rates: Dict[str, float]

# --- TOOLS & NODES ---

def extraction_node(state: AgentState):
    """Node 1: Extract (Uses CLOUD Gemini for Vision)."""
    llm = get_llm() 
    
    # Improved Prompt: Ask for a simple text representation first to ground the model
    prompt = """
    You are an expert OCR system. 
    1. First, read the invoice image and identify the Total Amount, Vendor, Date, and Currency.
    2. Then, output a VALID JSON object with these exact keys:
       - "vendor_name": string
       - "invoice_date": string (YYYY-MM-DD)
       - "total_amount": number (no currency symbols)
       - "currency_code": string (USD, INR, etc.)
       - "line_items": list of objects [{"description": str, "quantity": num, "unit_price": num, "total": num}]
    
    IMPORTANT: 
    - If you cannot find a value, use null. 
    - Do NOT output markdown code blocks (like ```json). Just the raw JSON string.
    - Ensure "total_amount" is a raw number (e.g., 1250.00), NOT a string like "$1,250.00".
    """
    
    msg = [
        SystemMessage(content=prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Extract data from this invoice image."},
            {"type": "image_url", "image_url": state["image_data"]}
        ])
    ]
    
    try:
        response = llm.invoke(msg)
        raw_text = response.content
        print(f"DEBUG: Raw LLM Response: {raw_text[:500]}...") # Log first 500 chars
        
        # Robust cleanup
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        
        # Attempt parsing
        try:
            data = json.loads(clean_json)
        except json.JSONDecodeError:
            # Fallback 1: Try to find the first '{' and last '}'
            start = raw_text.find('{')
            end = raw_text.rfind('}') + 1
            if start != -1 and end != -1:
                clean_json = raw_text[start:end]
                data = json.loads(clean_json)
            else:
                # Fallback 2: Regex extraction for critical fields if JSON fails completely
                print("⚠️ JSON parsing failed. Attempting Regex fallback.")
                import re
                vendor = re.search(r'"vendor_name":\s*"([^"]+)"', raw_text)
                amount = re.search(r'"total_amount":\s*([\d\.]+)', raw_text)
                currency = re.search(r'"currency_code":\s*"([^"]+)"', raw_text)
                
                data = {
                    "vendor_name": vendor.group(1) if vendor else "Unknown",
                    "total_amount": float(amount.group(1)) if amount else 0.0,
                    "currency_code": currency.group(1) if currency else "USD",
                    "line_items": [],
                    "error": "JSON Parse Error - Regex Fallback Used"
                }

        # Final Validation & Type Casting
        if "total_amount" in data:
            if isinstance(data["total_amount"], str):
                # Clean "$1,000.00" -> 1000.00
                clean_amt = str(data["total_amount"]).replace(',', '').replace('$', '').replace('₹', '').strip()
                try:
                    data["total_amount"] = float(clean_amt)
                except:
                    data["total_amount"] = 0.0
        else:
            data["total_amount"] = 0.0
            
        # Ensure other required keys exist
        for key in ["vendor_name", "currency_code", "line_items"]:
            if key not in data: data[key] = None
            
    except Exception as e:
        print(f"❌ Critical Extraction Error: {str(e)}")
        data = {
            "vendor_name": "Error extracting data",
            "total_amount": 0.0,
            "currency_code": "USD",
            "line_items": [],
            "error": f"Extraction Failed: {str(e)}"
        }
        
    return {
        "extracted_data": data, 
        "confidence_score": 1.0 if not data.get("error") else 0.0,
        "validation_logs": ["☁️ Gemini: Extraction complete."]
    }

def validation_categorization_node(state: AgentState):
    """Node 2: Categorize (Uses LOCAL Gemma)."""
    local_llm = get_local_llm()
    data = state["extracted_data"]
    logs = state["validation_logs"]
    
    vendor = data.get("vendor_name", "Unknown")
    # Robustly handle amount (strip commas if string)
    raw_amount = data.get("total_amount", 0.0)
    if isinstance(raw_amount, str):
        try: raw_amount = float(raw_amount.replace(',', ''))
        except: raw_amount = 0.0
            
    raw_currency = data.get("currency_code", "USD")
    base_curr = state.get("base_currency", "USD")
    
    # FX Logic
    rates = state.get("fx_rates", {})
    converted_val = calculate_fx(raw_amount, raw_currency, rates)
    logs.append(f"💱 FX: {raw_currency} {raw_amount} -> {base_curr} {converted_val}")

    # Threshold Check
    threshold = state.get("anomaly_threshold", 5000.0) 
    
    # Vendor Check
    known_vendors = ["Amazon", "Microsoft", "Uber", "Delta", "Walmart", "Google", "Apple", "Flipkart"]
    is_legit = any(v.lower() in str(vendor).lower() for v in known_vendors)
    if is_legit: logs.append(f"✅ Vendor verified: {vendor}")
    else: logs.append(f"⚠️ Unknown Vendor: {vendor}")

    # Categorization
    cat_prompt = f"""
    Categorize expense. Vendor: '{vendor}'. Items: {data.get('line_items', [])}.
    Categories: [Travel, Office Supplies, Software, Meals, Utilities, Consulting].
    Return ONLY the category name.
    """
    try:
        category = local_llm.invoke(cat_prompt).strip()
    except:
        category = "Uncategorized"
    
    # Anomaly
    anomaly = False
    if converted_val > threshold:
        anomaly = True
        logs.append(f"❗ Policy Violation: {base_curr} {converted_val} > {threshold}")
    else:
         logs.append(f"✅ Amount within limit")
    
    return {
        "category": category,
        "anomaly_detected": anomaly,
        "validation_logs": logs,
        "converted_amount": converted_val
    }

def quality_evaluation_node(state: AgentState):
    """Node 3: Quality Gate."""
    anomaly = state["anomaly_detected"]
    return {"human_review_needed": anomaly}

def reporting_node(state: AgentState):
    """Node 4: Report (Uses LOCAL Gemma)."""
    local_llm = get_local_llm()
    data = state["extracted_data"]
    logs = state["validation_logs"]
    
    vendor = data.get("vendor_name", "Unknown")
    raw_amount = data.get("total_amount", 0.0)
    date = data.get("invoice_date", datetime.now().strftime("%Y-%m-%d"))
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id FROM invoices WHERE vendor=? AND amount=? AND date=?", (vendor, raw_amount, date))
    exists = c.fetchone()
    
    status_msg = ""
    if not exists:
        c.execute("""INSERT INTO invoices 
                     (vendor, date, amount, currency, converted_amount, base_currency, category, anomaly_flag, processed_at) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (vendor, date, raw_amount, data.get("currency_code"), state["converted_amount"], state["base_currency"], state["category"], state["anomaly_detected"], datetime.now()))
        conn.commit()
        status_msg = "Saved to Ledger"
    else:
        status_msg = "Duplicate - Skipped DB"
    conn.close()
    logs.append(status_msg)
    
    email_prompt = f"""
    Draft email to Finance. Subject: Invoice - {vendor}.
    Body: Amount {raw_amount}. Category {state['category']}. Status: {'Anomaly' if state['anomaly_detected'] else 'Approved'}.
    """
    try:
        email = local_llm.invoke(email_prompt)
    except:
        email = "Email generation failed."
    
    return {"email_draft": email, "validation_logs": logs}

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("extractor", extraction_node)
workflow.add_node("validator", validation_categorization_node)
workflow.add_node("evaluator", quality_evaluation_node)
workflow.add_node("reporter", reporting_node)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "validator")
workflow.add_edge("validator", "evaluator")
workflow.add_edge("evaluator", "reporter")
workflow.add_edge("reporter", END)

app_graph = workflow.compile()

# --- STREAMLIT UI ---
def main():
    st.sidebar.markdown("### ⚡ HYBRID AGENT")
    st.sidebar.markdown("---")
    
    if HARDCODED_KEY:
        st.sidebar.success("☁️ Gemini Vision: Ready")
    elif os.environ.get("GOOGLE_API_KEY"):
        st.sidebar.success("☁️ Gemini Vision: Ready")
    else:
        st.sidebar.warning("☁️ Gemini Vision: Key Missing")
    st.sidebar.success("🏠 Gemma 3 (Local): Ready")
    
    # Settings
    st.sidebar.markdown("#### ⚙️ Settings")
    base_curr = st.sidebar.selectbox("Base Currency", ["USD", "INR", "EUR", "GBP"], index=1)
    
    live_rates = fetch_exchange_rates("USD") 
    final_rates_map = {}
    currencies_to_edit = ["USD", "EUR", "GBP", "INR"]
    if base_curr in currencies_to_edit: currencies_to_edit.remove(base_curr)
    
    with st.sidebar.expander("💱 FX Rates", expanded=False):
        for curr in currencies_to_edit:
            usd_to_base = live_rates.get(base_curr, 1.0)
            usd_to_target = live_rates.get(curr, 1.0)
            default_rate = usd_to_base / usd_to_target
            custom_rate = st.number_input(f"1 {curr} =", value=float(round(default_rate, 4)), format="%.4f")
            final_rates_map[curr] = custom_rate
        final_rates_map[base_curr] = 1.0
        final_rates_map["USD"] = live_rates.get(base_curr, 1.0)

    threshold_val = st.sidebar.slider(f"Alert Limit ({base_curr})", 100, 500000, 50000 if base_curr == "INR" else 5000, 100)

    # Dashboard
    st.markdown("<h1><span>Hybrid Invoice Agent</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888; margin-bottom: 30px;'>Cloud Vision (Gemini) + Local Reasoning (Gemma)</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if not uploaded_file:
        st.markdown("<div class='cyber-card' style='text-align: center; color: #888;'>📥 Drop Invoice Here</div>", unsafe_allow_html=True)

    if uploaded_file:
        col_img, col_actions = st.columns([1, 2])
        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption="Source", use_column_width=True)
        
        with col_actions:
            if st.button("▶ START HYBRID WORKFLOW"):
                if not HARDCODED_KEY and not os.environ.get("GOOGLE_API_KEY"):
                    st.error("⚠️ API Key missing.")
                    return

                with st.spinner("🤖 Processing..."):
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    img_url = f"data:image/jpeg;base64,{img_str}"
                    
                    inputs = {
                        "image_data": img_url, 
                        "validation_logs": [],
                        "anomaly_threshold": float(threshold_val),
                        "base_currency": base_curr,
                        "fx_rates": final_rates_map
                    }
                    result = app_graph.invoke(inputs)
                    
                    # Dashboard Layout
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"<div class='cyber-card'><div style='color:#888; font-size:0.8rem;'>VENDOR</div><div style='font-size:1.2rem; font-weight:bold;'>{result['extracted_data'].get('vendor_name', 'Unknown')}</div></div>", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"<div class='cyber-card'><div style='color:#888; font-size:0.8rem;'>TOTAL ({base_curr})</div><div style='font-size:1.2rem; font-weight:bold; color:#00F0FF;'>{base_curr} {result['converted_amount']:,.2f}</div></div>", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"<div class='cyber-card'><div style='color:#888; font-size:0.8rem;'>CATEGORY</div><div style='font-size:1.2rem; font-weight:bold; color:#6C5DD3;'>{result['category']}</div></div>", unsafe_allow_html=True)

                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown("### 📋 Line Items")
                        items = result['extracted_data'].get('line_items', [])
                        if items: st.dataframe(pd.DataFrame(items), use_container_width=True)
                        else: st.info("No items detected.")
                    with c2:
                        st.markdown("### 🔍 Hybrid Audit")
                        for log in result["validation_logs"]:
                            if "✅" in log: st.markdown(f"<div class='status-pill status-success'>{log}</div>", unsafe_allow_html=True)
                            elif "❗" in log: st.markdown(f"<div class='status-pill status-danger'>{log}</div>", unsafe_allow_html=True)
                            else: st.markdown(f"<div style='color:#ccc; font-size:0.8rem; padding:2px;'>• {log}</div>", unsafe_allow_html=True)

                    with st.expander("📧 Gemma-Drafted Email", expanded=False):
                        st.code(result["email_draft"], language="text")

    st.markdown("---")
    st.subheader("🗄️ Enterprise Memory")
    if st.button("Load Ledger"):
        conn = sqlite3.connect(DB_NAME)
        try:
            df = pd.read_sql_query("SELECT * FROM invoices ORDER BY id DESC", conn)
            if not df.empty:
                df['Status'] = df['anomaly_flag'].apply(lambda x: '🔴 Flagged' if x else '🟢 Approved')
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Ledger is empty.")
        except Exception as e:
            st.error(f"Error reading ledger: {e}")
        conn.close()

if __name__ == "__main__":
    main()