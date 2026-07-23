import streamlit as st  # type: ignore[import]
import pandas as pd
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="SL Vehicle Valuation Agent", page_icon="🚗", layout="wide")
st.title("🚗 Sri Lanka Vehicle Valuation & Compliance Agentic System")
st.caption("IT41043 Intelligent Systems Assignment Project | Real-World SME Valuation & RAG Compliance")

# Retrieve API keys from secrets or sidebar
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.header("🔑 Configuration")
    if not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    else:
        st.success("Groq API Key loaded securely!")

# Build / Load Vector DB
@st.cache_resource
def init_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists("./chroma_db"):
        docs = []
        if os.path.exists("policy_docs"):
            for fname in os.listdir("policy_docs"):
                fpath = os.path.join("policy_docs", fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r") as f:
                        docs.append(Document(page_content=f.read(), metadata={"source": fname}))
        return Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

vector_db = init_vector_store()

# Load Dataset
@st.cache_data
def load_car_data():
    if os.path.exists("car_price_dataset.csv"):
        return pd.read_csv("car_price_dataset.csv")
    return None

df = load_car_data()

if df is not None:
    st.sidebar.success(f"Loaded {len(df):,} listings!")
    with st.expander("🔍 Preview Car Dataset"):
        st.dataframe(df.head(10))

# Tools
def market_data_tool(brand: str, model: str, yom: int):
    if df is None:
        return {"error": "Dataset missing"}
    filtered = df[(df['Brand'].str.upper() == brand.upper()) & 
                  (df['Model'].str.upper() == model.upper()) & 
                  (df['YOM'] == yom)]
    if filtered.empty:
        filtered = df[(df['Brand'].str.upper() == brand.upper()) & (df['YOM'] == yom)]
    if filtered.empty:
        return {"found": False, "message": f"No data for {yom} {brand} {model}"}
    
    return {
        "found": True,
        "count": len(filtered),
        "avg_price_lakhs": round(filtered['Price'].mean(), 2),
        "avg_mileage_km": round(filtered['Millage(KM)'].mean(), 0)
    }

def rag_policy_tool(query: str):
    results = vector_db.similarity_search(query, k=2)
    return [r.page_content.strip() for r in results]

# Streamlit UI Controls
if df is not None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        user_brand = st.selectbox("Brand", sorted(df['Brand'].unique()))
    with c2:
        user_model = st.selectbox("Model", sorted(df[df['Brand'] == user_brand]['Model'].unique()))
    with c3:
        user_yom = st.number_input("YOM", min_value=2000, max_value=2026, value=2018)
    with c4:
        user_price = st.number_input("Asking Price (Lakhs)", min_value=1.0, value=95.0)

    if st.button("🚀 Analyze Deal"):
        if not groq_api_key:
            st.error("Missing Groq API Key!")
        else:
            with st.status("Running Multi-Agent Workflow...", expanded=True) as status:
                # 1. Router Agent
                st.write("🔄 **Router Agent** (Groq Llama 3.1 8B)...")
                router = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
                route_res = router.invoke([HumanMessage(content=f"Classify valuation request for {user_brand} {user_model}")])
                
                # 2. Market Data Tool
                st.write("📊 **Market Analyst Agent** querying dataset...")
                m_data = market_data_tool(user_brand, user_model, user_yom)
                
                # 3. RAG Tool
                st.write("📚 **Policy Agent** retrieving regulations...")
                p_data = rag_policy_tool(f"tax import laws for {user_brand}")
                
                # 4. Synthesizer
                st.write("🧠 **Synthesizer Agent** (Groq Llama 3.3 70B) generating advisory report...")
                synth = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
                prompt = f"""
                Analyze this used car deal in Sri Lanka:
                - Target: {user_yom} {user_brand} {user_model} asking {user_price} Lakhs LKR
                - Market Stats: {json.dumps(m_data)}
                - Policy Guidance: {p_data}
                
                Provide a structured report with: Verdict, Market Comparison, Regulatory Warnings, and 2 Negotiation Tips.
                """
                report = synth.invoke([HumanMessage(content=prompt)])
                status.update(label="Complete!", state="complete")
                
            st.markdown("### 📋 Final Advisory Report")
            st.markdown(report.content)