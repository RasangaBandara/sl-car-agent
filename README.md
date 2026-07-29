🚗 Sri Lanka Vehicle Valuation & RAG Compliance Agent

An intelligent, multi-agent AI system designed to solve real-world used vehicle valuation, market comparison, and import regulation compliance challenges for Sri Lankan SMEs and individual buyers.


---

* **GitHub Repository:** [https://github.com/RasangaBandara/sl-car-agent](https://github.com/RasangaBandara/sl-car-agent)

---

📌 Project Overview
Navigating the Sri Lankan automotive market requires balancing fluctuating local resale prices against complex government import duties, luxury tax brackets, and vehicle depreciation rules. 

This application deploys a specialized multi-agent workflow that:
1. Filters and analyzes local market listing datasets (`car_price_dataset.csv`) across Brands, Models, YOM, and Asking Prices.
2. Queries domain-specific regulatory policy documents via a Retrieval-Augmented Generation (RAG) pipeline.
3. Synthesizes statistical market trends and legal requirements into a structured valuation and negotiation report.

---

🤖 Agentic Design Patterns Implemented

As required by Section 4(a) of the assignment brief, this project implements three distinct agentic design patterns:

1. **Router Pattern (`app.py` -> Line 85–88):**
   * Uses `llama-3.1-8b-instant` on Groq to rapidly classify incoming vehicle evaluation queries and structure downstream execution parameters without high inference latency.
2. **Tool-Use Pattern (`app.py` -> Lines 50–74):**
   * Agents execute targeted local functions:
     * `market_data_tool()`: Queries Pandas DataFrames deterministically to calculate average historical market prices and mileage.
     * `rag_policy_tool()`: Executes semantic similarity searches against indexed Chroma vector store collections.
3. **Orchestrator-Worker / Synthesizer Pattern (`app.py` -> Lines 95–109):**
   * `llama-3.3-70b-versatile` acts as the primary synthesizing orchestrator, combining structured outputs from both the Market Data Agent and Policy RAG Agent into a cohesive advisory report.

---

🔄 Agent-to-Agent Communication Architecture

The system utilizes structured messaging across agents and deterministic tools to complete each evaluation cycle.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / SME
    participant UI as Streamlit App (app.py)
    participant Router as Router Agent (Llama 3.1 8B)
    participant MarketAgent as Market Analyst Agent (Tool)
    participant PolicyAgent as Policy RAG Agent (ChromaDB)
    participant Synth as Synthesizer Agent (Llama 3.3 70B)

    User->>UI: Selects Brand, Model, YOM & Price
    UI->>Router: Prompt: Classify request & validate parameters
    Router-->>UI: Classification & request payload acknowledged
    UI->>MarketAgent: market_data_tool(brand, model, yom)
    MarketAgent-->>UI: Returns JSON (avg price, count, avg mileage)
    UI->>PolicyAgent: rag_policy_tool("tax import laws for brand")
    PolicyAgent-->>UI: Returns retrieved policy context chunks
    UI->>Synth: Prompts with market metrics + retrieved policy text
    Synth-->>UI: Generates structured valuation & regulatory report
    UI-->>User: Displays report (Verdict, Comparison, Warnings, Tips)
