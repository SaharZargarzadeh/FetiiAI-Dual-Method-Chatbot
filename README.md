# 🚌 FetiiAI — Dual-Method Chatbot (Rule-based + RAG)

An interactive chatbot for exploring **Fetii’s Austin group rideshare data** in natural language.  
Built for the **FetiiAI Hackathon** to help Fetii understand group movement patterns, popular destinations, and rider behavior — turning raw trip data into actionable insights.

This is designed as *“ChatGPT for Fetii group movement trends in Austin.”*

---

## 🎯 Project Goal

Make Fetii’s rideshare data easy to explore with plain-English questions like:

- How many groups went to Moody Center last month?
- What are the top drop-off spots for 18–24 year-olds on Saturday nights?
- When do large groups (6+ riders) typically ride downtown?

The goal is to support **traffic optimization, event planning, and operational decision-making** by surfacing trends in group mobility.

---

## ⚙️ Two Methodologies

### 1) Rule-based (Offline)  
- Deterministic Pandas queries over the dataset  
- Fast, reliable, and works fully offline  
- Optional GPT assist to map natural-language questions to supported intents  

**Best for:** Known business questions, dashboards, and guaranteed accuracy.

---

### 2) RAG — Retrieval-Augmented Generation  
- Trip records embedded using **SentenceTransformers (MiniLM-L6-v2)**  
- Stored locally with SQLite + NumPy  
- User query → semantic search → retrieve most relevant trips  
- Optional GPT to synthesize a concise natural-language answer  

**Best for:** Flexible, open-ended questions and discovery of hidden patterns.

---

## 🧠 How It Works (High-Level)

1. Load Fetii’s Austin dataset (Trip, Rider, and Demographics tables)  
2. User selects Rule-based or RAG mode  
3. Rule-based → predefined filters + aggregations  
4. RAG → vector similarity search over all trips  
5. (Optional) GPT summarizes retrieved results into clear answers  
6. Streamlit UI displays metrics, maps, tables, and responses  

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then:

Upload FetiiAI_Data_Austin.xlsx

Choose Rule-based or RAG

(Optional) Enable GPT for better natural-language answers

---

## 🛠️ Tech Stack
Python + Streamlit — interactive demo UI

Pandas — data processing and analytics

SentenceTransformers (MiniLM-L6-v2) — lightweight semantic embeddings

SQLite + NumPy — local vector storage

OpenAI GPT (optional) — intent parsing and answer synthesis

---

## 🌱 Future Improvements
- Deploy to Hugging Face Spaces or Replit for one-click online access

- Add more predefined business dashboards (Rule-based)

- Integrate live data feeds instead of static Excel

- Scale to multiple Fetii cities and markets

- Add cohort analysis and event-based trend detection

---
## 🙌 Author
Built by Sahar Zargarzadeh for the FetiiAI Hackathon.
PhD researcher focused on applied AI, data-driven systems, and real-world ML deployment.
