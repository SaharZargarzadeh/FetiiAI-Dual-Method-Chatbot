
import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def _load_embeddings(npz_path):
    arr = np.load(npz_path)
    return arr['embeddings']

def _cosine_sim(a, b):
    return (b @ a)

def retrieve(db_path:str, npz_path:str, query:str, top_k:int=15):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    qv = model.encode([query], normalize_embeddings=True)[0]

    M = _load_embeddings(npz_path)
    sims = _cosine_sim(qv, M)
    idx = np.argsort(-sims)[:top_k]
    scores = sims[idx]

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT rowid AS row_idx, * FROM trips WHERE rowid IN ({','.join([str(i+1) for i in idx])})", conn)
    conn.close()

    order = {int(i)+1: float(s) for i, s in zip(idx, scores)}
    df['_score'] = df['row_idx'].map(order)
    df = df.sort_values('_score', ascending=False)
    return df[['row_idx','_score','trip_id','drop_addr','ts','total_passengers','drop_lat','drop_lon']]

def synthesize_answer(openai_client, question:str, retrieved_df:pd.DataFrame):
    facts = []
    for _, r in retrieved_df.head(10).iterrows():
        facts.append(f"- Trip {int(r['trip_id'])}: {r['ts']}, drop: {r['drop_addr']}, pax={int(r['total_passengers'])}")
    context = "\n".join(facts) if facts else "No matching rows."

    SYSTEM = ("You are a data analyst for a rideshare company. "
              "Answer strictly using the provided trip facts; if insufficient, say so. "
              "Prefer numeric counts and clear lists. No speculation.")
    user = f"Question: {question}\nFacts:\n{context}\n\nGive a concise answer."
    chat = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        temperature=0
    )
    return chat.choices[0].message.content.strip()
