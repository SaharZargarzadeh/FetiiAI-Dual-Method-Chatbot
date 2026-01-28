
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

def build_index(db_path:str, npz_path:str, trip):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trips (
    row_id INTEGER PRIMARY KEY,
    trip_id INTEGER,
    booking_user_id INTEGER,
    pickup_lat REAL, pickup_lon REAL,
    drop_lat REAL, drop_lon REAL,
    pickup_addr TEXT, drop_addr TEXT,
    ts TEXT,
    total_passengers INTEGER
)""")
    cur.execute("DELETE FROM trips")

    def make_text(row):
        return (f"trip {int(row['Trip ID'])} | {row['Trip Date and Time']} | "
                f"pickup: {row['Pick Up Address']} ({row['Pick Up Latitude']:.6f},{row['Pick Up Longitude']:.6f}) | "
                f"drop: {row['Drop Off Address']} ({row['Drop Off Latitude']:.6f},{row['Drop Off Longitude']:.6f}) | "
                f"passengers: {int(row['Total Passengers'])} | "
                f"dow: {row['dow']} hour:{int(row['hour'])}")

    texts = []
    for _, r in trip.iterrows():
        cur.execute("""INSERT INTO trips (trip_id, booking_user_id, pickup_lat, pickup_lon, drop_lat, drop_lon, pickup_addr, drop_addr, ts, total_passengers)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (int(r['Trip ID']), int(r['Booking User ID']), float(r['Pick Up Latitude']), float(r['Pick Up Longitude']),
       float(r['Drop Off Latitude']), float(r['Drop Off Longitude']), str(r['Pick Up Address']), str(r['Drop Off Address']),
       str(r['Trip Date and Time']), int(r['Total Passengers'])))
        texts.append(make_text(r))

    conn.commit()
    conn.close()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.savez_compressed(npz_path, embeddings=embs, count=len(texts))
    return len(texts)
