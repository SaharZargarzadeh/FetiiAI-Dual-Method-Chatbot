
import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import os, re

from data_utils import load_data_from_xlsx, last_n_days
from rb_engine import rule_based_router, gpt_intent_assist, answer_large_groups_downtown_hours
from rag_index import build_index
from rag_engine import retrieve, synthesize_answer

st.set_page_config(page_title="FetiiAI — Dual-Method Chatbot", page_icon="🚌", layout="wide")

st.title("🚌 FetiiAI — Dual-Method Chatbot")
st.caption("Toggle between Rule-based (offline) and RAG. GPT is optional and used only for intent/synthesis.")

with st.sidebar:
    st.header("Setup")
    xlsx_file = st.file_uploader("Upload Fetii Excel", type=['xlsx'])
    method = st.radio("Methodology", ["Rule-based (offline)", "RAG (embeddings)"], index=0)
    use_gpt = st.toggle("Use GPT (optional)", value=False)
    api_key = st.text_input("OpenAI API Key", type="password") if use_gpt else ""
    st.divider()
    st.markdown("**Examples**")
    st.write("• How many groups went to Moody Center last month?")
    st.write("• What are the top drop-off spots for 18–24 year-olds on Saturday nights?")
    st.write("• When do large groups (6+ riders) typically ride downtown?")
    st.divider()
    st.markdown("**RAG Index (build once)**")
    db_path = st.text_input("SQLite path", "rag_store.sqlite")
    npz_path = st.text_input("Embeddings path", "embeddings.npz")
    if st.button("Build / Rebuild RAG Index"):
        st.session_state['_build_rag'] = True
    else:
        st.session_state['_build_rag'] = st.session_state.get('_build_rag', False)

if not xlsx_file:
    st.info("⬅️ Upload `FetiiAI_Data_Austin.xlsx` to begin.", icon="📄")
    st.stop()

trip, ridermap, demo = load_data_from_xlsx(xlsx_file)

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Trips", f"{len(trip):,}")
with c2: st.metric("Riders", f"{ridermap['User ID'].nunique():,}")
with c3: st.metric("Unique Users", f"{pd.unique(pd.concat([trip['Booking User ID'], ridermap['User ID']])).size:,}")
with c4: st.metric("Date Range", f"{trip['Trip Date and Time'].min().date()} → {trip['Trip Date and Time'].max().date()}")

# Map
st.subheader("Recent drop-offs (last 7 days)")
last7 = last_n_days(trip, 7)
st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(latitude=30.2672, longitude=-97.7431, zoom=11),
    layers=[pdk.Layer(
        "ScatterplotLayer",
        data=last7.rename(columns={'Drop Off Latitude':'lat','Drop Off Longitude':'lon'}),
        get_position='[lon, lat]',
        get_radius=30,
        pickable=True
    )],
    tooltip={"text": "{[Pick Up Address]} → {[Drop Off Address]} \n{[Trip Date and Time]}"}
))

st.subheader("Ask a question")
q = st.text_input("Type your question:", "")

openai_client = None
if use_gpt and api_key:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY",""))
    except Exception as e:
        st.warning(f"OpenAI client not available: {e}")

# Build RAG index if requested
if st.session_state.get('_build_rag', False) and method.startswith("RAG"):
    with st.spinner("Building RAG index (embeddings)..."):
        n = build_index(db_path, npz_path, trip)
        st.success(f"Built RAG index with {n} rows.")
        st.session_state['_build_rag'] = False

if not q:
    st.stop()

if method.startswith("Rule-based"):
    res = rule_based_router(trip, ridermap, demo, q)
    if res is None and openai_client is not None:
        try:
            intent_json = gpt_intent_assist(openai_client, q)
            if "moody_last_month" in intent_json:
                from rb_engine import answer_moody_last_month
                res = answer_moody_last_month(trip)
            elif "top_drop_18_24_sat_night" in intent_json:
                from rb_engine import answer_top_drop_18_24_sat_night
                res = answer_top_drop_18_24_sat_night(trip, ridermap, demo)
            elif "large_groups_downtown_hours" in intent_json:
                from rb_engine import answer_large_groups_downtown_hours
                res = answer_large_groups_downtown_hours(trip)
            elif "loc_last_30" in intent_json:
                m = re.search(r'"location"\s*:\s*"([^"]+)"', intent_json)
                loc = m.group(1) if m else ""
                if loc:
                    from rb_engine import generic_loc_count_last30
                    res = generic_loc_count_last30(trip, loc)
        except Exception as e:
            st.warning(f"Intent mapping failed: {e}")

    if res is None:
        st.error("Sorry, I couldn't interpret that. Try rephrasing or use one of the example questions.")
    else:
        st.success(res['answer'])
        df = res.get('table')
        if df is not None and len(df):
            if set(df.columns) >= {'Drop Off Address','rides'}:
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('rides:Q', title='Rides'),
                    y=alt.Y('Drop Off Address:N', sort='-x', title='Drop-off spot')
                ).properties(height=320)
                st.altair_chart(chart, use_container_width=True)
            elif set(df.columns) >= {'hour','rides'}:
                chart = alt.Chart(df).mark_line(point=True).encode(
                    x=alt.X('hour:O', title='Hour of day'),
                    y=alt.Y('rides:Q', title='Rides')
                ).properties(height=320)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(df.head(100))
else:
    if not (os.path.exists(db_path) and os.path.exists(npz_path)):
        st.info("Build the RAG index first (see sidebar).")
        st.stop()

    with st.spinner("Retrieving relevant trips..."):
        ret = retrieve(db_path, npz_path, q, top_k=20)
    if ret.empty:
        st.error("No relevant rows found.")
        st.stop()

    st.write("**Top matches (by semantic similarity):**")
    st.dataframe(ret.head(20))

    answer = None
    if openai_client is not None:
        with st.spinner("Synthesizing answer with GPT..."):
            try:
                answer = synthesize_answer(openai_client, q, ret)
            except Exception as e:
                st.warning(f"GPT synthesis failed: {e}")

    if answer:
        st.success(answer)
    else:
        st.info("No GPT synthesis — showing quick aggregates:")
        addr_counts = ret['drop_addr'].value_counts().reset_index()
        addr_counts.columns = ['Drop Off Address','rides']
        chart = alt.Chart(addr_counts.head(10)).mark_bar().encode(
            x=alt.X('rides:Q', title='Rides'),
            y=alt.Y('Drop Off Address:N', sort='-x')
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(addr_counts.head(20))
