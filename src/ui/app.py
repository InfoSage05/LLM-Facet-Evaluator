import streamlit as st
import httpx
import json
import pandas as pd

import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Facet Evaluation Demo", layout="wide")

st.title("Conversation Facet Evaluation System")
st.markdown("Evaluate conversations automatically against dynamic facets.")

@st.cache_data
def load_facets():
    try:
        response = httpx.get(f"{API_URL}/facets")
        return response.json()
    except:
        return []

facets = load_facets()
facet_groups = sorted(list(set(f.get("group", "Other") for f in facets))) if facets else []

st.sidebar.header("Configuration")
selected_group = st.sidebar.selectbox("Filter Facets by Group", ["All"] + facet_groups)

if facets:
    if selected_group != "All":
        display_facets = [f for f in facets if f.get("group") == selected_group]
    else:
        display_facets = facets
    
    selected_facets = st.sidebar.multiselect(
        "Select Facets to Evaluate (leave empty for random 5)",
        options=[f["facet_id"] for f in display_facets],
        format_func=lambda x: next((f["name"] for f in display_facets if f["facet_id"] == x), x)
    )
    if not selected_facets:
        selected_facets = [f["facet_id"] for f in display_facets[:5]]
else:
    st.sidebar.error("Could not load facets from API.")
    selected_facets = []

st.subheader("1. Setup Conversation")
upload_method = st.radio("Provide Conversation", ["Upload JSONL", "Input Text"])

conv_to_score = None

if upload_method == "Upload JSONL":
    uploaded_file = st.file_uploader("Upload conversations.jsonl", type=["jsonl"])
    if uploaded_file:
        lines = uploaded_file.getvalue().decode('utf-8').splitlines()
        if lines:
            conv = json.loads(lines[0])
            conv_to_score = conv
            st.success(f"Loaded '{conv.get('scenario')}' conversation.")
elif upload_method == "Input Text":
    user_text = st.text_area("User Turn", "Hello, I need help with my internet.")
    agent_text = st.text_area("Agent Turn", "I can help with that. Have you tried turning it off and on?")
    
    if user_text and agent_text:
        conv_to_score = {
            "conversation_id": "manual_1",
            "scenario": "Manual Input",
            "description": "User input text",
            "turns": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": agent_text}
            ],
            "metadata": {}
        }

if conv_to_score:
    with st.expander("View Conversation Details"):
        for turn in conv_to_score["turns"]:
            role = "🗣️ **User:**" if turn["role"] == "user" else "🤖 **Agent:**"
            st.markdown(f"{role} {turn['content']}")
            
    if st.button("Evaluate Conversation", type="primary"):
        with st.spinner("Evaluating via Scoring Engine..."):
            try:
                payload = {
                    "conversation": conv_to_score,
                    "facet_ids": selected_facets
                }
                
                res = httpx.post(f"{API_URL}/score", json=payload, timeout=30.0)
                res.raise_for_status()
                data = res.json()
                
                st.subheader("Evaluation Results")
                evals = data.get("evaluations", {})
                
                result_data = []
                for fid, eval_res in evals.items():
                    fname = next((f["name"] for f in facets if f["facet_id"] == fid), fid)
                    result_data.append({
                        "Facet": fname,
                        "Score (1-5)": eval_res["score"],
                        "Confidence": f"{eval_res['confidence']:.2f}",
                        "Rationale": eval_res["rationale"]
                    })
                
                df = pd.DataFrame(result_data)
                st.dataframe(df, width="stretch")
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
