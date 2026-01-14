import streamlit as st
import os
from backend_agent import get_generator_agent, evaluate_relevance

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Scientist Judge", layout="wide")
st.title("👨‍🔬 AI Scientist Finder & Judge")

# --- SIDEBAR: API KEY ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password")
    
    st.info("Get your key from [OpenRouter](https://openrouter.ai/).")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

# --- SESSION STATE INITIALIZATION ---
if "generated_output" not in st.session_state:
    st.session_state["generated_output"] = ""
if "current_topic" not in st.session_state:
    st.session_state["current_topic"] = ""

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1])

# === SECTION 1: GENERATION ===
with col1:
    st.subheader("Step 1: Generate List")
    topic_input = st.text_input("Enter Research Topic", "Generative Adversarial Networks")
    
    if st.button("🔍 Find Scientists"):
        if not api_key:
            st.error("Please enter your OpenRouter API Key in the sidebar.")
        else:
            with st.spinner(f"Searching for top researchers in {topic_input}..."):
                try:
                    # 1. Get Agent
                    agent = get_generator_agent(api_key)
                    
                    # 2. Run Agent
                    response = agent.run(f"Find top 3 researchers in {topic_input}")
                    
                    # 3. Update State
                    st.session_state["generated_output"] = response.content
                    st.session_state["current_topic"] = topic_input
                    st.success("Generation Complete!")
                    
                except Exception as e:
                    st.error(f"Generation Error: {e}")

# === SECTION 2: EDIT & EVALUATE ===
with col2:
    st.subheader("Step 2: Edit & Evaluate")
    st.caption("You can edit the list below (e.g., add Cricketers) to test the Judge.")
    
    #  - Conceptual trigger, not inserted as per instructions
    
    # 4. Editable Text Area
    # We bind the value to session_state, but allow the user to change it.
    final_text_to_judge = st.text_area(
        "Scientist List (Editable)", 
        value=st.session_state["generated_output"],
        height=300
    )

    if st.button("⚖️ Evaluate with DeepSeek"):
        if not api_key:
            st.error("API Key missing.")
        elif not final_text_to_judge.strip():
            st.warning("The list is empty. Please generate or write something first.")
        else:
            with st.spinner("DeepSeek is judging validity..."):
                try:
                    # 5. Call Backend Evaluation
                    # We use the ORIGINAL topic, but the EDITED text
                    score, reason = evaluate_relevance(
                        api_key=api_key,
                        topic=st.session_state["current_topic"] or topic_input,
                        actual_output=final_text_to_judge
                    )
                    
                    # 6. Display Results
                    st.divider()
                    st.subheader("Judge Report")
                    
                    if score > 0.7:
                        st.success(f"✅ PASSED (Score: {score})")
                    else:
                        st.error(f"❌ FAILED (Score: {score})")
                        
                    st.info(f"**Reasoning:**\n\n{reason}")

                except Exception as e:
                    st.error(f"Evaluation Failed: {e}")
