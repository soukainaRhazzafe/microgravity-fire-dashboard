import streamlit as st

st.set_page_config(
    page_title="Microgravity Fire Dashboard",
    layout="wide",
)

st.title("Microgravity Fire Dashboard")
st.write("First prototype: testing the interface and deployment.")

st.sidebar.header("Explore")

category = st.sidebar.selectbox(
    "Choose a research area",
    ["Prevention", "Detection", "Suppression"],
)

st.subheader(category)
st.info("Placeholder only — no NASA data or AI is connected yet.")

question = st.text_input("What would you like to investigate?")

if st.button("Test interaction"):
    if question.strip():
        st.success(f"Your question was received: {question}")
    else:
        st.warning("Enter a question first.")
