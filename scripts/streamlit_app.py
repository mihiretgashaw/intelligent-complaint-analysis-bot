import streamlit as st
from rag_pipeline import generate_answer


st.title("CrediTrust Complaint Assistant")
question = st.text_input("Ask your question:")

if st.button("Get Answer") and question:
    answer = generate_answer(question)
    st.write("### Answer:")
    st.write(answer)
