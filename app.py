import streamlit as st
from query import load_qa_chain

st.set_page_config(page_title="📊 Customer Chatbot")
st.title("🤖 EasyTech Customer Support Bot")

qa_chain = load_qa_chain(model_name="phi")  # Change here to 'mistral' if needed

user_query = st.text_input("Ask me anything from the knowledge base:")

if user_query:
    with st.spinner("Searching..."):
        result = qa_chain(user_query)
        st.success(result['result'])

        with st.expander("Show source context"):
            for doc in result["source_documents"]:
                st.markdown(f"- {doc.page_content[:500]}...")
