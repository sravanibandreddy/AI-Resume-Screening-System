import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening System")

resume = st.text_area("Paste Resume Text")
job_desc = st.text_area("Paste Job Description")

if st.button("Check Match Score"):

    documents = [resume, job_desc]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])

    st.write("Match Score:", round(similarity[0][0]*100,2), "%")