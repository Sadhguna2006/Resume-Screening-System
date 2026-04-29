import streamlit as st
import fitz  # PyMuPDF
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load trained models
# -------------------------
best_log_model = joblib.load("log_model.pkl")
best_svm_model = joblib.load("svm_model.pkl")
le = joblib.load("label_encoder.pkl")

# -------------------------
# NLP Setup
# -------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(["the", "and", "is", "in", "to", "of", "for", "a", "an"])
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Utility functions
# -------------------------
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Resume Role Predictor", layout="centered")
st.title("🤖 AI Resume Role Predictor")
st.markdown("Upload your resume (PDF) to predict the most suitable role using **Logistic Regression** and **SVM models**.")

uploaded_file = st.file_uploader("📁 Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting and analyzing your resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned = clean_text(resume_text)
        X_new = embed_model.encode([cleaned])

        # Predictions
        log_pred_class = best_log_model.predict(X_new)[0]
        svm_pred_class = best_svm_model.predict(X_new)[0]

        log_role_name = le.inverse_transform([log_pred_class])[0]
        svm_role_name = le.inverse_transform([svm_pred_class])[0]

        # Display Results
        st.subheader("🔹 Model Predictions")
        st.write(f"🧮 **Logistic Regression:** {log_role_name}")
        st.write(f"⚙️ **SVM (LinearSVC):** {svm_role_name}")

        if log_role_name == svm_role_name:
            st.success(f"✅ Both models agree: **{log_role_name}**")
        else:
            st.warning("⚠️ The models disagree on prediction.")

        # Optional: Ranking of top 3 roles by cosine similarity
        all_roles = le.classes_
        role_embeddings = embed_model.encode(all_roles)
        sims = cosine_similarity(X_new, role_embeddings)[0]
        top3_idx = np.argsort(sims)[-3:][::-1]
        top_roles = [(all_roles[i], sims[i]) for i in top3_idx]

        st.subheader("🏆 Top 3 Matching Roles")
        for i, (role, score) in enumerate(top_roles, 1):
            st.write(f"**{i}. {role}** — Similarity: {score:.3f}")

        # Bar chart visualization
        fig, ax = plt.subplots()
        roles, scores = zip(*top_roles)
        ax.barh(roles[::-1], scores[::-1], color="#4CAF50")
        ax.set_xlabel("Similarity Score")
        ax.set_title("Top 3 Matching Roles")
        st.pyplot(fig)
