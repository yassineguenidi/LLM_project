import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -------------------------------------------------------------------------
# 1. Load model & tokenizer
# -------------------------------------------------------------------------
MODEL_PATH = "./best_model"

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()
st.title("🤖 LLM Judge – Compare Two AI Responses")
st.write("Cette interface utilise ton modèle finetuné pour choisir la meilleure réponse entre A et B.")
st.write("                                                ")




# -------------------------------------------------------------------------
# 2. User Input
# -------------------------------------------------------------------------
st.subheader("📝 Prompt")
prompt = st.text_area("Entre le prompt :", height=120)

st.subheader("💬 Réponse A")
response_a = st.text_area("Texte de la réponse A :", height=150)

st.subheader("💬 Réponse B")
response_b = st.text_area("Texte de la réponse B :", height=150)

# -------------------------------------------------------------------------
# 3. Predict button
# -------------------------------------------------------------------------
if st.button("🔍 Comparer les réponses"):
    if not prompt or not response_a or not response_b:
        st.error("Veuillez remplir tous les champs.")

        with st.spinner("Analyse en cours..."):

            # -------------------------------------------------------------
            # Build the input text exactly as during training
            # -------------------------------------------------------------
            input_text = (
                f"Prompt:\n{prompt}\n\n"
                f"Réponse A:\n{response_a}\n\n"
                f"Réponse B:\n{response_b}\n\n"
                "Laquelle est meilleure ?"
            )

            # Tokenize
            inputs = tokenizer(
                input_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            # Model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).flatten().tolist()

            prob_A = probs[0]
            prob_B = probs[1]

            # -------------------------------------------------------------
            # Display results
            # -------------------------------------------------------------
            st.subheader("📊 Résultats")

            st.write(f"**Probabilité que la meilleure réponse soit A :** `{prob_A:.3f}`")
            st.write(f"**Probabilité que la meilleure réponse soit B :** `{prob_B:.3f}`")

            if prob_A > prob_B:
                st.success("🟢 **Réponse A gagnante !**")
            elif prob_B > prob_A:
                st.success("🔵 **Réponse B gagnante !**")
            else:
                st.warning("⚪ Egalité parfaite entre A et B.")


# Footer
st.markdown("---")
st.write("Créé par Yassine – Modèle finetuné LLM Judge")
