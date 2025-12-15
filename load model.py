import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd





# -------------------------------------------------------------------------
# 1. Load model & tokenizer
# -------------------------------------------------------------------------
MODEL_PATH = "./best_model"   # <-- mets ici ton chemin du modèle sauvegardé

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

@st.cache_data
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model




# @st.cache_resource
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
#     return tokenizer, model
# tokenizer, model = load_model()

def visualization():
    st.title("this is a model")
    st.success("this title is affiched")
    st.subheader("----------------------")
    
