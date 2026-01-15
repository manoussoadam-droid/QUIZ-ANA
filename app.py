import json
import os
import re
from typing import Any, Dict, Optional

import google.generativeai as genai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv


load_dotenv()


def get_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_prompt() -> str:
    return (
        "Tu es un assistant qui transforme une image de QCM en JSON strict. "
        "Lis l'image et retourne uniquement un JSON valide sans texte autour. "
        "Schema exact:\n"
        "{\n"
        '  "question": "string",\n'
        '  "options": ["string", "string", "string", "string"],\n'
        '  "correct_index": 0,\n'
        '  "explanation": "string"\n'
        "}\n"
        "Regles: options doit avoir 2 a 6 elements. correct_index est l'index "
        "dans options. Si tu n'es pas sur, propose la meilleure estimation."
    )


def generate_mcq(image: Image.Image) -> Dict[str, Any]:
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)
    prompt = build_prompt()
    response = model.generate_content([prompt, image])
    data = extract_json(response.text or "")
    if not data:
        raise ValueError("JSON non valide retourne par le modele.")
    return data


st.set_page_config(page_title="QCM depuis image", page_icon="❓", layout="centered")
st.title("QCM depuis image")
st.caption("Upload une image de question et obtiens un QCM interactif.")

api_key = get_api_key()
if not api_key:
    st.error(
        "Cle API manquante. Ajoute GEMINI_API_KEY dans ton env ou dans .env."
    )
    st.stop()

genai.configure(api_key=api_key)

uploaded = st.file_uploader("Image de question", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Image chargee", use_container_width=True)

    if st.button("Generer le QCM", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                mcq = generate_mcq(image)
            except Exception as exc:
                st.error(f"Erreur: {exc}")
                st.stop()

        question = mcq.get("question", "").strip()
        options = mcq.get("options", [])
        correct_index = mcq.get("correct_index", None)
        explanation = mcq.get("explanation", "").strip()

        if not question or not isinstance(options, list) or len(options) < 2:
            st.error("Extraction invalide. Reessaie avec une image plus claire.")
            st.stop()

        st.subheader("Question")
        st.write(question)

        choice = st.radio("Choisis une reponse", options, index=None)

        if st.button("Verifier la reponse"):
            if choice is None:
                st.warning("Choisis une option.")
            else:
                user_index = options.index(choice)
                if user_index == correct_index:
                    st.success("Bonne reponse !")
                else:
                    correct_label = (
                        options[correct_index]
                        if isinstance(correct_index, int)
                        and 0 <= correct_index < len(options)
                        else "inconnue"
                    )
                    st.error(f"Mauvaise reponse. Bonne reponse: {correct_label}")
                if explanation:
                    st.info(explanation)

