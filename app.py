import json
import os
import re
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
import streamlit as st
from PIL import Image
from dotenv import load_dotenv


load_dotenv()


def get_api_key() -> Optional[str]:
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
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


def generate_mcq(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
    prompt = build_prompt()
    response = genai_client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        ],
    )
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

genai_client = genai.Client(api_key=api_key)

uploaded = st.file_uploader("Image de question", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Image chargee", width="stretch")
    uploaded_bytes = uploaded.getvalue()
    uploaded_type = uploaded.type or "image/png"

    if st.button("Generer le QCM", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                mcq = generate_mcq(uploaded_bytes, uploaded_type)
            except Exception as exc:
                message = str(exc)
                if "429" in message or "Quota exceeded" in message:
                    st.error(
                        "Quota Gemini depasse ou non active. Verifie que la "
                        "Generative Language API est active, que la cle est "
                        "valide, et que ton projet a un quota actif. "
                        "Si le quota free tier est a 0, il faut utiliser un "
                        "autre projet/cle ou activer la facturation."
                    )
                else:
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

