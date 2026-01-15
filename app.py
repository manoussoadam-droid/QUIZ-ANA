import hashlib
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
        "Si l'image contient plusieurs questions, renvoie-les toutes. "
        "Schema exact:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "string",\n'
        '      "options": ["string", "string", "string", "string"],\n'
        '      "correct_index": 0,\n'
        '      "explanation": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Regles: options doit avoir 2 a 6 elements. correct_index est l'index "
        "dans options. Si une question manque d'options, propose la meilleure "
        "estimation. Si une seule question, renvoie un tableau avec 1 element."
    )


def normalize_mcq_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "questions" in payload and isinstance(payload["questions"], list):
        return payload
    if "question" in payload:
        return {"questions": [payload]}
    return {"questions": []}


def generate_mcq(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
    prompt = build_prompt()
    response = genai_client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        ],
    )
    data = extract_json(response.text or "")
    if not data:
        raise ValueError("JSON non valide retourne par le modele.")
    return normalize_mcq_payload(data)


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
    uploaded_id = hashlib.sha256(uploaded_bytes).hexdigest()

    if st.session_state.get("uploaded_id") != uploaded_id:
        st.session_state["uploaded_id"] = uploaded_id
        st.session_state.pop("mcq_list", None)
        st.session_state.pop("checked", None)

    col_generate, col_next = st.columns(2)
    with col_generate:
        generate_clicked = st.button("Generer le QCM", type="primary")
    with col_next:
        next_clicked = st.button("Nouvelle question")

    if next_clicked:
        st.session_state.pop("mcq_list", None)
        st.session_state.pop("checked", None)
        generate_clicked = True

    if generate_clicked:
        with st.spinner("Analyse en cours..."):
            try:
                payload = generate_mcq(uploaded_bytes, uploaded_type)
                items = []
                for item in payload.get("questions", []):
                    items.append(
                        {
                            "question": (item.get("question") or "").strip(),
                            "options": item.get("options") or [],
                            "correct_index": item.get("correct_index"),
                            "explanation": (item.get("explanation") or "").strip(),
                            "choice": None,
                            "checked": False,
                        }
                    )
                st.session_state["mcq_list"] = items
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

    if "mcq_list" in st.session_state:
        if not st.session_state["mcq_list"]:
            st.error("Aucune question detectee. Reessaie avec une image plus claire.")
            st.stop()

        for idx, item in enumerate(st.session_state["mcq_list"], start=1):
            question = item["question"]
            options = item["options"]
            correct_index = item["correct_index"]
            explanation = item["explanation"]

            if (
                not question
                or not isinstance(options, list)
                or len(options) < 2
            ):
                st.warning(
                    f"Question {idx} invalide. Reessaie avec une image plus claire."
                )
                continue

            st.subheader(f"Question {idx}")
            st.write(question)

            choice = st.radio(
                "Choisis une reponse",
                options,
                index=None,
                key=f"choice_{uploaded_id}_{idx}",
            )
            item["choice"] = choice

            if st.button("Verifier la reponse", key=f"check_{uploaded_id}_{idx}"):
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
                        st.error(
                            f"Mauvaise reponse. Bonne reponse: {correct_label}"
                        )
                    if explanation:
                        st.info(explanation)

