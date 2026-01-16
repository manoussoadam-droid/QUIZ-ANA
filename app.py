import hashlib
import io
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
        '      "correct_indices": [0],\n'
        '      "explanation": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Regles: options doit avoir 2 a 6 elements. correct_indices contient "
        "un ou plusieurs index dans options. Si une question manque d'options, "
        "propose la meilleure estimation. Si une seule question, renvoie un "
        "tableau avec 1 element."
    )


def normalize_mcq_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "questions" in payload and isinstance(payload["questions"], list):
        return payload
    if "question" in payload:
        return {"questions": [payload]}
    return {"questions": []}


def normalize_correct_indices(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        indices: list[int] = []
        for item in value:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str) and item.isdigit():
                indices.append(int(item))
        return indices
    if isinstance(value, str) and value.isdigit():
        return [int(value)]
    return []


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

if "files" not in st.session_state:
    st.session_state["files"] = []
if "selected_id" not in st.session_state:
    st.session_state["selected_id"] = None

uploaded_files = st.file_uploader(
    "Image de question", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    existing_ids = {item["id"] for item in st.session_state["files"]}
    for upl in uploaded_files:
        file_bytes = upl.getvalue()
        file_id = hashlib.sha256(file_bytes).hexdigest()
        if file_id in existing_ids:
            continue
        st.session_state["files"].append(
            {
                "id": file_id,
                "name": upl.name,
                "bytes": file_bytes,
                "type": upl.type or "image/png",
                "mcq_list": None,
            }
        )
        existing_ids.add(file_id)
        if st.session_state["selected_id"] is None:
            st.session_state["selected_id"] = file_id

files = st.session_state["files"]
if files:
    st.caption("Fichiers charges")
    cols = st.columns(len(files))
    for idx, item in enumerate(files):
        with cols[idx]:
            thumb = Image.open(io.BytesIO(item["bytes"]))
            st.image(thumb, width=120)
            if st.button("Voir", key=f"select_{item['id']}"):
                st.session_state["selected_id"] = item["id"]
                st.rerun()
            if st.button("Supprimer", key=f"delete_{item['id']}"):
                st.session_state["files"] = [
                    f for f in st.session_state["files"] if f["id"] != item["id"]
                ]
                if st.session_state["selected_id"] == item["id"]:
                    st.session_state["selected_id"] = (
                        st.session_state["files"][0]["id"]
                        if st.session_state["files"]
                        else None
                    )
                st.rerun()

selected = next(
    (f for f in files if f["id"] == st.session_state["selected_id"]), None
)

if selected:
    image = Image.open(io.BytesIO(selected["bytes"]))
    st.image(image, caption=selected["name"], width="stretch")

    col_generate, col_next = st.columns(2)
    with col_generate:
        generate_clicked = st.button("Generer le QCM", type="primary")
    with col_next:
        next_clicked = st.button("Nouvelle question")

    if next_clicked:
        selected["mcq_list"] = None
        generate_clicked = True

    if generate_clicked:
        with st.spinner("Analyse en cours..."):
            try:
                payload = generate_mcq(selected["bytes"], selected["type"])
                items = []
                for item in payload.get("questions", []):
                    items.append(
                        {
                            "question": (item.get("question") or "").strip(),
                            "options": item.get("options") or [],
                            "correct_indices": normalize_correct_indices(
                                item.get("correct_indices", item.get("correct_index"))
                            ),
                            "explanation": (item.get("explanation") or "").strip(),
                            "choice": None,
                            "checked": False,
                        }
                    )
                selected["mcq_list"] = items
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

    if selected.get("mcq_list") is not None:
        if not selected["mcq_list"]:
            st.error("Aucune question detectee. Reessaie avec une image plus claire.")
            st.stop()

        for idx, item in enumerate(selected["mcq_list"], start=1):
            question = item["question"]
            options = item["options"]
            correct_indices = item["correct_indices"]
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

            use_multi = len(correct_indices) > 1
            if use_multi:
                choice_multi = st.multiselect(
                    "Choisis une ou plusieurs reponses",
                    options,
                    key=f"choice_multi_{selected['id']}_{idx}",
                )
                item["choice"] = choice_multi
            else:
                choice_single = st.radio(
                    "Choisis une reponse",
                    options,
                    index=None,
                    key=f"choice_{selected['id']}_{idx}",
                )
                item["choice"] = choice_single

            if st.button("Verifier la reponse", key=f"check_{selected['id']}_{idx}"):
                selected_choice = item["choice"]
                if not selected_choice:
                    st.warning("Choisis une option.")
                else:
                    if isinstance(selected_choice, list):
                        user_indices = {
                            options.index(val)
                            for val in selected_choice
                            if val in options
                        }
                    else:
                        user_indices = {options.index(selected_choice)}

                    correct_set = {
                        idx_val
                        for idx_val in correct_indices
                        if 0 <= idx_val < len(options)
                    }

                    if correct_set and user_indices == correct_set:
                        st.success("Bonne reponse !")
                    else:
                        if correct_set:
                            labels = ", ".join(options[i] for i in sorted(correct_set))
                        else:
                            labels = "inconnue"
                        st.error(
                            f"Mauvaise reponse. Bonne reponse: {labels}"
                        )
                    if explanation:
                        st.info(explanation)

