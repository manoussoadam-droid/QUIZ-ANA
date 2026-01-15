# QCM depuis image

Prototype Streamlit qui transforme une image de question en QCM interactif.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cle API

Definis la variable d'environnement `GEMINI_API_KEY` avant de lancer:

```bash
export GEMINI_API_KEY="AIza..."
```

## Lancer l'app

```bash
streamlit run app.py
```

## Notes

- Modele utilise: `models/gemini-2.0-flash` (modifiable via `GEMINI_MODEL`)
- Si l'image est floue, la qualite des resultats baisse.
