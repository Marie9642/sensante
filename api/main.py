from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib          # ← ajouter
import numpy as np     # ← ajouter
from fastapi.middleware.cors import CORSMiddleware   # ← AJOUTER

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
else:
    print("ATTENTION : clé non trouvée. /explain désactivé.")


class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str

class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"

# --- Schémas Pydantic (déjà faits à l'étape 3) ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

#Autoriser les requetes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # en dev : accepte tout
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ÉTAPE 4 : Charger le modèle ici ↓ ---
print("Chargement du modèle...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modèle chargé : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")
# -----------------------------------------

# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # 1. Encoder le sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    # 2. Encoder la région
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}"
        )

    # 3. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 4. Prédire
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    # 5. Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 6. Message de recommandation
    messages = {
        "palu": "Suspicion de paludisme. Consultez un médecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandés.",
        "typh": "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain": "Pas de pathologie détectée. Continuez à surveiller."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )

SYSTEM_PROMPT = """Tu es un assistant médical sénégalais.
Explique le diagnostic en français simple comme un médecin.
Sois rassurant mais recommande toujours une consultation.
Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-même."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    if not groq_client:
        return ExplainOutput(
            explication="Service indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic : {data.diagnostic} (probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur : {str(e)}"

    return ExplainOutput(explication=explication)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

