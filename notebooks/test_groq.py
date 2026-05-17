import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : clé non trouvée")
    exit()

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "Tu es un assistant médical sénégalais. Réponds en français simple. Maximum 3 phrases."
        },
        {
            "role": "user",
            "content": "Quels sont les symptômes du paludisme ?"
        }
    ],
    max_tokens=200,
    temperature=0.3
)

print(response.choices[0].message.content)
print(f"Tokens utilisés : {response.usage.total_tokens}")