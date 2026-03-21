import os
from google import genai

client = genai.Client(api_key=os.environ.get("GCP_API_KEY"))
for m in client.models.list():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)
