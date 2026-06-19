import requests
import json

api_key = "csk-n94j36ew6vp5p3538kwvpnvpyj8tvvrvdvnc2hwthh25fhmk"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 1. Test listing models
try:
    r = requests.get("https://api.cerebras.ai/v1/models", headers=headers)
    print("Models status:", r.status_code)
    print("Models response:", r.text[:1000])
except Exception as e:
    print("Models failed:", e)

# 2. Test chat completions
payload = {
    "model": "llama-3.3-70b",
    "messages": [
        {"role": "user", "content": "Say hello"}
    ]
}

try:
    r = requests.post("https://api.cerebras.ai/v1/chat/completions", headers=headers, json=payload)
    print("Chat status:", r.status_code)
    print("Chat response:", r.text)
except Exception as e:
    print("Chat failed:", e)
