import requests
import os

url = "https://api.cartesia.ai/voices/"

cartesia_api_key = os.environ.get("CARTESIA_API_KEY")

if not cartesia_api_key:
    raise ValueError("CARTESIA_API_KEY environment variable is not set")

headers = {
    "Cartesia-Version": "2024-06-10",
    "X-API-Key": cartesia_api_key
}

response = requests.get(url, headers=headers)
print(response.json()[2])
# [print("ID: " + voice["id"] + ", Name: " + voice["name"] + ", Description: " + voice["description"] + "\n") for voice in response.json()]