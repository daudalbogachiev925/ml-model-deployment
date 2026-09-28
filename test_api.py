"""Тест API."""
import requests

r = requests.post("http://localhost:8000/predict",
                  json={"features": [50.0, 3.0]})
print(r.json())
