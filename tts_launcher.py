from pyngrok import ngrok
import os

port = 8501
public_url = ngrok.connect(port)
print("🌍 Public link to share:", public_url)
os.system(f"streamlit run tts_app.py --server.port {port}")
