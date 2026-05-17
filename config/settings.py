# config/settings.py
import os
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "kumagenFinal"

FASTTEXT_MODEL_PATH = "cc.fr.300.bin"

"""
config/settings.py
Central configuration for the hybrid translation system.
Set LLM_BACKEND and optionally LLM_MODEL here or via environment variables.
"""


GEMINI_API_KEY    = os.getenv('GEMINI_API_KEY',    'AIzaSyDWh8nAn23eFdE3KC8EwCCORpjAKuPpEmA')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_BACKEND = os.getenv('LLM_BACKEND', 'ollama')
LLM_MODEL   = os.getenv('LLM_MODEL',   'qwen2.5:3b')

