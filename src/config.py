# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
EMB_DIR = MODELS_DIR / "embeddings"
CLUST_DIR = MODELS_DIR / "clusters"

DOCS_DIR = PROJECT_ROOT / "docs"
IMG_DIR = DOCS_DIR / "img"

RANDOM_STATE = 42
DEFAULT_K = 30