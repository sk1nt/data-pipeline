from dotenv import load_dotenv
from pathlib import Path


def load_env_file():
    """Load environment from repo root .env (regardless of cwd)."""
    root_env = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=root_env)
