import sys
from pathlib import Path

# Make src/ importable as top-level packages (config, services, etc.)
# This matches the production runtime path setup.
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
