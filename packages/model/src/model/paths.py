from pathlib import Path
from typing import Final

MODEL_PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR: Final[Path] = MODEL_PACKAGE_ROOT / "artifacts"
