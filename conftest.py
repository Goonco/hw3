# Makes `import posttrain` and `import tests.helpers` work when running `pytest` from this folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
