import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.skip(
    reason=(
        "The Streamlit dashboard in `notebooks/app.py` relies on a live "
        "`streamlit run` session and session_state, which cannot be safely "
        "initialized under pytest. Run the app with `streamlit run` instead."
    )
)
def test_app_import_smoke(monkeypatch):
    """
    Placeholder test documenting that the Streamlit app is exercised via
    manual `streamlit run` rather than automated pytest. Kept here so the
    intent is visible in test reports without breaking the suite.
    """
    pass


