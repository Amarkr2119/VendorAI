"""AT-18: app.py smoke test.

If real Streamlit is installed, Streamlit's own AppTest runs the app headlessly.
Otherwise a recording fake of the `streamlit` module drives app.py's real code paths (including a simulated
button click that registers and processes a synthetic application). The fake verifies OUR code, not Streamlit's rendering.
"""
import importlib.util

import pandas  # noqa: F401  (import once, before any fake-module swapping)
import os
import runpy
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@contextmanager
def fake_module(name, module):
    """Swap ONE entry of sys.modules. (patch.dict(sys.modules) would also unload numpy/pandas imported inside the block,
    and numpy refuses to be imported twice in one process.)"""
    missing = object()
    old = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        yield
    finally:
        if old is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


@contextmanager
def temp_env():
    tmp = tempfile.mkdtemp(prefix="vf_app_")
    env = {"DB_PATH": f"{tmp}/app.db", "UPLOAD_DIR": f"{tmp}/uploads", "LOG_FILE": f"{tmp}/app.log", "LLM_ENABLED": "false",
           "GEMINI_API_KEY": "", "GROQ_API_KEY": ""}
    with mock.patch.dict(os.environ, env):
        yield tmp


class FakeStreamlit(types.ModuleType):
    """Records calls; every widget returns a harmless default. `click` makes the named button return True."""

    def __init__(self, click=None):
        super().__init__("streamlit")
        self.calls, self.click = [], click
        self.sidebar = mock.MagicMock()
        self.session_state = {}

    def __getattr__(self, name):                    # any st.xxx(...) is recorded and returns a MagicMock
        if name.startswith("__"):
            raise AttributeError(name)
        m = mock.MagicMock(name=f"st.{name}")
        m.side_effect = lambda *a, **k: (self.calls.append((name, a, k)), mock.MagicMock())[1]
        return m

    # specific behaviours the app relies on
    def cache_resource(self, fn=None, **kw):
        return fn if fn else (lambda f: f)

    def tabs(self, labels):
        return [mock.MagicMock() for _ in labels]

    def columns(self, n, **kw):
        return [mock.MagicMock() for _ in range(n if isinstance(n, int) else len(n))]

    def selectbox(self, label, options, **kw):
        return list(options)[0]

    def button(self, label, **kw):
        self.calls.append(("button", (label,), kw))
        return label == self.click

    def file_uploader(self, *a, **k):
        return None

    def status(self, *a, **k):
        return mock.MagicMock()

    def set_page_config(self, **k):
        self.calls.append(("set_page_config", (), k))

    def stop(self):
        raise SystemExit


class TestAppSmoke(unittest.TestCase):
    @unittest.skipUnless(HAS_STREAMLIT, "real Streamlit not installed in this environment")
    def test_AT18_real_streamlit_apptest_runs_without_exception(self):
        from streamlit.testing.v1 import AppTest
        with temp_env():
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(at.exception, at.exception)
            at.button[0].click().run()
            self.assertFalse(at.exception, at.exception)

    @unittest.skipIf(HAS_STREAMLIT, "real Streamlit available: AppTest test is used instead")
    def test_AT18_fake_streamlit_page_loads_and_shows_connectivity(self):
        with temp_env():
            fake = FakeStreamlit()
            with fake_module("streamlit", fake):
                runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
            names = [c[0] for c in fake.calls]
            self.assertIn("set_page_config", names)
            self.assertIn("title", names)

    @unittest.skipIf(HAS_STREAMLIT, "real Streamlit available: AppTest test is used instead")
    def test_AT18_fake_streamlit_register_and_process_click_runs_the_pipeline(self):
        with temp_env() as tmp:
            fake = FakeStreamlit(click="Register and process")
            with fake_module("streamlit", fake):
                runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
            # the click went through app.py -> services -> pipeline -> SQLite
            from vendorflow.db.repositories import Repository
            repo = Repository(f"{tmp}/app.db")
            apps = repo.list_applications()
            self.assertEqual(len(apps), 1)
            self.assertEqual(apps[0]["source_ref"], "IN-001")
            self.assertEqual(apps[0]["processing_state"], "EXTRACTED")
            self.assertGreater(len(repo.current_fields(apps[0]["application_id"])), 20)
            shown = [c[0] for c in fake.calls]
            self.assertIn("dataframe", shown)                               # results table was rendered
            writes = " ".join(str(c[1]) for c in fake.calls if c[0] == "write")
            self.assertIn("EXTRACTION", writes)                             # stage events were displayed


if __name__ == "__main__":
    unittest.main()
