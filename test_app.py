"""Automated checks: model loading, sample predictions and Streamlit app behaviour."""
import joblib
from streamlit.testing.v1 import AppTest


def test_model_predictions():
    model = joblib.load("sentiment_model.joblib")
    assert model.predict(["An absolute masterpiece, I loved every minute."])[0] == 1
    assert model.predict(["Terrible plot, a complete waste of time."])[0] == 0
    assert model.predict(["not very good"])[0] == 0      # negation handled


def _run(text=None, click=None):
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    if text is not None:
        at.text_area[0].set_value(text).run()
    if click:
        next(b for b in at.button if click in b.label).click().run()
    assert not at.exception, at.exception
    return at


def test_app_positive():
    at = _run("Brilliant film, I loved it, wonderful acting.", "Analyze")
    assert at.session_state["result"] == 1


def test_app_negative():
    at = _run("Awful, boring and a total waste of time.", "Analyze")
    assert at.session_state["result"] == 0


def test_app_empty_input():
    at = _run("   ", "Analyze")
    assert at.session_state["result"] is None and at.session_state["error"]


def test_app_clear():
    at = _run("Great movie, loved it.", "Analyze")
    next(b for b in at.button if "Clear" in b.label).click().run()
    assert at.session_state["review_text"] == "" and at.session_state["result"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
