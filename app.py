"""CineSense – AI Movie Review Analyzer (Streamlit GUI)."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# The saved pipeline references functions in preprocessing.py; make sure it is importable.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import streamlit as st

MODEL_PATH = BASE_DIR / "sentiment_model.joblib"
MAX_CHARS = 10000

st.set_page_config(page_title="CineSense – AI Movie Review Analyzer",
                   page_icon="🎬", layout="centered")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

.stApp {
    background:
        radial-gradient(1200px 600px at 50% -10%, rgba(229,9,20,.22), transparent 60%),
        radial-gradient(800px 500px at 100% 100%, rgba(255,193,7,.08), transparent 60%),
        #0b0b10;
    color: #f2f2f2;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 820px;}

.filmstrip {
    height: 22px; margin: 0 0 1rem 0; border-radius: 4px; opacity: .85;
    background:
      repeating-linear-gradient(90deg, #14141b 0 16px, transparent 16px 26px) top/100% 6px no-repeat,
      repeating-linear-gradient(90deg, #14141b 0 16px, transparent 16px 26px) bottom/100% 6px no-repeat,
      #1d1d27;
}
.hero {text-align: center; padding: .5rem 0 1rem 0;}
.hero .logo {font-size: 3.4rem; filter: drop-shadow(0 0 18px rgba(229,9,20,.6));}
.hero h1 {
    font-family: 'Bebas Neue', sans-serif; font-size: clamp(2.4rem, 7vw, 4rem);
    letter-spacing: 4px; margin: 0; line-height: 1.05;
    background: linear-gradient(90deg, #ff2d3a, #ffb703);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero .tag {color: #a9a9b8; font-size: 1.02rem; margin-top: .35rem;}

.about {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-left: 4px solid #e50914; border-radius: 12px;
    padding: 14px 18px; margin: .8rem 0 1.4rem 0; color: #cfcfdb; font-size: .95rem;
}
.about b {color: #fff;}
.label {font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; font-size: 1.3rem; color: #ffb703; margin-bottom: .2rem;}

.stTextArea textarea {
    background: #14141c !important; color: #f2f2f2 !important;
    border: 1px solid #2c2c3a !important; border-radius: 12px !important;
    font-size: 1rem !important; line-height: 1.5;
}
.stTextArea textarea:focus {border-color: #e50914 !important; box-shadow: 0 0 0 2px rgba(229,9,20,.3) !important;}

div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100%; border-radius: 12px; font-weight: 600; font-size: 1.05rem;
    padding: .65rem 1rem; border: 1px solid rgba(255,255,255,.12);
    background: #1b1b25; color: #eee; transition: all .2s ease;
}
div.stButton > button:hover { border-color: #e50914; color: #fff; transform: translateY(-1px); }
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #e50914, #ff4b2b); border: none; color: #fff;
    box-shadow: 0 6px 22px rgba(229,9,20,.35);
}
div.stButton > button[kind="primary"]:hover {box-shadow: 0 8px 28px rgba(229,9,20,.55);}

.result {
    border-radius: 18px; padding: 26px 20px; text-align: center; margin-top: 1.4rem;
    animation: pop .45s ease;
}
.result .emoji {font-size: 4rem; line-height: 1;}
.result .verdict {font-family: 'Bebas Neue', sans-serif; font-size: 3rem; letter-spacing: 4px; margin: .2rem 0;}
.result .sub {font-size: 1rem; opacity: .9;}
.result.pos {
    background: linear-gradient(135deg, rgba(0,200,83,.18), rgba(0,120,60,.10));
    border: 1px solid rgba(0,230,118,.55); box-shadow: 0 0 40px rgba(0,230,118,.18);
}
.result.pos .verdict {color: #3dff8f;}
.result.neg {
    background: linear-gradient(135deg, rgba(229,9,20,.22), rgba(120,0,10,.12));
    border: 1px solid rgba(255,70,80,.6); box-shadow: 0 0 40px rgba(229,9,20,.22);
}
.result.neg .verdict {color: #ff5560;}
@keyframes pop {from {opacity: 0; transform: scale(.96) translateY(8px);} to {opacity: 1; transform: none;}}

.chips {display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: .6rem 0 .2rem 0;}
.chip {background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
       padding: 4px 12px; border-radius: 999px; font-size: .8rem; color: #bdbdcc;}
.footer {text-align: center; color: #6d6d80; font-size: .8rem; margin-top: 2.5rem;}

@media (max-width: 600px) {
    .hero .logo {font-size: 2.6rem;}
    .result .verdict {font-size: 2.2rem;}
    .result .emoji {font-size: 3rem;}
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading the projector… 🎞️")
def load_model():
    return joblib.load(MODEL_PATH)


def init_state():
    st.session_state.setdefault("review_text", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)


def clear_all():
    """Callback for the Clear button (runs before widgets are re-created)."""
    st.session_state.review_text = ""
    st.session_state.result = None
    st.session_state.error = None


def load_example(text: str):
    st.session_state.review_text = text
    st.session_state.result = None
    st.session_state.error = None


def analyze():
    st.session_state.result = None
    st.session_state.error = None
    text = st.session_state.review_text or ""
    if not text.strip():
        st.session_state.error = "🎟️ Please write or paste a movie review first."
        return
    if len(text) > MAX_CHARS:
        st.session_state.error = f"📏 That review is too long (max {MAX_CHARS:,} characters)."
        return
    try:
        model = load_model()
        st.session_state.result = int(model.predict([text])[0])
    except Exception as exc:  # unexpected runtime error
        st.session_state.error = f"🎞️ Something went wrong while analyzing the review: {exc}"


init_state()

# ------------------------------------------------------------------ header
st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="logo">🎬</div>
  <h1>CineSense</h1>
  <div class="tag">AI Movie Review Analyzer &nbsp;🍿</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="about">
  🎥 <b>What does it do?</b> Paste any movie review and CineSense tells you whether the
  reviewer's overall sentiment is <b>Positive 😊</b> or <b>Negative 😞</b>.
  It uses a <b>TF-IDF + Linear SVM</b> model trained on the IMDB reviews dataset, with
  <b>negation handling</b> so phrases like <i>“not very good”</i> are understood correctly.
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ model check
if not MODEL_PATH.exists():
    st.error("❌ `sentiment_model.joblib` was not found next to `app.py`. "
             "Run `python train_model.py \"IMDB Dataset.csv\"` to create it.")
    st.stop()
try:
    load_model()
except Exception as exc:
    st.error(f"❌ The model could not be loaded: {exc}")
    st.stop()

# ------------------------------------------------------------------ input
st.markdown('<div class="label">📝 YOUR REVIEW</div>', unsafe_allow_html=True)
st.text_area("Movie review", key="review_text", height=200,
             placeholder="e.g. “The cinematography was stunning and the cast delivered a powerful performance…”",
             label_visibility="collapsed")

col1, col2 = st.columns([3, 1])
with col1:
    st.button("🎬  Analyze Review", type="primary", on_click=analyze, use_container_width=True)
with col2:
    st.button("🗑️ Clear", on_click=clear_all, use_container_width=True)

st.markdown('<div class="chips">🎞️ Try an example:</div>', unsafe_allow_html=True)
EXAMPLES = {
    "😊 Loved it": "An absolute masterpiece. The acting was brilliant, the story gripped me from the first scene, and I loved every minute.",
    "😞 Waste of time": "Terrible plot, wooden acting and a painfully slow script. A complete waste of time and money.",
    "🤔 Tricky negation": "This movie was not good at all. I would never watch it again.",
}
ex_cols = st.columns(len(EXAMPLES))
for (label, text), c in zip(EXAMPLES.items(), ex_cols):
    with c:
        st.button(label, key=f"ex_{label}", on_click=load_example, args=(text,),
                  use_container_width=True)

# ------------------------------------------------------------------ output
if st.session_state.error:
    st.warning(st.session_state.error)

if st.session_state.result is not None:
    if st.session_state.result == 1:
        st.markdown("""
        <div class="result pos">
          <div class="emoji">😊</div>
          <div class="verdict">POSITIVE</div>
          <div class="sub">🍿 Thumbs up! This review has a positive sentiment.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result neg">
          <div class="emoji">😞</div>
          <div class="verdict">NEGATIVE</div>
          <div class="sub">🎞️ Thumbs down. This review has a negative sentiment.</div>
        </div>""", unsafe_allow_html=True)
    st.caption("The model is a Linear SVM, which outputs a class decision rather than calibrated "
               "probabilities, so no confidence percentage is shown.")

st.markdown('<div class="footer">🎬 CineSense · TF-IDF + Negation-aware Linear SVM · Trained on IMDB reviews</div>',
            unsafe_allow_html=True)
