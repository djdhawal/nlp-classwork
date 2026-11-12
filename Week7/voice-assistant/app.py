"""Web page for the voice assistant:  streamlit run app.py

The assistant (and its listening loop) lives in background threads created once by
st.cache_resource. Streamlit reruns this script on every click; the page only reads state.
"""
import json

import pandas as pd
import streamlit as st

from va.assistant import Assistant
from va.config import load_config

st.set_page_config(page_title="Loop voice assistant", page_icon="🎙️", layout="wide")


@st.cache_resource
def get_assistant() -> Assistant:
    cfg = load_config()
    return Assistant(cfg, voice=cfg["audio"].get("enabled", True), speak=True).start()


a = get_assistant()

BANNERS = {   # status -> (colour, text a beginner understands)
    "loading":   ("#6b7280", "⏳ Loading models… not listening yet"),
    "listening": ("#16a34a", "🎙️ Listening. Say “{name}, …”"),
    "thinking":  ("#d97706", "🤔 Thinking…"),
    "speaking":  ("#2563eb", "🔊 Speaking (microphone off)"),
    "muted":     ("#dc2626", "🔇 Muted. Not listening"),
    "ready":     ("#16a34a", "⌨️ Ready (typing only, no microphone)"),
    "error":     ("#991b1b", "⚠️ Error: {error}"),
}

# ---- controls (outside the live fragment) --------------------------------------------
with st.sidebar:
    st.header("Controls")
    muted = st.toggle("Mute microphone", value=a.muted)
    if muted != a.muted:
        a.set_muted(muted)
    if st.button("⏹ Stop speaking", use_container_width=True):
        a.stop_speaking()
    with st.form("typed", clear_on_submit=True):
        typed = st.text_input("Type a command", placeholder="book a dentist appointment tomorrow at 3 pm in Chicago")
        if st.form_submit_button("Send", use_container_width=True) and typed.strip():
            a.submit(typed.strip())
    st.caption(f"Model: `{a.snapshot()['model']}`")


@st.fragment(run_every=0.5)
def live():
    snap = a.snapshot()
    colour, text = BANNERS.get(snap["status"], ("#6b7280", snap["status"]))
    text = text.format(name=a.cfg["assistant"]["name"], error=snap["error"])
    if snap["status"] == "listening" and snap["follow_up"] > 0:
        text += f" (or just speak, no wake word needed for {snap['follow_up']:.0f} s)"
    for note in snap["notes"]:
        st.warning(note)
    st.markdown(f"<div style='background:{colour};color:white;padding:18px 24px;border-radius:12px;"
                f"font-size:1.6rem;font-weight:600'>{text}</div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Conversation")
        if not snap["turns"]:
            st.caption("Nothing yet.")
        for t in reversed(snap["turns"]):
            with st.container(border=True):
                icon = "🎙️" if t["source"] == "voice" else "⌨️"
                st.markdown(f"**{icon} {t['at']} heard:** {t['heard']}")
                if t["tools"]:
                    with st.expander(f"{len(t['tools'])} tool call(s)"):
                        for ev in t["tools"]:
                            ok = "✅" if ev["result"].get("ok") else "❌"
                            st.markdown(f"{ok} `{ev['tool']}`")
                            st.code(json.dumps({"arguments": ev["arguments"], "result": ev["result"]}, indent=1), "json")
                if t["state"] in ("queued", "thinking"):
                    st.markdown("*…thinking*")
                else:
                    timing = f" ({t['seconds']} s" + (f", first words {t['first_audio']} s)" if t.get("first_audio") else ")") if t["seconds"] else ""
                    st.markdown(f"**🔊 reply:** {t['reply']}{timing}")
                if t.get("extracted"):
                    st.caption("remembered: " + "; ".join(t["extracted"]))
        if snap["heard"]:
            with st.expander("Everything the microphone heard"):
                st.dataframe(pd.DataFrame(snap["heard"][::-1]), hide_index=True, use_container_width=True)

    with right:
        st.subheader("Appointments")
        appts = a.store.list()
        if appts:
            st.dataframe(pd.DataFrame(appts).set_index("id"), use_container_width=True)
        else:
            st.caption("No appointments.")
        st.subheader("Memory")
        if a.memory is None:
            st.caption("Memory is off (is Qdrant running?).")
        else:
            mems = a.memory.all()
            if not mems:
                st.caption("Nothing remembered yet.")
            for m in mems:
                c1, c2 = st.columns([6, 1])
                c1.markdown(f"• {m['text']}  \n<span style='color:gray;font-size:0.8em'>{m.get('source', '')}, {m.get('created_at', '')}</span>",
                            unsafe_allow_html=True)
                if c2.button("🗑", key=f"del-{m['id']}"):
                    a.memory.delete(m["id"])
                    st.rerun()


live()
