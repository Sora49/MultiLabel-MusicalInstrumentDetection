"""
Streamlit app: upload audio -> which of the 5 instruments are in it, how much,
and when.

Run from this folder:
    streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import altair as alt
import librosa
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so the project modules import from any launch directory

from features import SR  # noqa: E402
from predict import analyse_audio, fmt_time, load_model_info, load_thresholds  # noqa: E402
MODELS = {
    "Multi-label detector (recommended)": HERE / "results_multilabel/from_combined/multilabel_model.pt",
    "Multi-label detector, pretrained start": HERE / "results_multilabel/from_pretrain/multilabel_model.pt",
    "Single-label CNN (one instrument per window)": HERE / "cnn_model.pt",
}
NAMES = {"flu": "Flute", "gac": "Acoustic guitar", "pia": "Piano", "vio": "Violin", "voi": "Voice"}
# one fixed colour per instrument (validated categorical palette, light / dark variants)
COLORS = {
    "light": {"flu": "#2a78d6", "gac": "#eb6834", "pia": "#1baf7a", "vio": "#eda100", "voi": "#e87ba4"},
    "dark": {"flu": "#3987e5", "gac": "#d95926", "pia": "#199e70", "vio": "#c98500", "voi": "#d55181"},
}

st.set_page_config(page_title="Instrument Detector", page_icon="🎵", layout="wide")


@st.cache_resource
def get_model(path: str):
    model, classes, multilabel = load_model_info(path)
    return model, classes, multilabel, load_thresholds(path, classes, multilabel)


@st.cache_data(show_spinner=False)
def run_analysis(audio_bytes: bytes, suffix: str, model_path: str, hop: float, smooth: int):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        y, _ = librosa.load(tmp, sr=SR, mono=True)
    finally:
        Path(tmp).unlink(missing_ok=True)
    model, classes, multilabel, thresholds = get_model(model_path)
    starts, scores, present = analyse_audio(y, model, multilabel, thresholds, hop, smooth)
    return len(y) / SR, starts, scores, present


def segments(starts, present_col, duration):
    """Merge consecutive windows where an instrument is present into (start, end) segments."""
    out, seg_start, last_end = [], None, None
    for s, on in zip(starts, present_col):
        if on and seg_start is None:
            seg_start = s
        if on:
            last_end = min(s + 3.0, duration)
        if not on and seg_start is not None:
            out.append((seg_start, last_end))
            seg_start = None
    if seg_start is not None:
        out.append((seg_start, last_end))
    return out


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    available = {k: v for k, v in MODELS.items() if v.exists()}
    if not available:
        st.error("No trained models found. Run the training scripts first.")
        st.stop()
    model_label = st.selectbox("Model", list(available))
    hop = st.radio("Window step", [3.0, 1.5], format_func=lambda h: f"every {h:g} s",
                   horizontal=True, help="Windows are always 3 s long; 1.5 s gives a finer, overlapping timeline.")
    smooth = st.select_slider("Smoothing", options=[1, 3, 5], value=1,
                              format_func=lambda k: "off" if k == 1 else f"{k} windows",
                              help="Average each window with its neighbours before deciding. "
                                   "Fills one-window gaps and removes one-off blips.")
    model_path = str(available[model_label])
    _, classes, multilabel, thresholds = get_model(model_path)
    st.caption(
        ("Multi-label: each window can contain several instruments, or none of them. "
         "Thresholds per instrument: "
         + ", ".join(f"{NAMES[c]} {t:.2f}" for c, t in zip(classes, thresholds))
         if multilabel and thresholds is not None else
         "Single-label: each window gets exactly one instrument (the most prominent), "
         "even if none of the 5 is playing.")
    )

theme = "dark" if st.context.theme.type == "dark" else "light"
palette = COLORS[theme]
color_scale = alt.Scale(domain=[NAMES[c] for c in classes], range=[palette[c] for c in classes])

st.title("Instrument Detector")
st.caption("Detects flute, acoustic guitar, piano, violin and voice in any audio file. "
           "NNDL project · CNN on log-mel spectrograms · trained on IRMAS.")

tab_analyse, tab_model = st.tabs(["Analyse audio", "Model & results"])

# ---------------------------------------------------------------- analyse tab
with tab_analyse:
    upload = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac", "ogg", "m4a"])
    if upload is None:
        st.info("Upload a song or recording to see which instruments are in it and when.")
    else:
        audio_bytes = upload.getvalue()
        st.audio(audio_bytes)
        with st.spinner("Listening… (about 1 s per minute of audio)"):
            duration, starts, scores, present = run_analysis(
                audio_bytes, Path(upload.name).suffix, model_path, hop, smooth)

        share = present.mean(0)
        order = np.argsort(-share)
        st.subheader(f"{upload.name}  ·  {fmt_time(duration)}")

        # headline tiles: share of the audio each instrument is detected in
        cols = st.columns(len(classes))
        for col, i in zip(cols, order):
            col.metric(NAMES[classes[i]], f"{share[i] * 100:.0f}%",
                       help=f"Detected in {present[:, i].sum()} of {len(starts)} windows "
                            f"(mean score {scores[:, i].mean():.2f})")
        if not present.any(1).all():
            st.caption(f"No target instrument detected in {(~present.any(1)).sum()} of {len(starts)} windows "
                       "(silence, or other sounds such as drums, bass or other instruments).")

        # timeline: one row per instrument, coloured where it is detected
        st.markdown("#### Timeline")
        rows = []
        for i, c in enumerate(classes):
            for s, e in segments(starts, present[:, i], duration):
                in_seg = (starts >= s) & (starts < e)
                rows.append({"Instrument": NAMES[c], "start": s, "end": e,
                             "From": fmt_time(s), "To": fmt_time(e),
                             "Mean score": round(float(scores[in_seg, i].mean()), 2)})
        if rows:
            seg_df = pd.DataFrame(rows)
            timeline = (
                alt.Chart(seg_df)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("start:Q", title="Time (s)", scale=alt.Scale(domain=[0, duration], nice=False)),
                    x2="end:Q",
                    # every instrument keeps its row, so an empty row reads as "not detected"
                    y=alt.Y("Instrument:N", sort=[NAMES[classes[i]] for i in order], title=None,
                            scale=alt.Scale(domain=[NAMES[classes[i]] for i in order], paddingInner=0.35)),
                    color=alt.Color("Instrument:N", scale=color_scale, legend=alt.Legend(orient="top", title=None)),
                    tooltip=["Instrument", "From", "To", "Mean score"],
                )
                .properties(height=60 + 42 * len(classes))
            )
            st.altair_chart(timeline, use_container_width=True)
        else:
            st.warning("None of the 5 instruments was detected anywhere in this file.")

        # scores over time, for a closer look
        with st.expander("Scores over time"):
            score_df = pd.DataFrame(scores, columns=[NAMES[c] for c in classes])
            score_df["Time (s)"] = starts + 1.5  # window centre
            long = score_df.melt("Time (s)", var_name="Instrument", value_name="Score")
            hover = alt.selection_point(fields=["Time (s)"], nearest=True, on="mouseover", empty=False)
            lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
                x=alt.X("Time (s):Q", scale=alt.Scale(domain=[0, duration], nice=False)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Instrument:N", scale=color_scale, legend=alt.Legend(orient="top", title=None)),
            )
            points = lines.mark_point(size=60, filled=True).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                tooltip=["Instrument", alt.Tooltip("Time (s):Q", format=".1f"), alt.Tooltip("Score:Q", format=".2f")],
            ).add_params(hover)
            rule = alt.Chart(long).mark_rule(strokeWidth=1, color="#898781").encode(
                x="Time (s):Q").transform_filter(hover)
            st.altair_chart(lines + points + rule, use_container_width=True)
            st.caption("Each line is one instrument's score per window"
                       + (", before thresholds." if multilabel else " (softmax: the 5 scores sum to 1)."))

        # table view + download
        table = pd.DataFrame({"From": [fmt_time(s) for s in starts], "To": [fmt_time(min(s + 3, duration)) for s in starts]})
        table["Detected"] = [", ".join(NAMES[c] for c, on in zip(classes, p) if on) or "none of the 5" for p in present]
        for i, c in enumerate(classes):
            table[NAMES[c]] = scores[:, i].round(2)
        with st.expander("Window-by-window table"):
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("Download timeline (CSV)", table.to_csv(index=False).encode(),
                           file_name=f"{Path(upload.name).stem}_timeline.csv", mime="text/csv")

# ---------------------------------------------------------------- model tab
with tab_model:
    summary = HERE / "RESULTS_SUMMARY.md"
    if summary.exists():
        st.markdown(summary.read_text(encoding="utf-8"))
    curve = Path(model_path).with_name("training_curve.png")
    if curve.exists():
        st.markdown("#### Training curve of the selected model")
        st.image(str(curve))
