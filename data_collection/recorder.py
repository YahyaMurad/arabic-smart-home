import argparse
import csv
import html
import json
import random
import re
import threading
import time
from math import gcd
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
PROMPTS_FILE = ROOT / "config.yml"
DATA_DIR = ROOT.parent / "data"
AUDIO_DIR = DATA_DIR / "raw"
CSV_PATH = DATA_DIR / "recordings.csv"

TARGET_SR = 16000
MIN_DURATION_S = 0.4
MIN_PEAK = 0.02
CLIP_PEAK = 0.99

FIELDS = [
    "file",
    "transcript",
    "action",
    "speaker",
    "dialect",
    "distance",
    "noise",
    "prompt_id",
    "prompt_type",
    "duration_s",
    "timestamp",
]

_csv_lock = threading.Lock()

CFG = yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8"))
SETTINGS = CFG.get("settings", {})
PROMPTS = CFG["prompts"]
PROMPTS_BY_ID = {p["id"]: p for p in PROMPTS}
assert len(PROMPTS_BY_ID) == len(PROMPTS), "Duplicate prompt ids in prompts.yaml"


def to_mono_float(data):
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data


def resample(y, sr):
    if sr == TARGET_SR:
        return y
    g = gcd(sr, TARGET_SR)
    return resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)


def read_rows():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(row):
    with _csv_lock:
        new = not CSV_PATH.exists()
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def remove_row(file_rel):
    with _csv_lock:
        rows = [r for r in read_rows() if r["file"] != file_rel]
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)


def slug(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip()).strip("_")


def build_order(speaker, distance, noise):
    done = {}
    for r in read_rows():
        if (r["speaker"], r["distance"], r["noise"]) == (speaker, distance, noise):
            done[r["prompt_id"]] = done.get(r["prompt_id"], 0) + 1
    repeats = int(SETTINGS.get("repeats", 1))
    order = []
    for p in PROMPTS:
        order += [p["id"]] * max(0, repeats - done.get(p["id"], 0))
    if SETTINGS.get("shuffle", True):
        random.Random(f"{speaker}|{distance}|{noise}").shuffle(order)
    return order


def prompt_html(p):
    label = "قولها بطريقتك" if p["type"] == "goal" else "اقرأ الجملة"
    return (
        '<div dir="rtl" style="text-align:center;padding:28px 12px">'
        f'<div style="opacity:.65;font-size:1.1em">{label}</div>'
        f'<div style="font-size:2.4em;font-weight:600;margin-top:14px;line-height:1.4">'
        f"{html.escape(p['text'])}</div></div>"
    )


DONE_HTML = (
    '<div dir="rtl" style="text-align:center;padding:40px;font-size:2em">'
    "خلصت! يعطيك العافية 🙏</div>"
)


def render(state):
    if not state:
        return state, gr.update(visible=True), gr.update(visible=False), "", "", None
    order, pos = state["order"], state["pos"]
    if pos >= len(order):
        return (
            state,
            gr.update(visible=True),
            gr.update(visible=True),
            DONE_HTML,
            f"**{len(order)} / {len(order)}**",
            None,
        )
    p = PROMPTS_BY_ID[order[pos]]
    return (
        state,
        gr.update(visible=False),
        gr.update(visible=True),
        prompt_html(p),
        f"**{pos + 1} / {len(order)}**",
        None,
    )


def start(speaker, dialect, distance, noise):
    speaker = slug(speaker or "")
    if not speaker:
        raise gr.Error("Enter a speaker ID (letters, numbers, _ or -).")
    if not (dialect and distance and noise):
        raise gr.Error("Pick dialect, distance and noise.")
    order = build_order(speaker, distance, noise)
    if not order:
        gr.Info(
            "This speaker already finished this condition. Pick another distance/noise."
        )
    state = {
        "speaker": speaker,
        "dialect": dialect,
        "distance": distance,
        "noise": noise,
        "order": order,
        "pos": 0,
        "last_file": None,
        "last_pos": None,
    }
    return render(state)


def save_next(audio, state):
    if not state or state["pos"] >= len(state["order"]):
        return render(state)
    if audio is None:
        raise gr.Error("Record first, then press Save & Next.")
    sr, data = audio
    y = to_mono_float(data)
    dur = len(y) / sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if dur < MIN_DURATION_S:
        raise gr.Error(f"Too short ({dur:.2f}s). Please record again.")
    if peak < MIN_PEAK:
        raise gr.Error(
            "Almost no sound was captured. Check the microphone and record again."
        )
    if peak >= CLIP_PEAK:
        gr.Warning(
            "The recording is clipping (too loud). Saved anyway; consider moving back a bit."
        )

    y = np.clip(resample(y, sr), -1.0, 1.0)
    p = PROMPTS_BY_ID[state["order"][state["pos"]]]
    ts = int(time.time() * 1000)
    name = f"{state['speaker']}_{p['id']}_{state['distance']}_{state['noise']}_{ts}.wav"
    path = AUDIO_DIR / state["speaker"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, TARGET_SR, subtype="PCM_16")

    file_rel = path.relative_to(DATA_DIR).as_posix()
    append_row(
        {
            "file": file_rel,
            "transcript": p["text"]
            if p["type"] == "read"
            else "",  # goal prompts: fill in later
            "action": json.dumps(p["action"], ensure_ascii=False),
            "speaker": state["speaker"],
            "dialect": state["dialect"],
            "distance": state["distance"],
            "noise": state["noise"],
            "prompt_id": p["id"],
            "prompt_type": p["type"],
            "duration_s": f"{len(y) / TARGET_SR:.2f}",
            "timestamp": ts,
        }
    )
    state["last_file"], state["last_pos"] = file_rel, state["pos"]
    state["pos"] += 1
    return render(state)


def skip(state):
    if state and state["pos"] < len(state["order"]):
        state["pos"] += 1
    return render(state)


def undo(state):
    if not state or not state.get("last_file"):
        raise gr.Error("Nothing to undo.")
    remove_row(state["last_file"])
    (DATA_DIR / state["last_file"]).unlink(missing_ok=True)
    state["pos"] = state["last_pos"]
    state["last_file"] = state["last_pos"] = None
    gr.Info("Last take deleted. Record it again.")
    return render(state)


def change_setup(state):
    return render({})


def build_ui():
    with gr.Blocks(title="Arabic Command Recorder") as demo:
        state = gr.State({})
        gr.Markdown("## Arabic Smart-Room Command Recorder")

        with gr.Column(visible=True) as setup_col:
            speaker = gr.Textbox(label="Speaker ID", placeholder="e.g. spk01")
            with gr.Row():
                dialect = gr.Dropdown(SETTINGS.get("dialects", []), label="Dialect")
                distance = gr.Dropdown(
                    SETTINGS.get("distances", []), label="Distance from mic"
                )
                noise = gr.Dropdown(SETTINGS.get("noise", []), label="Background noise")
            start_btn = gr.Button("Start", variant="primary")

        with gr.Column(visible=False) as rec_col:
            progress = gr.Markdown()
            prompt = gr.HTML()
            audio = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Press record, say it, press stop. Listen back if you want.",
            )
            with gr.Row():
                undo_btn = gr.Button("↩ Undo last")
                skip_btn = gr.Button("Skip")
                next_btn = gr.Button("Save & Next ➜", variant="primary")
            setup_btn = gr.Button("Change speaker / condition", size="sm")

        outs = [state, setup_col, rec_col, prompt, progress, audio]
        start_btn.click(start, [speaker, dialect, distance, noise], outs)
        next_btn.click(save_next, [audio, state], outs)
        skip_btn.click(skip, [state], outs)
        undo_btn.click(undo, [state], outs)
        setup_btn.click(change_setup, [state], outs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--share", action="store_true", help="public HTTPS link for remote speakers"
    )
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
