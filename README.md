# Arabic Smart Home Assistant

A fully local, wake-word-activated voice assistant for controlling smart-home devices with **Levantine Arabic** commands — say "Alexa", give a command, get back structured JSON you can wire into anything.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-whisper--small--arabic--smart--home-yellow)](https://huggingface.co/YahyaMujahed/whisper-small-arabic-smart-home)

```
🎙️ "Alexa"          → Listening...
🎙️ "طفي الضو"        → Processing...
📤 {"device": "light", "action": "off", "value": null, "steps": null}
```

## What it does

```
mic → VAD → wake word → fine-tuned Whisper → keyword matcher → JSON output
```

1. **Mic** — continuous audio capture, nothing sent anywhere until you say the wake word.
2. **Wake word** ([openWakeWord](https://github.com/dscripka/openWakeWord)) — watches quietly, does nothing until triggered.
3. **VAD** ([Silero VAD](https://github.com/snakers4/silero-vad)) — once triggered, captures exactly your spoken command, nothing more, nothing less.
4. **Transcription** — a [Whisper-small model fine-tuned on Arabic smart-home commands](https://huggingface.co/YahyaMujahed/whisper-small-arabic-smart-home), downloaded automatically on first run.
5. **Matching** — turns the transcript into a structured action: `{device, action, value}`.
6. **Output** — printed as JSON, one line per command. Pipe it, log it, or write your own integration.

## Quickstart

```bash
git clone https://github.com/YahyaMurad/arabic-smart-home.git
cd arabic-smart-home
pip install -e .
arabic-assistant
```

No configuration required — say "alexa", wait for `Listening...`, then say a command. First run downloads the model from Hugging Face automatically.

## Configuring it

Drop an `assistant.yml` in the directory you run it from to override any defaults — only specify what you want to change:

```yaml
wake_word: alexa
wake_word_threshold: 0.5
vad_threshold: 0.6
vad_min_silence_duration_ms: 450
vad_speech_pad_ms: 150
repetition_penalty: 1.3
no_repeat_ngram_size: 3
max_new_tokens: 64
```

## Supported commands

Three device categories, matched against natural phrasing variation for each:

| Device | Actions |
|---|---|
| 💡 Lights | on, off, dim, brighten, set (brightness %) |
| ❄️ AC | on, off, set (temp), temp up/down, increase, decrease |
| 🪟 Shades | open, close, set (position %) |

## Integrating it with your own system

Output is one JSON line per recognized command:

```json
{"device": "light", "action": "off", "value": null, "steps": null}
```

Pipe stdout into whatever you already have, or write a custom [`Sink`](arabic_assistant/sinks/base.py) — one method, `send(action) -> None` — and pass it in:

```python
from arabic_assistant.assistant import Assistant
from arabic_assistant.config import AssistantConfig

class MySink:
    def send(self, action):
        ...  # call Home Assistant, publish to MQTT, whatever you use

Assistant(AssistantConfig.load(), sink=MySink()).run()
```

Command matching is pluggable the same way — implement [`Matcher`](arabic_assistant/nlu/base.py) (`match(transcript) -> Action | None`) to swap the keyword matcher for something else (an LLM, a different language, etc.).

## The model

[`YahyaMujahed/whisper-small-arabic-smart-home`](https://huggingface.co/YahyaMujahed/whisper-small-arabic-smart-home) — a LoRA fine-tune of [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) on Jordanian Arabic smart-home commands.

**Current limitation:** this is a single-speaker pilot dataset (one voice, one room, one mic). Expect noticeably worse accuracy on other voices, accents, or noisy/far-mic conditions until a larger multi-speaker version replaces it — check the model repo for updates.

## Project layout

This repo has two independent parts:

```
arabic_assistant/     the tool — self-contained, pip-installable, no dependency on anything below
src/, scripts/         the research pipeline — data collection, training, evaluation
data_collection/       the Gradio app used to record training data
```

`arabic_assistant/` never imports from `src/` — it's fully standalone, so `pip install`-ing it doesn't require the rest of this repo. Everything else here is how the model in `arabic_assistant/`'s default config got made, kept in the same repo for convenience.

## Acknowledgments

- [openai/whisper](https://github.com/openai/whisper) and [transformers](https://github.com/huggingface/transformers)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [openWakeWord](https://github.com/dscripka/openWakeWord)
