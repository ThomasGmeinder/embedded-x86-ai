# Reachy Mini Lab Guide

In ~10 minutes you'll take a desktop robot from a **cloud** voice → **100%
offline** → one that **sees and answers questions** through its own camera.

| Task | What it is | You do |
|------|-----------|--------|
| 1 · Voice | Expressive robot, **cloud** voice | edit persona/voice → run → **chat** |
| 2 · Local | Same robot, **100% offline** | edit persona → run → **chat** |
| 3 · Eyes | Reachy **sees** and answers your questions | run the action cell → **type a question** |

**Everything runs in one notebook: [`lab.ipynb`](lab.ipynb).** Open it, pick the
**venv kernel**, and run the **Setup** cell first. Then for each task:

1. *(Optional)* edit the `# >>> TRY ME <<<` config cell and run it.
2. **Tasks 1–2:** type a message in the chat bar and hit **Send**.
3. **Task 3:** run the action cell for a live feed, then type a question.

---

## Task 1 · Voice · *cloud*

Reachy moves *while it speaks*, using **Microsoft Edge-TTS** — a cloud service.
Remember that for Task 2.

**Edit → run → chat:** change `PERSONA_1` or `VOICE_1` in the `# >>> TRY ME <<<`
cell, run it, then type `tell me a joke` in the chat bar.

A few voices to try: `en-US-AnaNeural` (child), `en-GB-RyanNeural` (British),
`en-US-GuyNeural` (deep).

---

## Task 2 · Local · *100% offline*

Same expressive robot, but the LLM **and** voice (Piper-TTS) run entirely
on-device — nothing leaves the machine. Often snappier than Task 1.

**Edit → run → chat:** change `PERSONA_2` (or swap `PIPER_MODEL`) in the
`# >>> TRY ME <<<` cell, run it, then chat.

> **Proof it's local:** turn off Wi-Fi and keep chatting — Reachy won't miss a
> beat. Task 1 would go quiet.

---

## Task 3 · Eyes · *vision, still local*

Reachy looks through its camera (a live feed appears). When you type a question
it sends the current frame to a **local vision model** and answers out loud.

Hold up an object and type a question, or click a suggestion chip:
**What do you see?**, **What do I look like?**, **Describe where I am?**

**The key edit:** change `VISION_STYLE` in the `# >>> TRY ME <<<` cell:

- `VISION_STYLE = "Answer like a pirate."`
- `VISION_STYLE = "Answer like an excited scientist."`
- `VISION_STYLE = "Answer in a calm, poetic voice."`

---

## Fallback: terminal scripts

If Jupyter has trouble, the same tasks run as terminal scripts:

```bash
python lab/emo_v1.py --chat          # Task 1 — cloud voice
python lab/emo_v2.py --chat          # Task 2 — 100% offline
python lab/emo_v3.py --preview-web   # Task 3 — vision; open http://localhost:8080

# Task 3 simulation mode - default to built-in camera on laptop/computer
# Run `v4l2-ctl --list-devices` to find your device
python lab/emo_v3.py --preview-web --camera-device /dev/video0
```

Edit the `# >>> TRY ME <<<` block in each script, save, then **Ctrl+C** and re-run.
Task 3 uses **Enter / "Look & Describe"** with a `VISION_PROMPT` instead of the
notebook's question bar.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Robot doesn't move | Make sure the daemon (Terminal A) is still running. |
| `Connection refused` / Ollama errors | Run `ollama serve` to start the local LLM server. |
| Offline chat fails with `Cannot connect to host localhost:11434` | Wi-Fi off can cause `localhost` to resolve to IPv6. Use `127.0.0.1` — the lab scripts already do this; make sure you're on the latest version. |
| No audio / too quiet | **Settings → Sound** — set Output Device + Output Volume. Or run `alsamixer` (F6 to pick the device). |
| Task 3: "Could not read camera" | Try `python lab/emo_v3.py --camera-device /dev/video0`. List devices with `v4l2-ctl --list-devices`. |
| Task 3: "Device or resource busy" | Restart the daemon with `reachy-mini-daemon --no-media`. |
| Task 3: GStreamer webrtcsink warning | Expected and harmless — Task 3 reads the camera directly via ffmpeg, bypassing the daemon's media server. |
| Script stuck, Ctrl+C won't stop it | From another terminal: `pkill -9 -f emo_v` — then re-run. Don't kill the daemon. |
| Sim window won't open | Set `export PYGLFW_LIBRARY_VARIANT=x11` before starting the daemon. |
