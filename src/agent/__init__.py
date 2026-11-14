"""Supervisor agent: a VLM that audits the perception pipeline (YOLO -> ByteTrack ->
DINOv2/colour identity -> behavior classifier) only where it is unsure or something
drastic happens, keeps a memory card per cow, verifies its own corrections and logs
reasoning traces.

  triggers.py  what needs attention (drastic events, low confidence, 30-min audits)
  memory.py    per-cow memory cards (references, traits, notes, last seen)
  evidence.py  frame / crop access on the raw video + appearance embeddings
  tools.py     the tools the agent can call
  vlm.py       OpenAI-compatible client (Ollama/Qwen locally, TAMU Chat, vLLM) + offline policy
  loop.py      plan -> gather evidence -> decide -> verify -> trace
  audit.py     30-minute ID-check sheets
"""
