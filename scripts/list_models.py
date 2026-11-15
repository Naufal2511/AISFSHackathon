#!/usr/bin/env python3
"""List the models an OpenAI-compatible backend offers (sends only the API key, no data).

  python scripts/list_models.py --backend tamu
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml

from src.agent.vlm import load_dotenv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tamu")
    ap.add_argument("--base-url")
    a = ap.parse_args()
    load_dotenv()
    p = yaml.safe_load(open("configs/agent.yaml"))[a.backend]
    base = (a.base_url or os.environ.get("AGENT_VLM_BASE_URL") or p["base_url"]).rstrip("/")
    key = os.environ.get(p.get("api_key_env", "")) or os.environ.get("AGENT_VLM_API_KEY")
    if not key:
        sys.exit(f"no key: put {p.get('api_key_env')}=... in .env")
    for path in ("/models", "/v1/models", "/api/models"):
        try:
            req = urllib.request.Request(base + path, headers={"Authorization": f"Bearer {key}",
                                                                "User-Agent": "Mozilla/5.0 (Macintosh) pen6-agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            items = data.get("data", data) if isinstance(data, dict) else data
            print(f"{base + path}: {len(items)} models")
            for m in items:
                mid = m.get("id") if isinstance(m, dict) else m
                print("  ", mid)
            return
        except Exception as e:
            print(f"{base + path}: {type(e).__name__}: {str(e)[:120]}")
    sys.exit("could not list models: check base_url on docs.tamus.ai")


if __name__ == "__main__":
    main()
