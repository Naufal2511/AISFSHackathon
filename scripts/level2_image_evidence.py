#!/usr/bin/env python3
"""Link every image the Level 2 risk agent looked at to the decisions it supports.

  python scripts/level2_image_evidence.py [--level2 submission/level2] [--no-csv]

Reads <level2>/agent/traces/*.json and <level2>/evidence/facts.jsonl, and writes:
  <level2>/IMAGE_EVIDENCE.md        every cow: each sign verdict with the images it rests on, the behavior check,
                                    the artifacts set aside, and every agent step with the images it returned
                                    (renders on GitHub; click a thumbnail for the full image)
  <level2>/evidence/image_index.csv one row per image: cow, kind, camera, time, size, the step that took it, what cites it
  <level2>/brd_risk.csv             gains an evidence_images column (the images cited in that cow's decisions)
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from src.risk.data import hms

IMG_ID = re.compile(r"\bI\d{4}\b")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STATUS_ORDER = {"present": 0, "absent": 1, "not_assessable": 2}
STATUS_TEXT = {"present": "**present**", "absent": "absent", "not_assessable": "not assessable"}
KINDS = {  # fact source -> (label, what it is)
    "gallery": ("gallery crop", "the closest daylight crops of the cow saved over the 48 h (tool `gallery`)"),
    "get_crop": ("head crop", "a close, enhanced crop of the head at a moment the agent chose (tool `get_crop`)"),
    "breathing": ("breathing window", "the still, close window her breath rate was measured on (tool `breathing`)"),
    "get_sequence": ("8-frame sequence", "frames over a few seconds around a sudden movement, for coughing posture (tool `get_sequence`)"),
    "trend": ("trend chart", "an indicator over the 48 h against the herd, with the sustained-change onset (tool `trend`)"),
    "compare_herdmates": ("herdmate comparison", "the cow next to up to 3 herdmates at the same moment (tool `compare_herdmates`)"),
}


def when(sec) -> str:
    """'11-07 12:11:00' (the cameras' clock, as everywhere in Level 2) -> '7 Nov 12:11'."""
    mmdd, t = hms(sec).split(" ")
    m, d = mmdd.split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]} {t[:5]}"


def cell(text) -> str:
    """Text that is safe inside a Markdown table cell."""
    s = " ".join(str(text or "").split())
    return s.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def caption(f) -> str:
    d = f.get("data", {})
    if f["source"] == "trend":
        return f"trend chart: `{d.get('indicator', '?')}`"
    bits = [d.get("camera", "")]
    if d.get("time"):
        bits.append(when(d["time"]))
    view = ", ".join(x for x in [d.get("region"), d.get("facing")] if x and x != "unknown")
    if view:
        bits.append(view)
    if d.get("box_px"):
        bits.append(f"box {int(d['box_px'])} px")
    if d.get("daylight") is False:
        bits.append("IR night")
    if d.get("bpm"):
        bits.append(f"{d['bpm']:.1f} breaths/min")
    return " · ".join(b for b in bits if b)


def thumb(fid, facts, h=90, cap=True, shown=None) -> str:
    """Linked thumbnail. With `shown` (a set per cow), the caption is written only the first time."""
    f = facts[fid]
    p = f["images"][0]
    img = f'<a href="evidence/{p}"><img src="evidence/{p}" height="{h}" alt="{fid}"></a>'
    if shown is not None:
        cap = cap and fid not in shown
        shown.add(fid)
    return f"{img}<br><sub><b>{fid}</b> {cell(caption(f))}</sub>" if cap else f"{img}<sub>{fid}</sub>"


def expand(ids, facts):
    """Evidence ids -> (image ids, other fact ids). A recorded observation (V####) stands for the images it is based on."""
    imgs, other = [], []
    for e in ids or []:
        f = facts.get(e)
        if f and f["kind"] == "image":
            imgs.append(e)
        elif f and f["kind"] == "visual":
            other.append(e)
            imgs += [i for i in f["data"].get("based_on", []) if i in facts]
        else:
            other.append(e)
    seen = set()
    return [i for i in imgs if not (i in seen or seen.add(i))], other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level2", default="submission/level2")
    ap.add_argument("--no-csv", action="store_true", help="do not add evidence_images to brd_risk.csv")
    a = ap.parse_args()
    L2 = a.level2
    cfg = yaml.safe_load(open("configs/risk.yaml"))
    label = {k: v["label"] for k, v in cfg["visual_signs"].items()}
    facts = {}
    for line in open(os.path.join(L2, "evidence", "facts.jsonl")):
        f = json.loads(line)
        facts[f["id"]] = f
    images = {k: f for k, f in facts.items() if f["kind"] == "image"}
    risk = list(csv.DictReader(open(os.path.join(L2, "brd_risk.csv"))))
    rank = {r["cow_id"]: r for r in risk}
    traces = {}
    for p in glob.glob(os.path.join(L2, "agent", "traces", "*.json")):
        t = json.load(open(p))
        traces[t["cow_id"]] = t
    cows = sorted(traces, key=lambda c: int(rank[c]["rank"]) if c in rank else 99)

    cited_for = defaultdict(list)   # image id -> ["ears drooping: absent", "behavior check", ...]
    step_of = {}                    # image id -> (cow, step number, tool)
    for c in cows:
        t, fin = traces[c], traces[c]["final"]
        for n, s in enumerate(t["steps"], 1):
            if not s.get("tool"):
                continue
            for i in set(s.get("facts") or []) | set(IMG_ID.findall(str(s.get("result") or ""))):
                if i in images and images[i]["cow_id"] == c:
                    step_of.setdefault(i, (c, n, s["tool"]))
        for sign, v in (fin.get("signs") or {}).items():
            for i in expand(v.get("evidence"), facts)[0]:
                cited_for[i].append(f"{label.get(sign, sign)}: {v.get('status', '?').replace('_', ' ')}")
        for art in fin.get("artifacts") or []:
            for i in expand(art.get("evidence"), facts)[0]:
                cited_for[i].append(f"artifact: {art['indicator']}")
        for i in set(IMG_ID.findall(str(fin.get("behavior_assessment") or ""))):
            if i in images:
                cited_for[i].append("behavior check")
        for i in expand(fin.get("key_evidence"), facts)[0]:
            if not cited_for[i]:
                cited_for[i].append("key evidence")

    # ---- page
    out = []
    w = out.append
    n_img, n_cited = len(images), sum(1 for i in images if cited_for.get(i))
    model = sorted({m for t in traces.values() for m in (t.get("models") or [])} or {"the risk agent"})
    w("# Level 2 · the images behind each decision\n")
    w(f"While reviewing the 10 cows, the risk agent ({', '.join(m.replace('protected.', '') for m in model)}) "
      f"looked at **{n_img} images**. Each image has an ID (`I0001`–`I{n_img:04d}`), and every verdict cites "
      f"the IDs it rests on. **{n_cited} of the {n_img} images are cited in a decision.** The agent looked at the "
      f"other {n_img - n_cited} without citing them; the steps under each cow show what it made of them. "
      "**Click a thumbnail to open the full image.**\n")
    w("Day 1 is **7 Nov**, Day 2 is **8 Nov**. Times are the cameras' clock. Cameras: **cam1** front, "
      "**cam2** over the feed bunk, **cam3** back corner. *box* is the cow's height in the 4000 × 3000 source frame.\n")
    w("**What an image can settle.** Code checks every verdict, in `src/risk/agent.py`, before it counts:")
    w(f"- **absent:** needs a crop of that cow at least {cfg['visual_min_box_px']} px tall. A head sign also needs "
      f"the head facing the camera or side-on. Discharge needs a box of at least {cfg['discharge_min_box_px']} px, "
      "in daylight.")
    w("- **not assessable:** the honest answer when no image clears that bar.")
    w("- **present:** would raise the risk. No image showed a sign in any cow.\n")
    w("The ranking, the clips and the reasons are in [`EVIDENCE.md`](EVIDENCE.md). This page holds the still images.\n")
    w("**Contents:** " + " · ".join(f"[{c}](#{c.lower()}--rank-{rank[c]['rank']})" for c in cows if c in rank)
      + " · [Every image](evidence/image_index.csv)\n")

    w("## Summary\n")
    w("| Rank | Cow | Risk | Images looked at | Cited in a decision | absent | not assessable | present |")
    w("|:-:|---|--:|--:|--:|--:|--:|--:|")
    tot = Counter()
    for c in cows:
        signs = traces[c]["final"].get("signs") or {}
        st = Counter(v.get("status") for v in signs.values())
        mine = [i for i, f in images.items() if f["cow_id"] == c]
        k = sum(1 for i in mine if cited_for.get(i))
        tot.update({"img": len(mine), "cited": k, **st})
        w(f"| {rank[c]['rank']} | [**{c}**](#{c.lower()}--rank-{rank[c]['rank']}) | {float(rank[c]['risk_score']):.2f} | "
          f"{len(mine)} | {k} | {st.get('absent', 0)} | {st.get('not_assessable', 0)} | {st.get('present', 0)} |")
    w(f"| | **all** | | **{tot['img']}** | **{tot['cited']}** | {tot['absent']} | {tot['not_assessable']} | "
      f"{tot['present']} |\n")
    w("| Kind of image | Count | What it is |")
    w("|---|--:|---|")
    for src, n in Counter(f["source"] for f in images.values()).most_common():
        lab, what = KINDS.get(src, (src, ""))
        w(f"| {lab} | {n} | {what} |")
    w("")

    for c in cows:
        t, fin, r = traces[c], traces[c]["final"], rank.get(c, {})
        mine = [i for i, f in images.items() if f["cow_id"] == c]
        w("---\n")
        w(f"## {c} · rank {r.get('rank', '?')}\n")
        w(f"Risk **{float(r.get('risk_score', 'nan')):.2f}** · {r.get('confidence', '?')} confidence · "
          f"{len(mine)} images looked at, {sum(1 for i in mine if cited_for.get(i))} cited\n")
        if fin.get("summary"):
            w(f"> **The agent's summary.** {cell(fin['summary'])}\n")
        signs = fin.get("signs") or {}
        shown = set()
        if signs:
            w("**Sign verdicts and the images they rest on**\n")
            w("| Sign | Verdict | Images | The agent's note |")
            w("|---|---|---|---|")
            for sign, v in sorted(signs.items(), key=lambda kv: (STATUS_ORDER.get(kv[1].get("status"), 9), kv[0])):
                imgs, other = expand(v.get("evidence"), facts)
                pics = " ".join(thumb(i, facts, shown=shown) for i in imgs) or "<sub>no image qualified</sub>"
                more = ", ".join(f"`{o}`" for o in other)
                pics += f"<br><sub>also cites {more}</sub>" if more else ""
                w(f"| {cell(label.get(sign, sign))} | {STATUS_TEXT.get(v.get('status'), cell(v.get('status')))} | "
                  f"{pics} | {cell(v.get('note'))} |")
            w("")
        if fin.get("behavior_assessment"):
            beh = [i for i in dict.fromkeys(IMG_ID.findall(fin["behavior_assessment"])) if i in images]
            w(f"**Behavior check: are the deviations real?** {cell(fin['behavior_assessment'])}\n")
            if beh:
                w(" ".join(thumb(i, facts, 120, shown=shown) for i in beh) + "\n")
        arts = fin.get("artifacts") or []
        if arts:
            w("**Indicators set aside as artifacts** (their contribution is removed from the score)\n")
            w("| Indicator | Why | Images |")
            w("|---|---|---|")
            for art in arts:
                imgs, _ = expand(art.get("evidence"), facts)
                w(f"| `{art['indicator']}` | {cell(art.get('why'))} | {' '.join(thumb(i, facts, shown=shown) for i in imgs)} |")
            w("")
        w(f"<details><summary><b>Every step of the review, with the images each one returned "
          f"({len(t['steps'])} steps, {t.get('tool_calls', '?')} tool calls)</b></summary>\n")
        w("| Step | Tool | The agent's reasoning before the call | Images returned |")
        w("|:-:|---|---|---|")
        for n, s in enumerate(t["steps"], 1):
            if not s.get("tool"):
                continue
            got = sorted(i for i, (cc, nn, _) in step_of.items() if cc == c and nn == n)
            pics = " ".join((thumb(i, facts, 70, cap=False) + (" ✓" if cited_for.get(i) else "")) for i in got)
            args = {k: v for k, v in (s.get("args") or {}).items() if k != "cow_id"}
            arg = ", ".join(f"{k}={v}" for k, v in args.items() if k not in ("claim", "images"))
            w(f"| {n} | `{s['tool']}`<br><sub>{cell(arg)}</sub> | {cell(s.get('thought'))} | {pics} |")
        w("\n✓ = cited in a decision.\n")
        w("</details>\n")

    md = os.path.join(L2, "IMAGE_EVIDENCE.md")
    open(md, "w").write("\n".join(out))

    idx = os.path.join(L2, "evidence", "image_index.csv")
    with open(idx, "w", newline="") as fh:  # \n line ends, as the other Level 2 CSVs
        cols = ["image_id", "cow_id", "kind", "camera", "time", "region", "facing", "box_px", "daylight",
                "agent_step", "tool", "cited_for", "path"]
        wr = csv.DictWriter(fh, cols, lineterminator="\n")
        wr.writeheader()
        for i in sorted(images):
            f, d = images[i], images[i].get("data", {})
            st = step_of.get(i, (f["cow_id"], "", ""))
            wr.writerow({"image_id": i, "cow_id": f["cow_id"], "kind": KINDS.get(f["source"], (f["source"],))[0],
                         "camera": d.get("camera", ""), "time": when(d["time"]) if d.get("time") else "",
                         "region": d.get("region", ""), "facing": d.get("facing", ""), "box_px": d.get("box_px", ""),
                         "daylight": d.get("daylight", ""), "agent_step": st[1], "tool": st[2],
                         "cited_for": "; ".join(dict.fromkeys(cited_for.get(i, []))),
                         "path": f"evidence/{f['images'][0]}"})

    if not a.no_csv:
        fields = list(risk[0].keys())
        if "evidence_images" not in fields:
            fields.append("evidence_images")
        for r in risk:
            ids = sorted(i for i, f in images.items() if f["cow_id"] == r["cow_id"] and cited_for.get(i))
            r["evidence_images"] = ";".join(f"evidence/{images[i]['images'][0]}" for i in ids)
        with open(os.path.join(L2, "brd_risk.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fields, lineterminator="\n")
            wr.writeheader()
            wr.writerows(risk)
    print(f"{n_img} images, {n_cited} cited -> {md}, {idx}" + ("" if a.no_csv else ", brd_risk.csv evidence_images"))


if __name__ == "__main__":
    main()
