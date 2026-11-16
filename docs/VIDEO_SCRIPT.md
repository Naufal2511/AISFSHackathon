# Demo video: script and shot list

**Length:** under 5 min (≈ 615 spoken words at 135 words/min ≈ 4:35, plus short pauses between scenes).
**Voices:** A (Naufal): scenes 1–7 · B (Afeef): scenes 8–10 · C (Nooh): scenes 11–13 · all three: scene 14. Swap names freely.
**Tone:** grounded and technical. Say what the system measured, then show it. The finding is *early suspects, not sick*.
**Look:** the brief's palette (teal `#0f5a6b`, orange `#a85a24` for agents and the focus cow), Avenir Next. Every animation is driven by our real outputs.

> **Event data rule:** the video contains pen footage. Send it to the organizers only (file or private link). Do not post it publicly.

**Build:** `python3 video/build.py`. It renders every scene from `video/scenes/`, uses your recordings from `video/vo/` when they exist (otherwise a scratch computer voice for timing), and writes `outputs/video/demo.mp4` plus `demo.srt` captions. Each scene stretches to fit its voice-over.

---

| # | Scene | Voice | Visual (all built by `video/build.py`) |
|---|---|---|---|
| 1 | Cold open | A | Night IR footage of the pen, then daylight; title "10 Angus cattle · 3 cameras · 48 hours" |
| 2 | Live tracking | A | Feed rush, 8 Nov 16:28: three real camera feeds with our boxes, IDs and behavior, synced with the world-state map |
| 3 | Pipeline | A | The brief's architecture diagram; each row lights up as it is named |
| 4 | World state | A | Top-down pen playback from `cow_state.csv`: heading, seen vs predicted (± m halo), per-camera chips, live callout for one cow |
| 5 | Where to look | A | One cow, three real camera crops scored with the `best_view` rule (box size × visibility × facing weight); then the funnel of 28,800 cow-minutes |
| 6 | Identity | A | Must-link / cannot-link; the consensus update run step by step with its falling disagreement energy; a real second where every cow standing clear gets her ID locked; a real ID-check sheet |
| 7 | Memory cards | A | A real memory card (crops, ear tag, traits, notes with time and source) and a `compare_to_cards` ranking |
| 8 | Behavior | B | Our hand-labelled crops with their labels; "rumination only when chewing is measured" |
| 9 | Agents | B | The real success trace stepping through plan → tools → decision; the self-check rules |
| 10 | Level 1 results | B | The basis bar for 28,800 cow-minutes; correction counts |
| 11 | Stitching | C | 33 chunks on a 48 h timeline; joins by box overlap |
| 12 | BRD risk | C | The feed rush again with COW_02 highlighted; then the ranking with reasons and resampling stability |
| 13 | Honest reading | C | "early suspects, not sick" and what 10 cows × 48 h cannot support |
| 14 | Close | A · B · C | Three lines; team and repository |

---

## Voice-over (read exactly; one file per scene: `video/vo/vo_01.wav` … `vo_14.wav`)

**1 · A**
> Ten Angus cattle. Three cameras. Two days and nights. Respiratory disease changes behavior days before it shows in the clinic. Can a camera notice first?

**2 · A**
> This is our system on the real footage. Every box is a detection, every colour is one cow, the same in all three cameras. Black Angus look almost identical, so identity can't come from looks alone.

**3 · A**
> So we measure first and judge second. A fine-tuned YOLO detector finds every cow, ByteTrack follows her within each camera, and a homography puts her on the pen floor, in metres. Then agents check that record, and code checks the agents.

**4 · A**
> That becomes a world state: every cow, every second. Where she stands, which way she faces, which cameras see her, and whether she's fully visible, partly hidden, or blocked by another cow. When no camera sees her, we predict her position, and say how unsure we are.

**5 · A**
> The world state also tells the agents where a look is worth it. A head check needs her facing the camera; an identity check wants her side-on; blocked or distant views are skipped. Most minutes need no model at all: sixty percent pass cheap rule checks. The agent only looks where it can learn something.

**6 · A**
> Identity comes from geometry and time. Two cameras seeing a cow on the same spot at the same seconds: one cow. Seen together, metres apart: two cows. Each camera's belief is pulled into agreement by a consensus protocol whose disagreement can only fall. And whenever a cow stands clear of the others, seen and agreed on, her identity is locked.

**7 · A**
> Each cow also has a memory card: her best views, by day and night, her marks, and an appearance fingerprint. Every new fact an agent learns goes on her card, with when and how. When a track is in doubt, the agent compares it to the cards, and checks the match against where she is.

**8 · B**
> For behavior, we hand-labelled about three hundred and thirty clips and trained a DINOv2 classifier, plus a public cattle dataset. Rumination is only called when chewing is measured. Cameras vote as experts, and a smoothing model removes impossible jumps.

**9 · B**
> Here's the behavior agent. Stage one said drinking. The agent picked the best camera, checked zone, track and frames, and found her walking. But code has the last word: a correction needs two kinds of evidence. Fail twice, and the answer is "uncertain", never a silent guess.

**10 · B**
> The result: every cow, every minute, for forty-eight hours, and each minute says how we know it. The agent corrected three hundred and thirty-five minutes and refused to guess on five thousand.

**11 · C**
> Forty-eight hours of 4K didn't fit one machine in time, so thirty-three chunks ran in parallel, joined where neighbouring chunks saw the same boxes in the same frames.

**12 · C**
> Remember that feed rush? While the herd eats, COW_02 lies alone. Across thirteen indicators from the BRD literature, each compared with the herd at the same hours, she ranks first: twenty-seven minutes to reach the bunk after feeding, against nineteen for the herd, and she walks and eats less. COW_01 and COW_10 follow, and all three hold their place under resampling.

**13 · C**
> But no cow is far from the herd, and no visible clinical sign was confirmed. Our call: early suspects, not sick. Take their temperatures. Ten cows over two days can't validate a BRD model, and our system says so.

**14 · A, B, C**
> A: Measure first. B: Check everything. C: And say what you don't know.

---

## Recording checklist

1. One file per scene: `video/vo/vo_01.wav` … `vo_14.wav` (any of wav / m4a / mp3). For scene 14, either one file with all three voices or `vo_14a`, `vo_14b`, `vo_14c`.
2. Same phone, same small soft room, 20–30 cm from the mouth. Leave half a second of silence at each end.
3. Read calmly. The edit stretches each scene to your read, so there is no need to rush.
4. Then run `python3 video/build.py` again: your voices replace the scratch voice and the timing follows them.
