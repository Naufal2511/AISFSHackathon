# Demo video: script and shot list

**Length:** about 5 min 50 s (≈ 790 spoken words at a calm 135 words/min). To get under 5 min, drop scene 11 and shorten scenes 2 and 6.
**Voices:** three, each owning one part of the story, so the hand-offs feel natural:

| Voice | Owns | Scenes |
|---|---|---|
| **A: Naufal** | the problem, live tracking, world state, where to look, identity, memory | 1–7 |
| **B: Afeef** | behavior, the agents, Level 1 results | 8–10 |
| **C: Nooh** | stitching, BRD risk, limits, close | 11–14 |

(Swap names freely; the split is by topic, not by person.)

**Tone:** grounded and technical. Say what the system measured, then show it. Never claim a cow is sick: the honest finding is *early suspects, not sick*.

**Look:** the same palette as the brief (deep teal `#0f5a6b`, warm orange `#a85a24` for anything an agent does, off-white panels). Avenir Next for on-screen text. One idea per shot. **Every animation is driven by our real outputs** (`cow_state.csv`, `observations.csv`, memory cards, traces), not drawn by hand, so what you see is what the system computed.

> **Event data rule:** the video contains pen footage. Send it to the organizers only (file or private link). Do not post it publicly.

**Legend:** 🎞 real footage · ✳ animation built from our data · 🖥 screen recording of our outputs

---

## Part A: seeing and knowing who is who (A)

### 1 · Cold open (0:00–0:15)
🎞 Night IR footage of the pen, slow push-in. Hard cut to the same pen in daylight.
**TEXT:** `10 Angus cattle · 3 cameras · 48 hours`
> Ten Angus cattle, three cameras, two full days and nights. Bovine respiratory disease shows up in behavior days before it shows in the clinic. So we asked: can a camera notice first?

### 2 · Live tracking (0:15–0:40)
🎞 The three camera feeds side by side, same seconds, with our boxes drawn: a colour per cow, `COW_xx` label, live behavior label (lying, feeding, ruminating…), and a faint trail. Day first, then a few seconds of IR night.
*Built from `observations.csv` + the raw clips: real detections, not a mock-up.*
**Rendered:** `outputs/video/animations/three_cams_world_state_feed_rush.mp4`: the three feeds with boxes above the world-state map, the same seconds (8 Nov 16:28–16:30, 4×). It shows the afternoon feed rush: most of the herd feeding at the bunk, COW_09 drinking, and COW_02 (our top risk) lying alone in the alley. Rebuild any window with `video/animations/extract_window.py`.
**TEXT:** `same cow, same colour, every camera`
> This is the system running on our footage. Every box is a detection, every colour is one cow, and the label is what she's doing right now. Same cow, same colour, in all three cameras, by day and by night. Black Angus look almost identical: to an appearance model, two *different* cows score up to 0.94 similarity. So identity can't come from looks alone.

### 3 · The pipeline (0:40–1:00)
✳ The architecture diagram from the brief. Each row lights up as it is named; the orange agent boxes pulse once.
**TEXT:** `measurement first, judgement second`
> So we built it as measurement first, judgement second. A YOLO26 detector, fine-tuned on 216 frames we labelled by hand, finds every cow. ByteTrack follows each one within a camera, and a homography puts every box on the pen floor, in metres.

### 4 · The world state (1:00–1:30)
✳ Top-down pen map (feed bunk on the left, waterer and resting area in the middle, back fence on the right). **Prototype rendered:** `outputs/video/animations/world_state.mp4` (real data, 7 Nov 15:58–16:02). Real cow positions from `cow_state.csv` play back at 20× speed. Each dot has:
- a **heading arrow**;
- a **ring** that is solid when cameras see her, dashed when the position is *predicted*, with an uncertainty halo that grows while she's unseen;
- three small **camera chips** (front · bunk · back) per cow: green = visible, amber = partial (with *"blocked by COW_06"*), grey = not seen.

A camera's view cone sweeps once; for one cow a callout shows `cam1: partial · blocked by COW_06 · facing away · box 1.8 %`.
**TEXT:** `every cow, every second: position · heading · who can see her · how well`
> All of that becomes a world state: every cow, every second. Where she stands, which way she faces, which cameras can see her, whether she's fully visible, partly hidden, or blocked, and by which cow. When no camera sees her, we predict where she is and say how unsure we are. Impossible moves, like a three-metre jump, are caught by physics, not by eye.

### 5 · Deciding where to look (1:30–2:05)
✳ One cow at one moment. Three camera tiles slide in with the real crop from each camera. Under each tile a score builds up, term by term:
`box size × visibility (1.0 visible, 0.4 partial) × facing weight ÷ time offset`
For a **head** check, facing toward the camera counts ×2.0, side-on ×0.8, away ×0.2. For an **identity** check, side-on gets ×1.5. The winning tile lifts and turns orange: *"agent looks here."* The losing tiles grey out with a reason: *"occluded"*, *"facing away"*.
Then a second beat: a funnel of the 28,800 cow-minutes. **17,385 (60 %)** pass the cheap rule check (speed, zone, chewing screen) and need no model call. A calm lying bout on a timeline shows one minute verified and the next ones *carried* (grey hatch), re-checked every 10 min. Only flagged or audited minutes reach the agent.
**TEXT:** `look only where a look can teach you something`
> The world state also tells the agents where it's worth looking. Before an agent spends a model call, it asks: which camera, at which second, can actually answer this question? A head check needs her facing the camera; an identity check wants her side-on; a blocked or distant view is skipped. And most minutes need no model at all: sixty percent pass cheap rule checks, and a cow lying still is verified once and carried. The agents spend their time only where a look can teach them something.

### 6 · Identity: agreement, consensus, anchors (2:05–2:45)
Three short animated beats on the top-down map:
1. ✳ **Agreement.** A front-camera track and a back-camera track land on the same floor spot for the same seconds → a teal line snaps them together (*must-link*). Two tracks seen together, metres apart → a red bar between them (*cannot-link*).
2. ✳ **Consensus.** Three small bar charts (one per camera) show a belief over the 10 cows for one track. They start uneven, then pull toward each other step by step until they agree on one cow. Beside them, a line labelled *disagreement energy* falls and never rises (the Lyapunov function).
3. ✳ **Anchors.** The playback freezes on a real anchor second: every cow gets a 2.5 m ring, none overlap, every camera's chip is green → a lock icon appears on each ID: *"identity set fixed here."*

🖥 End on a real 30-min ID-check sheet (`outputs/stitched/id_agent/<chunk>/id_checks/id_check_*.jpg`).
**TEXT:** `must-link · cannot-link · consensus · anchor`
> Identity comes from geometry and time. If two cameras see a cow on the same spot at the same seconds, it's one cow; if they see two cows together, far apart, it's two. Each camera holds a belief about who it's watching, and those beliefs are pulled into agreement by a consensus protocol. Its disagreement energy can only go down, so IDs settle instead of flickering. And at anchor moments, when every cow stands apart and every camera agrees, we lock the whole set. Every thirty minutes, the identity agent audits it and writes a check sheet like this.

### 7 · Memory cards (2:45–3:15)
✳ A card per cow slides in, built from a real `cards.json` entry: a strip of reference crops (day and IR, all three cameras), traits (*face: all black · ear tag: yellow · white marks: none · build: large*), last seen, and an **appearance prototype** (a small embedding swatch). New notes append as the agent learns them, each with time and source, e.g. *"14:22 · agent · confirmed on cam3 after camera conflict"*.
Then the agent asks `compare_to_cards` for an unknown track: similarity bars to every card, top two highlighted, the decision checked against position before it is accepted.
**TEXT:** `each cow has a memory · every fact has a source`
> Each cow also has a memory card: her best views from every camera, by day and night, the marks that set her apart, and an appearance fingerprint. A person confirms the cards once; after that, every time an agent learns something about a cow, it writes the fact onto her card, with when and how it learned it. When a track is in doubt, the agent compares it to the cards. And because black cattle look alike, the match must still agree with where the cow is.

---

## Part B: behavior and the agents (B)

### 8 · Reading behavior (3:15–3:40)
🎞 A grid of the 5-second labelling crops. ✳ Then a zoom on one muzzle with a small spectrum plot peaking at ~1 Hz.
**TEXT:** `10 classes · ~330 hand-labelled clips + CBVD-5` · `rumination = chewing measured, 55–75 /min`
> For behavior, we labelled about 330 five-second clips ourselves and trained a DINOv2 classifier on them, plus a public cattle dataset. We never guess rumination: we only call it when the jaw is measured chewing at 55 to 75 movements a minute. Cameras vote as experts, and a smoothing model removes impossible jumps, like lying to walking in one second.

### 9 · Agents that are checked (3:40–4:10)
🖥 The success trace scrolling: plan → tool calls → frames → decision. Freeze on the decision. ✳ The self-check rules type onto the screen in orange.
**TEXT:** `stage 1 said drinking (0.69) → agent verified walking (0.96)` · `no evidence, no change → "uncertain"`
*Note: the trace file uses the chunk's own ID (COW_02). After stitching, this cow is COW_09 (see `traces/README.md`). Overlay the stitched ID so the video matches `activity.csv`.*
> Here's the behavior agent at work. Stage one said this cow was drinking. The agent planned, picked the best camera, checked her zone, her track, the frames, and found her walking. And it doesn't get the last word: code checks every decision. A relabel needs two kinds of evidence, and rumination needs measured chewing. If it fails twice, the answer becomes "uncertain", never a silent guess.

### 10 · Level 1 results (4:10–4:25)
✳ The basis bar for the 28,800 cow-minutes (observed 59 %, verified 2 %, unverified 7 %, estimated 2 %, not visible 30 %). 🖥 Then a scroll through `activity.csv`.
**TEXT:** `28,800 cow-minutes · every one says how we know`
> The result: every cow, every minute, for 48 hours, and every minute says how we know it. The agent corrected 335 minutes and refused to guess on 5,000. We'd rather show a gap than invent a cow.

---

## Part C: risk and what it means (C)

### 11 · Why we stitched (4:25–4:40)
✳ A timeline of 33 chunk blocks (Day 1 on three laptops, Day 2 on HPRC). The 2-minute overlaps glow; matching boxes snap together across a seam.
**TEXT:** `33 chunks · 2-min overlaps · 204 of 290 joins from identical boxes`
> 48 hours of 4K didn't fit on one machine in time, so we ran 33 chunks in parallel, and joined the cows where neighbouring chunks saw the same boxes in the same frames.

### 12 · BRD risk (4:40–5:10)
🖥 The ranking page (`outputs/stitched/level2/index.html`), then the COW_02 risk card, then 🎞 its clip `cards/clips/COW_02_1.mp4` with the reason on screen.
**TEXT:** `COW_02 · 0.37 · in the top 3 in 92 % of resamples` · `27 vs 19 min to reach the bunk after feeding`
> For Level 2 we score thirteen indicators from the BRD literature, each against the same pen at the same hours, so weather and feeding cancel out. COW_02 ranks first: she takes 27 minutes to come to the bunk after the herd rushes in, against 19 for the herd, and she walks and eats less. COW_01 and COW_10 follow, and all three hold their place when we resample the data.

### 13 · The honest reading (5:10–5:25)
🖥 The three risk cards side by side, then a calm text card.
**TEXT:** `early suspects, not sick` · `next step: take temperatures, examine`
> But no cow is far from the herd, and the agent confirmed no visible clinical sign. So our call is early suspects, not sick: take their temperatures and examine them. Ten cows over two days can't validate a BRD model, and our system says so.

### 14 · Close (5:25–5:35) · all three
🎞 Team names over the daylight pen; repository and brief named on screen.
**A:** > Measure first.
**B:** > Check everything.
**C:** > And say what you don't know.

---

## Recording checklist

1. **One file per scene per voice:** `vo_04_A.wav`, `vo_09_B.wav`, … Two takes each is plenty. Leave a second of silence before and after.
2. **Room:** a small room with soft furnishings, phone 20–30 cm from the mouth, slightly off-axis. Same phone and room for all three voices so they match.
3. **Pace:** read slower than feels natural. Animations are timed to your voice, so a calm read gives them room.
4. **Footage the edit still needs:** the folder of Label Studio crops (scene 8). Raw clips are already in `input/`.

## Build list (what I make)

| Scene | Asset | Source |
|---|---|---|
| 2 | 3-camera live overlay (boxes, IDs, behavior, trails), day + night | `observations.csv`, `activity_minute.csv`, raw clips |
| 3 | diagram animation | brief SVG |
| 4 | world-state map playback | `cow_state.csv` (position, heading, `predicted`, `pos_sd_m`, `cams_visible`), `observations.csv` (`visibility`, `occluded_by`, `facing`) |
| 5 | best-view scoring + minute funnel | world-state `best_view` weights, real crops, `activity_minute.csv` (`reason`) |
| 6 | must-link / cannot-link, consensus bars + energy, anchor freeze | identity outputs, consensus update rule, `find_anchors.py` criteria |
| 7 | memory card + `compare_to_cards` ranking | `memory/cards.json`, reference crops, prototypes |
| 10–11 | basis bar, chunk timeline | `activity_minute.csv`, `stitch_manifest.json`, `id_map.csv` |

Rendering: animations are HTML/SVG pages in the brief's style, captured frame by frame in headless Chrome and assembled with ffmpeg at 1080p30. Voice levelled to −16 LUFS, captions burned in from this script.
