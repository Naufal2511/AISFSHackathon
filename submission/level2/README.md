# Level 2 · BRD Risk Ranking and Decision Evidence

**Merck–TAMU AISFS Hackathon 2026 · Pen 6 · 10 Angus cattle · 48 h**

This folder contains the complete Level 2 BRD screening results: `brd_risk.csv`, risk cards, indicator contributions, agent reasoning traces, and the **video evidence clips** used to reach the clinical decisions for the top-risk cattle.

> **Key finding:** COW_02, COW_01, and COW_10 are **early suspects, not sick**. The reasoning agent checked daylight close-ups and confirmed **no visible clinical sign** (normal breathing rate, no drooped ears, no confirmed nasal discharge). They are flagged solely on early sickness behavior deviations (appetite and bunk attendance down, latency up, lying increased).

---

## 📊 BRD Risk Ranking (`brd_risk.csv`)

| Rank | Cow | Risk Score | Top-3 Stability | Confidence | Top Reasons (vs herd at same hours) | Relevant Evidence Clips (Click to view) |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **1** | **COW_02** | `0.372` | 92% | high | Slower to reach bunk (27 vs 19 min, z +1.2); walks less (87.5 vs 101 m/h, z -1.0); feeding share down (7% vs 10%, z -0.6) | 🎬 [Clip 1: Feed latency](cards/clips/COW_02_1.mp4)<br>🎬 [Clip 2: Closest view](cards/clips/COW_02_2.mp4) |
| **2** | **COW_01** | `0.325` | 67% | medium | Feeding share down (4% vs 8%, z -1.1); short bunk visits (5.8 vs 9.5 min, z -1.3); lying increased (57% vs 51%, z +1.4) | 🎬 [Clip 1: Bunk feeding](cards/clips/COW_01_1.mp4)<br>🎬 [Clip 2: Closest view](cards/clips/COW_01_2.mp4) |
| **3** | **COW_10** | `0.259` | 88% | high | Short bunk visits (6.5 vs 9.5 min, z -1.1); longest lying bout (139 vs 100 min, z +1.9); bunk visits falling day-to-day | 🎬 [Clip 1: Bunk visit](cards/clips/COW_10_1.mp4)<br>🎬 [Clip 2: Closest view](cards/clips/COW_10_2.mp4) |
| 4 | COW_09 | `0.185` | 1% | low | Day-to-day feeding change (-24% vs -12%); herd desync | *(Not flagged)* |
| 5 | COW_03 | `0.163` | 6% | low | Meal interval gap (532 vs 357 min, z +2.4); fewer visits | *(Not flagged)* |
| 6 | COW_04 | `0.149` | 24% | low | Activity down; time alone increased | *(Not flagged)* |
| 7 | COW_05 | `0.149` | 15% | low | Short bunk visits (7.5 vs 9.5 min); feed latency | *(Not flagged)* |
| 8 | COW_07 | `0.132` | 7% | low | Sudden movement jerk rate candidate; longest lying bout | *(Not flagged)* |
| 9 | COW_06 | `0.121` | 0% | low | Waterer visit rate down; resting respiratory rate | *(Not flagged)* |
| 10 | COW_08 | `0.101` | 1% | low | Meal gap; normal herd baseline | *(Not flagged)* |

---

## 🎬 Decision Evidence Clips (Top 3 Cattle)

These 6 clips (104 frames / 12 seconds each, ~10 MB total) are the exact video clips retrieved and cited by the risk evaluation pipeline to assess behavior and visual signs:

### COW_02 (Rank 1 · Risk 0.372)
- **Behavior deviation clip**: [`cards/clips/COW_02_1.mp4`](cards/clips/COW_02_1.mp4)  
  *Camera 3 · 2026-11-07 00:04:24 to 00:04:36* — Captures feed call latency and out-of-step behavior during a herd feeding rush.
- **Closest view clip**: [`cards/clips/COW_02_2.mp4`](cards/clips/COW_02_2.mp4)  
  *Camera 1 · 2026-11-07 17:49:54 to 17:50:06* — Highest resolution Daylight close-up inspected by the agent to assess ears, head carriage, and muzzle.
- **Risk card**: [`cards/COW_02.html`](cards/COW_02.html) · [Indicators PNG](cards/COW_02_indicators.png)

### COW_01 (Rank 2 · Risk 0.325)
- **Behavior deviation clip**: [`cards/clips/COW_01_1.mp4`](cards/clips/COW_01_1.mp4)  
  *Camera 1 · 2026-11-08 09:44:24 to 09:44:36* — Feeding share deficit and short visit duration at the bunk.
- **Closest view clip**: [`cards/clips/COW_01_2.mp4`](cards/clips/COW_01_2.mp4)  
  *Camera 1 · 2026-11-07 07:33:12 to 07:33:24* — Close frontal profile inspected for nasal discharge and eye signs.
- **Risk card**: [`cards/COW_01.html`](cards/COW_01.html) · [Indicators PNG](cards/COW_01_indicators.png)

### COW_10 (Rank 3 · Risk 0.259)
- **Behavior deviation clip**: [`cards/clips/COW_10_1.mp4`](cards/clips/COW_10_1.mp4)  
  *Camera 1 · 2026-11-07 16:39:24 to 16:39:36* — Bunk visit duration deficit at the feed bunk.
- **Closest view clip**: [`cards/clips/COW_10_2.mp4`](cards/clips/COW_10_2.mp4)  
  *Camera 2 · 2026-11-08 16:32:39 to 16:32:51* — Side/profile close view inspected for respiratory effort and ear position.
- **Risk card**: [`cards/COW_10.html`](cards/COW_10.html) · [Indicators PNG](cards/COW_10_indicators.png)

---

## 📁 Key Files Reference

- [`brd_risk.csv`](brd_risk.csv): Summary risk table with scores, ranks, z-deviations, and clip references.
- [`indicators.csv`](indicators.csv): 13 behavioral indicators computed for each cow vs. the herd at identical hours.
- [`contributions.csv`](contributions.csv): Mathematical breakdown of how each indicator shifted the prior log-odds.
- [`features_hourly.csv`](features_hourly.csv) & [`features_summary.csv`](features_summary.csv): Hourly and aggregated sensor features across all 48 hours.
- [`agent/traces/`](agent/traces/): Full JSON reasoning transcripts from Claude Opus 4.8 evaluating each cow.
- [`evidence/facts.jsonl`](evidence/facts.jsonl): Chronological fact database queried by the reasoning agent.
- [`risk_meta.json`](risk_meta.json): Run metadata, model versions, time spans, and clip timestamps.
