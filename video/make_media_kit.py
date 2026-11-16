#!/usr/bin/env python3
"""Collect the clips, pictures, data and agent traces a video editor needs into one zip.

  python3 video/make_media_kit.py        ->  outputs/pen6_media_kit.zip

Run python3 video/build.py first (it renders the annotated animations this kit reuses).
Event data: the kit contains pen footage. Keep it within the team / event.
"""
import glob
import os
import re
import shutil
import subprocess
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ST = os.path.join(ROOT, "outputs", "stitched")
VC = os.path.join(ROOT, "outputs", "video")
KIT = os.path.join(ROOT, "outputs", "media_kit", "pen6_media_kit")


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print("missing", src)


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, stderr=subprocess.DEVNULL)


def clip(name):
    p = glob.glob(os.path.join(ROOT, "input", "**", name), recursive=True)
    return p[0]


def shot(page, out, width=1440, full=1, sel=None):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["node", os.path.join(ROOT, "video", "screenshot.js"), page, out, str(width), str(full)] + ([sel] if sel else []),
                   check=True, stdout=subprocess.DEVNULL)


def main():
    if os.path.exists(KIT):
        shutil.rmtree(KIT)
    os.makedirs(KIT)
    K = lambda *p: os.path.join(KIT, *p)

    # ---------------- clips: annotated (real footage + our boxes, IDs, behavior, world state)
    A = os.path.join(VC, "animations")
    cp(os.path.join(A, "three_cams_world_state_feed_rush.mp4"), K("clips", "annotated", "feed_rush_8nov_1628_three_cams_plus_world_state.mp4"))
    cp(os.path.join(A, "three_cams_world_state_cow02.mp4"), K("clips", "annotated", "feed_rush_8nov_1628_COW_02_highlighted.mp4"))
    cp(os.path.join(A, "three_cams_world_state.mp4"), K("clips", "annotated", "afternoon_7nov_1558_three_cams_plus_world_state.mp4"))
    cp(os.path.join(A, "world_state.mp4"), K("clips", "annotated", "world_state_map_7nov_1558_20x.mp4"))
    # single panels cut out of the feed-rush composite (same seconds, 4x real time)
    src = os.path.join(A, "three_cams_world_state_feed_rush.mp4")
    for name, crop in [("cam1_front", "624:468:24:78"), ("cam2_bunk_top", "624:468:648:78"), ("cam3_back", "624:468:1272:78"),
                       ("world_state_map_and_table", "1872:490:24:566")]:
        os.makedirs(K("clips", "annotated", "feed_rush_panels"), exist_ok=True)
        ff("-i", src, "-vf", f"crop={crop}", "-c:v", "libx264", "-crf", "18", "-an", K("clips", "annotated", "feed_rush_panels", f"{name}.mp4"))

    # ---------------- clips: animations (one per scene of our cut, no voice)
    S = os.path.join(VC, "cache", "scenes")
    names = {"s01": "01_cold_open_night_to_day", "s02": "02_live_tracking_feed_rush", "s03": "03_pipeline_diagram",
             "s04": "04_world_state", "s05": "05_where_to_look_best_view_and_minute_funnel", "s06": "06_identity_agreement_consensus_lock_idsheet",
             "s07": "07_memory_card_and_compare_to_cards", "s08": "08_behavior_labels_and_chewing_check", "s09": "09_behavior_agent_trace",
             "s10": "10_level1_basis_and_corrections", "s11": "11_stitching_33_chunks", "s12a": "12a_feed_rush_COW_02_focus",
             "s12b": "12b_brd_risk_ranking", "s13": "13_early_suspects_not_sick", "s14": "14_close"}
    for k, v in names.items():
        cp(os.path.join(S, f"{k}.mp4"), K("clips", "animations", f"{v}.mp4"))

    # ---------------- clips: raw footage (no overlays), 10 s each
    raw = [("day_feed_rush_8nov_1628_cam1_front", "Event20261108162543006.Avi", 147),
           ("day_feed_rush_8nov_1628_cam2_bunk_top", "Event20261108162556017.Avi", 134),
           ("day_feed_rush_8nov_1628_cam3_back", "Event20261108162542003.Avi", 148),
           ("night_ir_8nov_0201_cam1_front", "Event20261108020058006.Avi", 60),
           ("night_ir_8nov_0201_cam2_bunk_top", "Event20261108020110017.Avi", 48),
           ("night_ir_8nov_0201_cam3_back", "Event20261108020047003.Avi", 71)]
    for name, f, off in raw:
        out = K("clips", "raw_footage", f"{name}.mp4")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ff("-vcodec", "hevc", "-ss", str(off), "-i", clip(f), "-t", "10", "-vf", "scale=1600:-2", "-c:v", "libx264", "-crf", "20", "-an", out)

    # ---------------- clips: the risk agent's evidence clips (top 3 cows)
    cp(os.path.join(ST, "level2", "cards", "clips"), K("clips", "risk_evidence"))

    # ---------------- pictures
    for f in glob.glob(os.path.join(ST, "id_agent", "*", "id_checks", "*")):
        chunk = f.split(os.sep)[-3]
        cp(f, K("pictures", "id_check_sheets", chunk, os.path.basename(f)))
    cards = os.path.join(ST, "level2", "cards")
    for c in ("COW_02", "COW_01", "COW_10"):
        cp(os.path.join(cards, f"{c}_indicators.png"), K("pictures", "risk", f"{c}_indicators.png"))
        shot(os.path.join(cards, f"{c}.html"), K("pictures", "risk", f"{c}_risk_card.png"), 1300)
    shot(os.path.join(ST, "level2", "index.html"), K("pictures", "risk", "ranking_page.png"), 1440)
    cp(os.path.join(ST, "level2", "evidence", "img"), K("pictures", "risk_agent_evidence_images"))
    mem = os.path.join(ROOT, "outputs", "agent_d1_1600", "memory")
    cp(os.path.join(mem, "enrollment_sheet.jpg"), K("pictures", "memory_cards", "enrollment_sheet_7nov_1600.jpg"))
    cp(os.path.join(mem, "crops"), K("pictures", "memory_cards", "reference_crops"))
    cp(os.path.join(mem, "cards.json"), K("pictures", "memory_cards", "cards.json"))
    cp(os.path.join(S, "s08", "tiles"), K("pictures", "behavior_label_crops"))
    cp(os.path.join(S, "s05", "crops"), K("pictures", "best_view_crops_8nov"))
    shot(os.path.join(ROOT, "docs", "TECHNICAL_BRIEF.html"), K("pictures", "architecture_diagram.png"), 1100, 0, "figure")
    # stills of every animation (1080p), at 85 % of each scene
    for k, v in names.items():
        p = os.path.join(S, f"{k}.mp4")
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
        os.makedirs(K("pictures", "animation_stills"), exist_ok=True)
        ff("-ss", f"{d * 0.85:.2f}", "-i", p, "-frames:v", "1", K("pictures", "animation_stills", f"{v}.png"))

    # ---------------- traces
    cp(os.path.join(ST, "final", "traces"), K("traces", "behavior_agent"))
    picks = [("success_relabel_camera_conflict_7nov_0830", "afeef_training_c01", "20261107_083039_camera_conflict_COW_08"),
             ("success_relabel_id_jump_7nov_1036", "afeef_training_c02", "20261107_103655_id_jump_COW_07"),
             ("refused_uncertain_duplicate_COW_05_8nov_1620", "hprc_day2_c18", "20261108_162020_weak_id_COW_05"),
             ("night_confirm_id_jump_8nov_0144", "hprc_day2_c02", "20261108_014445_id_jump_COW_01"),
             ("audit_30min_8nov_1630", "hprc_day2_c18", "20261108_163000_id_audit_COW_01_COW_02")]
    for label, chunk, stem in picks:
        tdir = os.path.join(ST, "id_agent", chunk, "traces")
        dst = K("traces", "identity_agent", label)
        for ext in (".md", ".json"):
            cp(os.path.join(tdir, stem + ext), os.path.join(dst, stem + ext))
        for img in re.findall(r"\]\((img/[^)]+)\)", open(os.path.join(tdir, stem + ".md")).read()):
            cp(os.path.join(tdir, img), os.path.join(dst, img))
        cp(os.path.join(ST, "id_agent", chunk, "id_checks", "legend.csv"), os.path.join(dst, "legend_chunk_ids_to_global.csv"))
    cp(os.path.join(ST, "level2", "agent", "traces"), K("traces", "risk_agent"))
    cp(os.path.join(ST, "level2", "evidence", "facts.jsonl"), K("traces", "risk_agent", "evidence_facts.jsonl"))

    # ---------------- data
    for f in ("activity.csv", "activity_minute.csv", "time_budgets.csv"):
        cp(os.path.join(ST, "final", f), K("data", "level1", f))
    for f in ("brd_risk.csv", "indicators.csv", "contributions.csv", "features_summary.csv", "features_hourly.csv", "risk_meta.json"):
        cp(os.path.join(ST, "level2", f), K("data", "level2", f))
    cp(os.path.join(ST, "id_map.csv"), K("data", "stitching", "id_map.csv"))
    cp(os.path.join(ST, "stitch_manifest.json"), K("data", "stitching", "stitch_manifest.json"))
    for w in ("win_feed", "win_world"):
        for f in ("world.json", "boxes.json"):
            cp(os.path.join(VC, "cache", w, f), K("data", "animation_windows", w, f))

    # ---------------- docs
    cp(os.path.join(ROOT, "docs", "TECHNICAL_BRIEF.pdf"), K("docs", "TECHNICAL_BRIEF.pdf"))
    cp(os.path.join(ROOT, "docs", "VIDEO_SCRIPT.md"), K("docs", "VIDEO_SCRIPT_our_draft.md"))
    cp(os.path.join(ROOT, "video", "MEDIA_KIT_README.md"), K("README.md"))

    zp = os.path.join(ROOT, "outputs", "pen6_media_kit.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(KIT):
            for f in fs:
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, os.path.dirname(KIT)))
    print("wrote", zp, f"{os.path.getsize(zp) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
