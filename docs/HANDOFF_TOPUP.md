# Pen 6 annotation: top-up

This adds **new tasks only** to the two Label Studio projects you set up from `pen6_annotation.zip`.
Nothing here replaces or re-imports anything you already have, so labels you've already made are safe.

What's inside:
- `behavior_tasks.json`: new cow clips. The first 75 are the **high-quality batch** (25 per camera,
  spread over day/night and both days, sharper and contrast-enhanced at night; `priority` = 0).
  After them come the remaining earlier clips (`priority` = 1).
- `detection_tasks.json`: a few new cow-box frames with pre-labels.
- `data/...`: only the new image and clip files. Their names don't clash with your existing files.

---

## Prompt to paste into Claude Code (run it inside your `pen6_annotation` folder)

```text
I'm annotating cattle data in a LOCAL Label Studio. My folder `pen6_annotation` (this working
directory) was set up from pen6_annotation.zip: Label Studio runs via ./start_labelstudio.sh
(or start_labelstudio.bat), and it has two projects: a cow-box project (detection_tasks.json)
and a behavior project (behavior_tasks.json). I already labeled some tasks; those labels must
not be lost or overwritten.

I received pen6_annotation_topup.zip with NEW tasks only. Please:
1. Unzip it to a temporary folder and MERGE its `data/` folder into my `pen6_annotation/data/`,
   copying only files that don't exist yet (never overwrite an existing file). Report how many
   files were added and confirm none were overwritten.
2. Keep its `detection_tasks.json` and `behavior_tasks.json` next to my folder as
   `detection_tasks_topup.json` and `behavior_tasks_topup.json`.
   Do NOT replace my original task files, and do NOT delete or re-import any tasks.
3. Tell me exactly what to click in Label Studio to IMPORT each topup file into the matching
   existing project (Import button, upload the file). Importing only adds tasks.
4. Tell me how to put the high-quality batch first: in the behavior project's Data Manager,
   show the `priority` column and sort ascending (0 first), so "Label All Tasks" serves those first.
5. Don't send any of this data to the internet or any cloud service. It's event data and stays local.
```

## Or by hand (2 minutes)

1. Unzip, then copy the `data/` folder over your `pen6_annotation/data/`, choosing **"Skip"/"Keep both"
   for existing files, never "Replace"** (there shouldn't be any clashes anyway).
2. In each project, click **Import** and upload the matching file. Don't delete anything first.
3. Behavior project → Data Manager → column settings → show **priority** → sort ascending.

Labeling rules and hotkeys are unchanged (see the original HANDOFF.md). Export → JSON from both
projects when done and send the two files back.
