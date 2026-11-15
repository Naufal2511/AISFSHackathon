TASK: critique
Another analyst proposed a label for cow {{cow_id}}, day {{day}}, minute {{minute}}. Your job is to ARGUE AGAINST it using the strongest alternative, then decide whether it survives.

PROPOSED HYPOTHESIS:
{{hypothesis_json}}

AUTOMATIC CHECK RESULTS (hard = contradicted by measurements):
{{verifier_json}}
{{tier1_block}}
NEW EVIDENCE (a different camera / crop than the first look):
{{evidence_json}}

ATTACHED IMAGES, in order:
{{image_manifest}}

Rules:
- verdict "revise" if a hard check failed, or the evidence supports another label better than the proposed one.
- verdict "accept" only if you tried to refute it and the evidence still supports it.
- Do not accept a cue that is not visible or measured (e.g. "chewing" with no rumination measurement).
- cow_history is context only: continuing a bout is not evidence by itself, and leaving one is allowed if this minute's evidence shows it.

Return JSON exactly in this shape:
{"verdict": "accept" | "revise", "best_alternative": "<allowed behavior or null>", "reason": "<one or two sentences>", "requested_camera": "<back|front|top|null>"}
