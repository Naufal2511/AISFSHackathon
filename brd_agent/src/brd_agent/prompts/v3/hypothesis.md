TASK: hypothesis
Decide the dominant behavior of cow {{cow_id}} during minute {{minute}} of day {{day}}, using ONLY the evidence below.

CONTEXT:
- night: {{is_night}}
- previous minute's final label: {{prev_label}}
{{tier1_block}}{{hint_block}}
MEASUREMENTS (tool outputs, latest per tool; cow_history is context, not proof):
{{evidence_json}}

ATTACHED IMAGES, in order:
{{image_manifest}}

Rules: the tier-1 line above (if any) is the upstream detector's CLAIM; confirm or correct it from the evidence. At night, cap confidence at 0.85 and prefer other_not_visible over a guess.

Return JSON exactly in this shape:
{"label": "<one allowed behavior>",
 "confidence": <0..1>,
 "cues": ["<short cue you actually observed or measured>", "..."],
 "alternatives": [["<other behavior>", <0..1>]],
 "tags": ["<0-3 optional tags from the TAG LIST, only if visibly supported>"],
 "needs": "<what extra evidence would settle it (name a camera, a time offset or a measurement), or null>",
 "rationale": "<one or two sentences: why this label, citing the cues and why the main alternative is weaker>"}
