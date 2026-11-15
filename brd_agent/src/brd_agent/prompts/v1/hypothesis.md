TASK: hypothesis
Decide the dominant behavior of cow {{cow_id}} during minute {{minute}} of day {{day}}, using ONLY the evidence below.

CONTEXT:
- night: {{is_night}}
- previous minute's final label: {{prev_label}}
{{tier1_block}}{{hint_block}}
MEASUREMENTS (tool outputs, latest per tool):
{{evidence_json}}

ATTACHED IMAGES, in order:
{{image_manifest}}

Return JSON exactly in this shape:
{"label": "<one allowed behavior>",
 "confidence": <0..1>,
 "cues": ["<short cue you actually observed or measured>", "..."],
 "alternatives": [["<other behavior>", <0..1>]],
 "needs": "<what extra evidence would settle it, or null>"}
