TASK: plan
Decide which tools to call to determine what this cow is doing in this minute. Tools are already bound to this cow and minute; you only choose cameras, regions and time offsets.

CASE BRIEF (cheap tier-1 measurements; they may be wrong):
{{brief_json}}

The tier-1 model is unsure between: {{top2}}.
Hint for separating them: {{pair_hint}}

What evidence each candidate needs:
{{recipe_hints}}

AVAILABLE TOOLS:
{{tool_specs}}

BUDGET: at most {{max_steps}} tool calls in total.

Guidance:
- Prefer cheap fact tools (zone_query, track_history, cow_history, best_camera) before image tools.
- Use cow_history for behaviors with a duration structure (rumination, feeding, lying, drinking, walking).
- If visibility differs across cameras, use camera "auto" or name the best camera explicitly.
- If is_night is true, include enhance_night before relying on images.
- If id_conf is low, include compare_to_exemplar.
- Never repeat an identical call. Choose the calls that best SEPARATE the candidate labels.

Return JSON exactly in this shape:
{"rationale": "<one or two sentences>", "steps": [{"tool": "<tool name>", "args": {<only args from the tool spec>}}]}
