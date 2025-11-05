EdgeWisePersona — Project Design (No-Code)

This document is a design-only blueprint for your personalization study using the EdgeWisePersona dataset. It covers phases, roles, artifacts, metrics, experiments, and decision rules — with no code. You can paste this into your repo as DESIGN.md.

0) Purpose & Scope

Goal: Determine whether lightweight adaptation methods help an LLM better fit individual users (not role-play personas) and show how the model’s behavior evolves over time.

Methods compared:

LoRA (PEFT) adapters

Soft/Prefix prompts (a.k.a. “extra input tokens”)

(Baseline) Prompt-only (fixed template; no learning)

(Optional extension) DSPy-controlled prompts + simple bandit

Two evaluation tracks:

Task Prediction: utility/quality on the dialogue tasks in EdgeWisePersona

Persona Prediction from Query: retrieval/classification — “Which user/persona is this query most like?”

1) Dataset Strategy

Source: EdgeWisePersona (dialogues grouped by persona_id ≈ “user”).
Unit of analysis: Persona-as-User. Each persona_id is treated as a distinct user.

Normalization (conceptual):

Clean whitespace and normalize message fields (role, text).

Discard degenerate/ultra-short exchanges that are uninformative.

Preserve original interaction order within each user.

Splits per user (time-aware):

Train: first ~60% of that user’s sessions

Val: next ~20%

Test: final ~20%

Online tiny-batch: a small subset (e.g., 8–12 examples) reserved for per-user micro updates (used only by adaptive arms)

Variant splits for robustness:

Cold-start test: first 3–5 turns of users that are unseen during training

Cluster splits: group users by style features (verbosity, sentiment, formality) for cluster-level adaptation

Artifacts:

dialogs_clean.jsonl (cleaned dataset)

edgesplits.json (per-user indices for train/val/test/online)

cluster_map.json (optional; user → cluster id)

2) Models & Conditions

Base model: A single small–medium open LLM (e.g., a 7–8B instruct variant).
Why: Keeps local training feasible and isolates the effect of adaptation.

Arms (conditions):

Baseline (Prompt-Only): Fixed, concise helper prompt; no personalization.

Soft/Prefix (All-Persona): One global learned prefix for all users.

Soft/Prefix (Per-Persona): One prefix per user (or per cluster with tiny user deltas).

LoRA (All-Persona): One global LoRA adapter trained on all users together.

LoRA (Per-Persona): One adapter per user (or per cluster with tiny user deltas).

(Optional) DSPy/Bandit: A fixed set of 3–5 prompt templates; per-user bandit selects template + memory policy online.

Compute boundaries (design constraints):

Keep sequence length modest (≈1k) for parity across arms.

Limit per-user train tokens for micro-updates to mimic realistic online personalization.

3) Tasks & Prompts (Conceptual)

Task Prediction (primary):

Input: Most recent user message(s) + available context.

Output: Assistant reply matching reference quality and style constraints.

Persona Prediction from Query (secondary):

Input: A short user query or early-turn snippet.

Output: Predicted user/persona id (top-1) or ranked list.

Prompt policy (for parity):

Use a minimal, neutral system instruction across all arms.

For adaptive arms, conditioning occurs via prefix/adapters (not by leaking persona text directly into prompts for the eval), ensuring we test learned personalization.

4) Evaluation Framework
4.1 Primary Metrics (Task Prediction)

Utility/Quality: ROUGE-L (or task-appropriate: BLEU, exact match/F1 for QA; rubric for advice tasks)

User-Fit / Style Match:

Verbosity error (difference vs. user’s typical length)

Sentiment/formality alignment (classifier-based)

Embedding similarity between output and the user’s historical style

Stability & Safety Checks:

Factuality proxy or hallucination rate (if applicable)

Output length consistency within accepted bounds

Aggregation views:

Macro-average across users (overall)

Per-user scores (to detect who benefits/hurts)

Cold-start vs. warm-start comparisons

4.2 Secondary Metric (Persona Prediction from Query)

Accuracy@1 over test split

Confusion matrix across personas (or clusters)

Success criteria (example thresholds):

Macro lift over baseline with 95% CI not crossing zero

≥60% of users benefit on at least one adaptive arm

No notable degradation in safety/consistency relative to baseline

5) Experimental Phases
Phase A — Data Readiness

Inputs: raw dataset
Outputs: cleaned dialogs, per-user time-aware splits, optional cluster map

Checks:

Coverage: each user has sufficient train/val/test items

Balance: ensure no user dominates the global training

Phase B — Baseline Establishment

Goal: Set the “no personalization” reference.

Procedure:

Fix a single neutral prompt and generation settings.

Evaluate on val (for sanity) then on test.

Record macro metrics and per-user breakdowns.

Store configuration as Baseline v1.0 (frozen).

Artifacts: baseline_eval.json, per-user table, diagnostic notes

Phase C — Global Adaptation (All-Persona)

Arms: Soft/Prefix (All), LoRA (All)

Objective: Learn shared adaptations that may still help many users.

Procedure (shared design):

Train on all users’ train data.

Early-stop on macro val metric.

Run single global eval on test (one score & per-user table).

Compare to baseline (macro & per-user deltas).

Artifacts: prefix_all_eval.json, lora_all_eval.json, comparative plots

Phase D — Personalized Adaptation (Per-Persona)

Arms: Soft/Prefix (Per-Persona), LoRA (Per-Persona)

Objective: Learn user-specific micro-adapters or prefixes.

Procedure (per user):

Initialize from the global model (or a cluster-level model if using clusters).

Apply tiny online updates using that user’s online slice (token budget capped).

Freeze, then run that user’s test examples.

Save per-user metrics and a change log (e.g., prefix shift magnitude or adapter norm).

Outputs: 50 separate eval results for each persona, plus a merged table.

Artifacts:

per_user/<user_id>/eval.json for each arm

Aggregated per_user_summary.csv with deltas vs baseline

Phase E — Persona Prediction from Query

Goal: Assess whether the system can identify which user a new query resembles.

Design:

Build a compact profile representation per user from train messages (text centroid or embedding average).

Evaluate top-1 accuracy on held-out queries from test.

Compare performance with vs without adaptive conditioning to detect whether adaptation alters separability.

Artifacts: persona_pred_eval.json, confusion matrix figure

6) Analysis Plan

Core visualizations:

Per-user lift violin/box plots — ∆(metric) vs baseline for each arm

Cold-start vs warm-start curves — performance as interactions increase

Adapter/Prefix change magnitude over online updates — stability vs. drift

Persona prediction confusion matrix — cluster patterns and hardest-to-separate users

Representative before/after qualitative snippets — illustrate style alignment and clarity gains

Statistical tests:

Paired bootstrap or paired t-tests across users for macro lift

Sign test on “helped vs. hurt” users per arm

Ablations:

LoRA target modules (attention-only vs. broader)

Prefix length (short vs. longer)

Online-update budget (few vs. more examples)

Decision rules:

If All-Persona arm yields stable macro gains and low variance, prefer it for simplicity.

If Per-Persona arms show large gains for a subset but instability for others, adopt a cluster-first strategy: train cluster adapters/prefixes, then allow tiny user deltas for heavy users only.

If Persona Prediction accuracy improves with adaptation, it suggests stronger user signal capture; if it collapses, investigate overfitting to surface form.

7) Risks, Ethics, and Guardrails

Privacy:

Treat “user” features as preferences only; avoid PII.

Provide a conceptual “Reset my profile” capability.

Safety:

Maintain a non-personalized safety layer (system prompts/filters) untouched by personalization.

Monitor hallucination proxies and factuality drift; enforce floors equal to baseline.

Fairness:

Compare lifts across clusters to ensure no group regresses.

Cap per-user updates; add rollback if validation quality drops.

Reproducibility:

Fix seeds, track configs, log dataset hashes, freeze baselines.

Keep an experiment registry table: arm, split, date, commit, result.

8) Deliverables & Reporting

Tables:

Baseline vs. All-Persona vs. Per-Persona (macro metrics)

Per-user performance with deltas and confidence intervals

Persona prediction accuracy (overall + per cluster)

Figures:

Violin/box plots of per-user lifts

Learning curves (if you simulate multi-turn adaptation)

Confusion matrix for persona prediction

Stability plots (adapter/prefix change magnitudes over time)

Narrative (short paper outline):

Introduction: personalization as user-fit, not role-play

Methods: dataset, splits, arms (LoRA/prefix), evaluation plan

Results: macro gains, who benefits, stability, cold-start vs warm-start

Persona Prediction: how adaptation affects separability

Discussion: trade-offs (auditability vs strength), recommended deployment path

Limitations & Future Work: meta-learning, federated updates, richer user states

9) Phase Checklist (At-a-Glance)

A. Data Ready → cleaned dialogs, per-user splits, (optional) clusters

B. Baseline → macro & per-user metrics locked

C. All-Persona → global prefix/adapter results vs baseline

D. Per-Persona → 50 evals aggregated, stability assessed

E. Persona Prediction → accuracy + confusion matrix

F. Analysis & Report → plots, tables, decisions, next steps

10) Success Criteria & Next Decisions

Success if:

Adaptive arms outperform baseline macro-wise with significance, and

≥60% of users see improvements without safety regressions, and

Cold-start is competitive using cluster-level adaptation.

Next decisions:

Deploy path: choose All-Persona model + cluster adapters/prefixes, unlock tiny per-user micro-updates for frequent users.

Further research: add DSPy/Bandit for transparent prompt-space evolution; explore logit/attention bias with a compact user embedding for finer control.