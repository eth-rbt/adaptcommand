# EdgeWisePersona Personalization Study — Implementation Plan

This plan breaks down the experimental phases from design.md into concrete, actionable steps.

---

## Phase A — Data Readiness

**Goal:** Prepare the EdgeWisePersona dataset with proper cleaning, splits, and optional clustering.

### A.1 Dataset Acquisition & Loading
- [ ] Download/acquire EdgeWisePersona dataset
- [ ] Create `data/raw/` directory structure
- [ ] Load raw dialogues and verify format (fields: persona_id, role, text, timestamps)
- [ ] Document data schema and statistics (# personas, # dialogues per persona, avg length)

### A.2 Data Cleaning & Normalization
- [ ] Implement cleaning function:
  - [ ] Normalize whitespace (strip, deduplicate spaces)
  - [ ] Standardize message fields (role, text)
  - [ ] Filter degenerate exchanges (length < threshold, e.g., <10 chars)
- [ ] Preserve original interaction order per persona_id
- [ ] Save cleaned data as `data/cleaned/dialogs_clean.jsonl`
- [ ] Log cleaning statistics (# removed, # kept)

### A.3 Per-User Time-Aware Splits
- [ ] For each persona_id, sort dialogues by timestamp/order
- [ ] Implement split function (60% train / 20% val / 20% test)
- [ ] Reserve small online batch (8-12 examples) from train for micro-updates
- [ ] Create `data/splits/edgesplits.json`:
  ```json
  {
    "user_<id>": {
      "train": [indices],
      "val": [indices],
      "test": [indices],
      "online": [indices]
    }
  }
  ```
- [ ] Validate splits:
  - [ ] Each user has minimum examples in each split
  - [ ] No overlap between splits
  - [ ] Train comes before val comes before test temporally

### A.4 Coverage & Balance Checks
- [ ] Compute per-user statistics (train/val/test counts)
- [ ] Check no single user dominates (e.g., max user ≤ 10% of total data)
- [ ] Identify and handle edge cases (users with very few examples)
- [ ] Generate `data/splits/split_stats.json` summary

### A.5 Optional: Cold-Start & Cluster Splits
- [ ] Extract cold-start test set:
  - [ ] First 3-5 turns from held-out users
  - [ ] Save as `data/splits/cold_start_test.jsonl`
- [ ] Compute user style features:
  - [ ] Verbosity (avg message length)
  - [ ] Sentiment (using classifier or heuristics)
  - [ ] Formality score
- [ ] Cluster users (k-means, k=5-10 clusters)
- [ ] Save `data/splits/cluster_map.json`:
  ```json
  {
    "user_<id>": <cluster_id>
  }
  ```

**Phase A Deliverables:**
- `data/cleaned/dialogs_clean.jsonl`
- `data/splits/edgesplits.json`
- `data/splits/split_stats.json`
- `data/splits/cluster_map.json` (optional)
- `data/splits/cold_start_test.jsonl` (optional)

---

## Phase B — Baseline Establishment

**Goal:** Establish the no-personalization reference point.

### B.1 Model Setup
- [ ] Select base model (e.g., Llama-3.2-7B-Instruct or Mistral-7B-Instruct-v0.3)
- [ ] Download and cache model locally
- [ ] Verify model loads correctly with inference framework (vLLM, HF Transformers)
- [ ] Document model config (max_tokens, temperature, top_p)

### B.2 Baseline Prompt Design
- [ ] Create fixed, neutral system prompt template
- [ ] Example: "You are a helpful assistant. Respond to the user's query."
- [ ] Set generation parameters:
  - [ ] max_new_tokens: 256
  - [ ] temperature: 0.7
  - [ ] top_p: 0.9
  - [ ] Sequence length cap: ~1024 tokens
- [ ] Save config as `configs/baseline_v1.0.json`

### B.3 Baseline Evaluation Pipeline
- [ ] Implement evaluation script `scripts/eval_baseline.py`:
  - [ ] Load test split for all users
  - [ ] For each test example:
    - [ ] Format with baseline prompt
    - [ ] Generate model response
    - [ ] Compute metrics vs reference
  - [ ] Aggregate per-user and macro scores
- [ ] Implement metrics:
  - [ ] ROUGE-L (primary utility metric)
  - [ ] Verbosity error (|output_len - user_avg_len|)
  - [ ] Embedding similarity (sentence-transformers)
  - [ ] Output length consistency (std dev)

### B.4 Validation Run
- [ ] Run baseline on validation split first (sanity check)
- [ ] Review outputs qualitatively (sample 10-20 examples)
- [ ] Adjust prompt or generation params if needed
- [ ] Document any adjustments

### B.5 Test Run & Freeze
- [ ] Run baseline evaluation on full test split
- [ ] Save results:
  - [ ] `results/baseline/baseline_eval.json` (macro metrics)
  - [ ] `results/baseline/per_user_scores.csv` (per-user breakdown)
  - [ ] `results/baseline/sample_outputs.jsonl` (qualitative examples)
- [ ] Freeze baseline config (version control, no further changes)
- [ ] Document baseline performance in README or report

**Phase B Deliverables:**
- `configs/baseline_v1.0.json`
- `scripts/eval_baseline.py`
- `results/baseline/baseline_eval.json`
- `results/baseline/per_user_scores.csv`
- `results/baseline/sample_outputs.jsonl`

---

## Phase C — Global Adaptation (All-Persona)

**Goal:** Train shared adaptations that help many users without personalization.

### C.1 Soft/Prefix (All-Persona)

#### C.1.1 Setup Prefix Training
- [ ] Choose prefix implementation (HF PEFT or custom)
- [ ] Configure prefix parameters:
  - [ ] Prefix length: 20-50 tokens
  - [ ] Trainable params: only prefix embeddings
  - [ ] Base model frozen
- [ ] Create training config `configs/prefix_all.json`

#### C.1.2 Training Data Preparation
- [ ] Aggregate all users' train splits
- [ ] Format as conversational pairs (input context → target response)
- [ ] Create dataloaders with batching

#### C.1.3 Training
- [ ] Implement training script `scripts/train_prefix_all.py`
- [ ] Train with:
  - [ ] Early stopping on macro val metric (ROUGE-L)
  - [ ] Learning rate: 1e-4 to 5e-4
  - [ ] Batch size: 4-8 (depending on GPU)
  - [ ] Max epochs: 3-5
- [ ] Log training curves (loss, val metric)
- [ ] Save best checkpoint to `models/prefix_all/`

#### C.1.4 Evaluation
- [ ] Load trained prefix
- [ ] Run evaluation on test split (same script as baseline)
- [ ] Save results:
  - [ ] `results/prefix_all/prefix_all_eval.json`
  - [ ] `results/prefix_all/per_user_scores.csv`
- [ ] Compute deltas vs baseline (macro and per-user)

### C.2 LoRA (All-Persona)

#### C.2.1 Setup LoRA Training
- [ ] Configure LoRA parameters:
  - [ ] Target modules: q_proj, v_proj (attention-only) or broader
  - [ ] Rank: 8-16
  - [ ] Alpha: 16-32
  - [ ] Dropout: 0.05
- [ ] Create training config `configs/lora_all.json`

#### C.2.2 Training
- [ ] Implement training script `scripts/train_lora_all.py`
- [ ] Use same data aggregation as prefix training
- [ ] Train with early stopping on val metric
- [ ] Save best checkpoint to `models/lora_all/`

#### C.2.3 Evaluation
- [ ] Load trained LoRA adapter
- [ ] Run evaluation on test split
- [ ] Save results:
  - [ ] `results/lora_all/lora_all_eval.json`
  - [ ] `results/lora_all/per_user_scores.csv`
- [ ] Compute deltas vs baseline

### C.3 Comparative Analysis
- [ ] Create comparison script `scripts/compare_all_persona.py`
- [ ] Generate plots:
  - [ ] Macro metric comparison (baseline vs prefix vs LoRA)
  - [ ] Per-user lift distributions (box plots)
  - [ ] Metric-by-metric breakdown
- [ ] Save figures to `results/figures/all_persona_comparison/`
- [ ] Document findings in `results/all_persona_summary.md`

**Phase C Deliverables:**
- `models/prefix_all/` (trained prefix checkpoint)
- `models/lora_all/` (trained LoRA checkpoint)
- `results/prefix_all/prefix_all_eval.json`
- `results/lora_all/lora_all_eval.json`
- `results/figures/all_persona_comparison/`
- `results/all_persona_summary.md`

---

## Phase D — Personalized Adaptation (Per-Persona)

**Goal:** Train user-specific micro-adapters or prefixes with online updates.

### D.1 Soft/Prefix (Per-Persona)

#### D.1.1 Setup Per-User Prefix Training
- [ ] Create per-user training pipeline `scripts/train_prefix_per_user.py`
- [ ] For each user:
  - [ ] Initialize from global prefix (if available) or base model
  - [ ] Load that user's online slice (8-12 examples)
  - [ ] Set token budget cap (e.g., max 500 tokens total)

#### D.1.2 Micro-Update Training Loop
- [ ] For each user_id:
  - [ ] Load user's online data
  - [ ] Fine-tune prefix for 1-3 epochs (very limited updates)
  - [ ] Track prefix shift magnitude (L2 norm of change)
  - [ ] Freeze updated prefix
  - [ ] Save to `models/prefix_per_user/<user_id>/`

#### D.1.3 Per-User Evaluation
- [ ] For each user_id:
  - [ ] Load personalized prefix
  - [ ] Evaluate on that user's test split only
  - [ ] Save `results/prefix_per_user/<user_id>/eval.json`
  - [ ] Log prefix change metrics

#### D.1.4 Aggregation
- [ ] Collect all per-user results
- [ ] Compute macro average across users
- [ ] Create `results/prefix_per_user/per_user_summary.csv`
- [ ] Include columns: user_id, ROUGE-L, verbosity_error, delta_vs_baseline, prefix_shift_norm

### D.2 LoRA (Per-Persona)

#### D.2.1 Setup Per-User LoRA Training
- [ ] Create per-user training pipeline `scripts/train_lora_per_user.py`
- [ ] Same structure as prefix per-user training

#### D.2.2 Micro-Update Training Loop
- [ ] For each user_id:
  - [ ] Initialize LoRA from global adapter or base model
  - [ ] Load user's online slice
  - [ ] Fine-tune for 1-3 epochs with token budget cap
  - [ ] Track adapter norm changes
  - [ ] Save to `models/lora_per_user/<user_id>/`

#### D.2.3 Per-User Evaluation
- [ ] Evaluate each user's LoRA on their test split
- [ ] Save `results/lora_per_user/<user_id>/eval.json`

#### D.2.4 Aggregation
- [ ] Create `results/lora_per_user/per_user_summary.csv`
- [ ] Include adapter change magnitude metrics

### D.3 Stability Analysis
- [ ] Implement stability checks `scripts/analyze_stability.py`:
  - [ ] Plot adapter/prefix change magnitudes over online updates
  - [ ] Identify users with high drift vs low drift
  - [ ] Correlate drift with performance improvement
- [ ] Save plots to `results/figures/stability/`

### D.4 Cluster-Level Adaptation (Optional Enhancement)
- [ ] If per-user is unstable:
  - [ ] Train cluster-level adapters (one per cluster)
  - [ ] Then apply tiny user deltas on top
  - [ ] Re-evaluate and compare

### D.5 Comparative Analysis
- [ ] Compare all arms: baseline, all-persona (prefix/LoRA), per-persona (prefix/LoRA)
- [ ] Generate comprehensive plots:
  - [ ] Per-user lift violin plots for each arm
  - [ ] Who benefits vs who regresses (sign test)
  - [ ] Cold-start vs warm-start curves (if applicable)
- [ ] Save to `results/figures/per_persona_comparison/`

**Phase D Deliverables:**
- `models/prefix_per_user/<user_id>/` (50 personalized prefixes)
- `models/lora_per_user/<user_id>/` (50 personalized LoRAs)
- `results/prefix_per_user/per_user_summary.csv`
- `results/lora_per_user/per_user_summary.csv`
- `results/figures/stability/`
- `results/figures/per_persona_comparison/`

---

## Phase E — Persona Prediction from Query

**Goal:** Assess whether the system can identify which user a new query resembles.

### E.1 Profile Representation
- [ ] Implement profile builder `scripts/build_user_profiles.py`:
  - [ ] For each user, aggregate all train messages
  - [ ] Compute text centroid using embeddings (sentence-transformers)
  - [ ] Save profiles as `data/profiles/user_profiles.json`:
    ```json
    {
      "user_<id>": [embedding_vector]
    }
    ```

### E.2 Query Encoding
- [ ] Extract test queries:
  - [ ] Use first user message from each test dialogue
  - [ ] Encode with same embedding model
  - [ ] Create `data/persona_pred/test_queries.jsonl`:
    ```json
    {"query_id": "...", "text": "...", "true_user_id": "..."}
    ```

### E.3 Prediction Pipeline
- [ ] Implement prediction script `scripts/predict_persona.py`:
  - [ ] For each test query:
    - [ ] Encode query
    - [ ] Compute cosine similarity vs all user profiles
    - [ ] Rank users by similarity
    - [ ] Return top-1 prediction
  - [ ] Compute metrics:
    - [ ] Accuracy@1
    - [ ] Accuracy@5
    - [ ] Confusion matrix

### E.4 Adaptive vs Non-Adaptive Comparison
- [ ] Run persona prediction with:
  - [ ] Base embeddings (no adaptation)
  - [ ] Embeddings from adapted models (prefix/LoRA)
- [ ] Compare whether adaptation helps or hurts separability
- [ ] Document findings

### E.5 Results & Visualization
- [ ] Save results:
  - [ ] `results/persona_pred/persona_pred_eval.json`
  - [ ] `results/persona_pred/confusion_matrix.csv`
- [ ] Generate confusion matrix heatmap
- [ ] Identify hardest-to-separate user pairs
- [ ] Save figures to `results/figures/persona_pred/`

**Phase E Deliverables:**
- `data/profiles/user_profiles.json`
- `data/persona_pred/test_queries.jsonl`
- `scripts/predict_persona.py`
- `results/persona_pred/persona_pred_eval.json`
- `results/persona_pred/confusion_matrix.csv`
- `results/figures/persona_pred/`

---

## Phase F — Analysis & Reporting

**Goal:** Synthesize all results, perform statistical tests, and create deliverables.

### F.1 Statistical Analysis
- [ ] Implement stats script `scripts/statistical_analysis.py`:
  - [ ] Paired bootstrap tests (macro lift vs baseline)
  - [ ] Paired t-tests across users
  - [ ] Sign test (# users helped vs hurt per arm)
  - [ ] Confidence intervals (95% CI)
- [ ] Save stats tables to `results/stats/`

### F.2 Ablation Studies
- [ ] LoRA target modules ablation:
  - [ ] Train LoRA with attention-only vs broader targets
  - [ ] Compare performance
- [ ] Prefix length ablation:
  - [ ] Try prefix lengths: 10, 20, 50, 100
  - [ ] Plot performance vs prefix length
- [ ] Online-update budget ablation:
  - [ ] Try 4, 8, 12, 20 examples
  - [ ] Plot performance vs budget
- [ ] Save ablation results to `results/ablations/`

### F.3 Visualization Suite
- [ ] Generate all final plots `scripts/create_all_plots.py`:
  - [ ] Per-user lift violin/box plots
  - [ ] Cold-start vs warm-start curves
  - [ ] Adapter/prefix change magnitude plots
  - [ ] Persona prediction confusion matrix
  - [ ] Before/after qualitative examples (formatted nicely)
- [ ] Save high-quality figures to `results/figures/final/`

### F.4 Decision Framework
- [ ] Implement decision script `scripts/make_recommendations.py`:
  - [ ] Apply decision rules from design.md:
    - [ ] If all-persona is stable + positive, recommend it
    - [ ] If per-persona is high-variance, recommend cluster-first
    - [ ] If persona prediction improves, note stronger signal
  - [ ] Generate recommendation report
  - [ ] Save as `results/recommendations.md`

### F.5 Final Report
- [ ] Create comprehensive report `REPORT.md`:
  - [ ] **Introduction:** Personalization as user-fit, not role-play
  - [ ] **Methods:** Dataset, splits, arms, evaluation
  - [ ] **Results:**
    - [ ] Macro gains table (all arms vs baseline)
    - [ ] Per-user performance distribution
    - [ ] Stability analysis
    - [ ] Cold-start vs warm-start findings
  - [ ] **Persona Prediction:** Impact of adaptation on separability
  - [ ] **Discussion:** Trade-offs, deployment recommendations
  - [ ] **Limitations & Future Work:** Meta-learning, federated updates
- [ ] Include all key tables and figures
- [ ] Add appendix with configs and hyperparameters

### F.6 Reproducibility Package
- [ ] Create `REPRODUCIBILITY.md` with:
  - [ ] Environment setup instructions
  - [ ] Random seeds used
  - [ ] Dataset hash/version
  - [ ] Command sequence to reproduce all phases
  - [ ] Expected runtime estimates
- [ ] Document all configs in `configs/` directory
- [ ] Create experiment registry table:
  ```csv
  arm,split,date,commit,macro_rouge_l,macro_verbosity_error
  baseline,test,2025-11-05,abc123,0.42,15.3
  prefix_all,test,2025-11-06,def456,0.45,14.1
  ...
  ```

**Phase F Deliverables:**
- `results/stats/` (statistical test results)
- `results/ablations/` (ablation study results)
- `results/figures/final/` (all publication-ready figures)
- `results/recommendations.md`
- `REPORT.md` (final comprehensive report)
- `REPRODUCIBILITY.md`
- `experiment_registry.csv`

---

## Implementation Timeline Estimate

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| A | Data Readiness | 3-5 days |
| B | Baseline Establishment | 2-3 days |
| C | Global Adaptation | 5-7 days |
| D | Personalized Adaptation | 7-10 days |
| E | Persona Prediction | 3-4 days |
| F | Analysis & Reporting | 5-7 days |
| **Total** | | **25-36 days** |

---

## Key Dependencies & Prerequisites

- **Hardware:** GPU with ≥16GB VRAM (for 7B model training)
- **Libraries:**
  - PyTorch ≥2.0
  - Transformers (HuggingFace)
  - PEFT (for LoRA/Prefix)
  - sentence-transformers (for embeddings)
  - scikit-learn (for clustering, metrics)
  - matplotlib, seaborn (for plots)
  - pandas, numpy
- **Data:** EdgeWisePersona dataset (confirm access/license)
- **Model:** Base LLM (e.g., Llama-3.2-7B-Instruct or Mistral-7B)

---

## Success Checkpoints

After each phase, verify:

- **Phase A:** ✓ All users have valid train/val/test splits, no data leakage
- **Phase B:** ✓ Baseline metrics are reasonable and reproducible
- **Phase C:** ✓ Global adaptation shows measurable difference vs baseline
- **Phase D:** ✓ Per-user results aggregate correctly, stability is tracked
- **Phase E:** ✓ Persona prediction accuracy is above random chance
- **Phase F:** ✓ Statistical tests are significant, visualizations are clear

---

## Next Steps After Plan

1. Set up project structure (directories, configs)
2. Acquire EdgeWisePersona dataset
3. Begin Phase A implementation
4. Track progress in experiment registry
5. Iterate and document findings

---

*This plan is a living document. Update as implementation progresses and new findings emerge.*