---

## Day 0 — 2026-03-26 — Project scaffold, CI hardening, and docstring coverage
> Project: B1-HuggingFace-FastAPI

### What was done

#### Task A — Scaffold & GoEmotions EDA
- Initialised repo with `CLAUDE.md`, project config in `config/config.yaml`.
- Set up full toolchain: `Makefile` targets (`install`, `train`, `test`, `lint`, `serve`, `docker-build`, `gradio`, `audit`).
- Defined `SentimentInput`/`SentimentOutput` Pydantic schemas, pandera `EMOTION_SCHEMA`, `BaseMLModel` ABC.
- Mapped GoEmotions 28 labels → 7 macro-categories (joy, sadness, anger, fear, surprise, disgust, neutral).
- Wired structured JSON logging (`src/logger.py`) and custom exceptions (`PredictionError`, `DataLoadError`).

#### Task B — CI static-analysis fixes
- Added `# nosec B615` inline to `load_dataset` call to silence Bandit false positive.
- Annotated `logging.Formatter` type on formatter variable to satisfy mypy strict mode.
- Pinned `pip-audit` ignore for CVE-2026-4539 (pygments — no upstream fix available).
- Migrated `schemas.py` validators from Pydantic v1 `@validator` to Pydantic v2 `@field_validator`.

#### Task C — Interrogate docstring coverage
- Added module-level docstrings to all 6 empty `__init__.py` files (`src`, `api`, `data`, `evaluation`, `models`, `training`).
- Coverage raised from 78.6% → 100%, clearing `interrogate --fail-under=80` CI gate.

### Why it was done
- Project needed a production-grade scaffold matching the stack before any model training begins.
- Pre-commit hooks (bandit, mypy, flake8, isort, black, detect-secrets) were blocking every commit until static-analysis issues were resolved.
- CI interrogate gate enforces documentation hygiene from Day 0.

### How it was done
- `git mv CLAUDE.md learning.md` preserved rename history cleanly.
- Pydantic v2 migration: replaced `@validator("field", pre=True)` with `@field_validator("field", mode="before")` and changed method signature to `@classmethod`.
- Docstrings: one-liner `"""..."""` at module level in each `__init__.py` — interrogate counts the module node, not just functions/classes.
- `pip-audit` CVE ignore added to `pyproject.toml` / audit config with justification comment.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| `interrogate` | Enforces docstring coverage as a CI gate, configurable threshold | `pydocstyle` | Checks style rules, not coverage percentage |
| `bandit` | SAST for Python — catches security anti-patterns pre-commit | `semgrep` | Heavier setup; overkill for a single-repo project |
| `pandera` | Schema validation with typed DataFrames, integrates with pandas | `great_expectations` | Much heavier; designed for data pipelines, not model input validation |
| Pydantic v2 `@field_validator` | Native v2 API, faster Rust core, better error messages | Pydantic v1 `@validator` | Deprecated; raises warnings in v2, will be removed |
| `python-json-logger` | Emits structured JSON logs — parseable by Datadog/Loki/CloudWatch | Plain `logging` | Free-text logs are hard to query in production observability stacks |

### Definitions (plain English)
- **interrogate**: A tool that counts how many Python functions, classes, and modules have docstrings and fails the build if the percentage is too low.
- **pandera**: A library that lets you declare rules for what a pandas DataFrame must look like (column types, value ranges) and raises an error if the data breaks those rules.
- **`@field_validator` (Pydantic v2)**: A decorator that runs a custom validation function on a specific field whenever a Pydantic model is created from input data.
- **bandit**: A static analysis tool that scans Python source code for common security mistakes (like shell injection or hardcoded passwords) without running the code.
- **macro-category mapping**: Collapsing many fine-grained labels into a smaller set of broader ones — here, 28 GoEmotions labels → 7 human-readable emotions.

### Real-world use case
- `interrogate` is used by the `requests` and `httpx` libraries to enforce documentation standards across contributors.
- Pydantic v2 is the validation layer in FastAPI — every major API company (Uber, Microsoft) using FastAPI relies on it for request/response validation.
- `python-json-logger` + structured logs → Datadog ingestion is the standard pattern at companies like Stripe and Shopify for searchable production logs.

### How to remember it
- **interrogate**: Think of it as a *coverage report for docstrings* — same idea as `pytest --cov`, but for documentation instead of tests.
- **Pydantic v2 migration**: `@validator` → `@field_validator` + add `mode="before"` + add `@classmethod`. The mnemonic: **"field, mode, class"** — three words, three changes.
- **pandera**: "Pandas + bouncer" — it stands at the door and refuses DataFrames that don't meet the dress code.

### Status
- [x] Done
- Next step: Implement `SentimentClassifier` training loop with HuggingFace `Trainer` API (Day 1).

---

## Day 1 — 2026-03-26 — Tokenization, stratified splits, and HuggingFace Dataset objects
> Project: B1-HuggingFace-FastAPI

### What was done
- Implemented `src/data/dataset.py`: `load_goemotions()` loads GoEmotions, filters to single-label rows, maps 27→7 labels, saves SHA-256 checksum to `data/raw/checksums.json`, validates with `EMOTION_SCHEMA`, logs distribution and text-length stats.
- Implemented `src/data/preprocessing.py`: `stratified_split()` uses two `StratifiedShuffleSplit` passes (carve test, then val from remainder), asserts zero overlap, logs balanced class weights, saves CSVs to `data/processed/`.
- Implemented `tokenize_dataset(df, tokenizer, max_len=128)`: converts DataFrame → HuggingFace `Dataset`, maps tokenizer with `padding='max_length'`, sets torch format on `input_ids`, `attention_mask`, `label`.
- Added `tests/test_dataset.py` (6 tests, mocking `load_dataset`) and `tests/test_preprocessing.py` (11 tests, synthetic data + mock tokenizer).
- Added `pyyaml` to `requirements.txt` (was only a transitive dep of `transformers`).

### Why it was done
- The model training loop needs clean, validated, stratified data in HuggingFace `Dataset` + torch tensor format before `Trainer` can be called.
- SHA-256 checksum ensures reproducibility — any change to the raw data is immediately detectable.
- Stratified splits prevent class imbalance from skewing validation metrics.

### How it was done
- **Two-pass split**: `sss1` carves out `test` (10%), `sss2` carves `val / (train + val) = 0.1111` from the remainder. `remainder_df` index is never reset so `set(train_df.index) & set(test_df.index)` gives a reliable zero-overlap assertion.
- **Checksum**: `hashlib.sha256(df.to_csv().encode()).hexdigest()` — deterministic because `pd.concat(..., ignore_index=True)` always produces the same row order from fixed HuggingFace splits.
- **Tokenization**: `Dataset.from_pandas(df, preserve_index=False)` → `.map(_tokenize, batched=True)` → `.set_format("torch", columns=[...])`. `token_type_ids` excluded from set_format since DistilBERT doesn't use them.
- **Mocking in tests**: `patch("src.data.dataset.load_dataset")` returns a `_FakeDataset` class implementing `.filter()` and `__iter__`. `monkeypatch.setattr` redirects `_CHECKSUM_PATH` and `_PROCESSED_DIR` to `tmp_path`.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| `StratifiedShuffleSplit` (×2) | Guarantees class ratio preservation in all three splits | Random split via `train_test_split` | No stratification; rare classes can vanish from val/test |
| `compute_class_weight("balanced")` | sklearn formula: `n_samples / (n_classes * np.bincount(y))` | Manual ratio calculation | Error-prone; `sklearn` version is battle-tested |
| `Dataset.from_pandas()` | Converts DataFrame → HuggingFace Dataset in one call | Building Dataset from dict manually | More verbose; `from_pandas` handles dtypes automatically |
| `dataset.map(batched=True)` | Vectorised: tokenizer processes a list of strings per call | Row-by-row map | 10-100× slower; tokenizers are optimised for batches |
| `set_format("torch")` | Makes Dataset return `torch.Tensor` directly in DataLoader | Manual `torch.tensor()` in `__getitem__` | Boilerplate; `set_format` integrates with `Trainer` API |

### Definitions (plain English)
- **StratifiedShuffleSplit**: Splits data while keeping the same class ratio in every subset — like shuffling a deck and dealing equal suits to each hand.
- **SHA-256 checksum**: A 64-character fingerprint of a file; if even one byte changes, the fingerprint changes completely.
- **`dataset.map(batched=True)`**: Runs a function on many rows at once instead of one-by-one — like processing a tray of parts on an assembly line rather than each part individually.
- **`set_format("torch")`**: Tells the HuggingFace Dataset to return PyTorch tensors instead of Python lists when rows are accessed — no manual conversion needed.
- **Class weight**: A multiplier assigned to each class so rare classes are treated as more important during loss computation; prevents the model from ignoring minority classes.

### Real-world use case
- Two-pass stratified split is the standard pattern at Hugging Face, Google Brain, and every NLP team that fine-tunes on imbalanced datasets (e.g., sentiment, NER, intent classification).
- `dataset.map()` with batched tokenization is used in the official HuggingFace `transformers` examples for GLUE benchmark fine-tuning (BERT, RoBERTa, DistilBERT).
- SHA-256 checksums on processed data are used by DVC (Data Version Control) — the standard MLOps tool at companies like Spotify, Weights & Biases, and Iterative.ai.

### How to remember it
- **Two-pass split mnemonic**: "First cut the test slice, then split the leftover cake into train and val." Always cut from the full loaf, never from a slice.
- **`dataset.map(batched=True)`**: Think "batch = tray" — always send a tray to the tokenizer, not single items.
- **`set_format("torch")`**: It's the Dataset's "language setting" — switch it to PyTorch so tensors come out automatically.

### Status
- [x] Done
- Next step: Implement `SentimentClassifier` training loop with HuggingFace `Trainer` API (Day 2).

---

#### Sub-task B — CI fix: mypy missing stubs

**What:** CI failed on `mypy type check` — `import-untyped` error for `yaml` in both `dataset.py` and `preprocessing.py`.

**Why:** `pyyaml` was added to `requirements.txt` but `types-PyYAML` (the mypy stub package) was missing from `requirements-dev.txt`. The local venv had it as a transitive dep; CI did not.

**Fix:** Added `types-PyYAML` to `requirements-dev.txt`. All 11 CI steps now pass.

**Lesson:** Whenever you add a third-party library, also add its `types-*` stub to dev deps so CI mypy matches local mypy.

---

#### Sub-task C — Remove AI attribution from git history

**What:** `claude Claude` appeared as a GitHub contributor because earlier commits contained `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` trailers.

**Fix:** `FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter 'sed "/noreply@anthropic.com/d"' -- --all` rewrote all 13 commits. Backup refs (`refs/original/`) deleted. Force-pushed to origin.

**Lesson:** `git filter-branch --msg-filter` rewrites commit message text for every commit in history. The backup in `refs/original/` must be manually deleted afterwards or `git log --all` will still show the old commits.

---

#### Sub-task D — Repo rename

**What:** Renamed from `b1-hf-fastapi` → `distilbert-emotion-classifier-fastapi`.

**Files changed:** `pyproject.toml` (`name =`), `Makefile` (docker image tag). GitHub renamed via Settings UI. Local remote updated with `git remote set-url`.

**Lesson:** GitHub repo rename automatically redirects old URLs, but the local `.git/config` remote URL must be updated manually with `git remote set-url origin <new-url>`.

---

#### Sub-task E — Real data pipeline run + verification


**What:** Executed `load_goemotions()` + `stratified_split()` on real GoEmotions data to generate the required artefacts.

**Real data stats (45,446 samples after single-label filter):**
| Class | Count | % | Weight |
|-------|-------|---|--------|
| neutral | 16,021 | 35.3% | 0.41 |
| joy | 8,295 | 18.3% | 0.78 |
| anger | 7,148 | 15.7% | 0.91 |
| surprise | 5,662 | 12.5% | 1.15 |
| sadness | 3,439 | 7.6% | 1.89 |
| fear | 3,236 | 7.1% | 2.01 |
| disgust | 1,645 | 3.6% | 3.95 |

**Text length:** mean 67 chars, std 37, p95 131 — fits comfortably in `max_seq_len=128` tokens.

**Split:** train 36,356 / val 4,545 / test 4,545.

**Final checklist:**
- [x] `data/raw/checksums.json` exists
- [x] `data/processed/train.csv`, `val.csv`, `test.csv` exist
- [x] 27/27 tests pass, 93.5% coverage
- [x] `bandit -r src/ -ll` — No issues identified
- [x] All commits pushed, CI green

---

## Day 2 — 2026-03-27 — SentimentClassifier, WeightedTrainer, and MLflow fine-tuning pipeline
> Project: B1-HuggingFace-FastAPI

### What was done
- Implemented `src/models/model.py`: `SentimentClassifier(BaseMLModel)` with `predict→list[str]`, `predict_proba→list[dict[str,float]]`, `save`, `load`, and `_safe_inputs` guard for empty/NaN strings.
- Implemented `src/training/train.py`: full fine-tuning pipeline — loads config, validates splits with `EMOTION_SCHEMA`, tokenizes with `tokenize_dataset`, builds `WeightedTrainer` with balanced class weights, logs every epoch to MLflow via `MLflowEpochCallback`, saves model + `training_stats.json`.
- Added `tests/test_model.py` (24 tests, fully mocked HuggingFace calls) to restore coverage from 60% → 92%.
- Added `# nosec B615` to all four `from_pretrained()` calls; `bandit -r src/ -ll` returns zero findings.

### Why it was done
- `SentimentClassifier` is the inference boundary: the API layer calls `predict`/`predict_proba` — it must be safe, typed, and independently testable.
- `WeightedTrainer` compensates for class imbalance (neutral 35% vs disgust 3.6%) by scaling CrossEntropyLoss with sklearn balanced weights.
- MLflow tracks every hyperparameter + per-epoch `eval_f1` so experiments are reproducible and comparable.

### How it was done
- **`_safe_inputs`**: rejects empty list, non-str types, whitespace-only, and `math.isnan(float(stripped))` — covers the edge where `"nan"` is a valid float.
- **`WeightedTrainer`**: overrides `compute_loss` — pops `labels` from inputs, runs forward pass, applies `CrossEntropyLoss(weight=class_weights.to(logits.device))`. Moving weights to `logits.device` inside the method ensures GPU/CPU compatibility.
- **`MLflowEpochCallback`**: hooks `on_evaluate` (fires after each epoch's eval pass) and calls `mlflow.log_metric(key, value, step=global_step)` for every scalar metric.
- **Mock strategy in tests**: `patch("src.models.model.AutoTokenizer.from_pretrained")` returns a MagicMock whose `__call__` returns real `torch.zeros` tensors, so `_forward` runs real PyTorch ops (argmax, softmax) on dummy data — no network, no GPU needed.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| `WeightedTrainer(Trainer)` subclass | Minimal override — inherits all Trainer optimisation, scheduling, and checkpointing | Custom training loop | 200+ lines of boilerplate; easy to miss grad clipping, eval, checkpointing |
| `MLflowEpochCallback(TrainerCallback)` | Hooks into Trainer's event system — zero coupling to training logic | `mlflow.autolog()` with Trainer | Logs too many raw metrics; harder to control what's tracked per epoch |
| `torch.nn.CrossEntropyLoss(weight=...)` | Native PyTorch — fused with CUDA kernels, gradient flows automatically | Manual sample reweighting | Sample-level reweighting changes batch statistics; class-level weights are simpler and well-understood |
| `unittest.mock.patch` + real tensors | Mocks HuggingFace I/O but keeps real PyTorch ops — tests actual softmax and argmax logic | `MagicMock` for full model output | Would hide bugs in tensor operations; real tensors validate the full computation path |

### Definitions (plain English)
- **`WeightedTrainer`**: A subclass of HuggingFace `Trainer` that adjusts how much each mistake matters during training — errors on rare classes (like `disgust`) cost more than errors on common ones (like `neutral`).
- **`TrainerCallback`**: A hook object that HuggingFace `Trainer` calls at specific moments (start of epoch, end of eval, etc.) — like event listeners in a UI framework.
- **`mlflow.log_metric(key, value, step)`**: Records a single number at a specific training step so you can plot it as a curve in the MLflow UI.
- **`math.isnan(float(t))`**: A safe way to detect NaN-like strings — `float("nan")` returns Python's NaN float, which `math.isnan` catches.
- **`# nosec B615`**: An inline comment that tells Bandit to ignore a specific security finding on that line — used here because `from_pretrained` is intentional and the model source is controlled by config.

### Real-world use case
- `WeightedTrainer` pattern (override `compute_loss`) is used by Google and Meta in production emotion/intent classifiers where class imbalance is 10-100×.
- MLflow experiment tracking is the standard at Airbnb, Databricks, and Microsoft — every fine-tuning run is an MLflow run so metrics are searchable and reproducible.
- `TrainerCallback` pattern is how the official HuggingFace `WandbCallback` and `TensorBoardCallback` are implemented — the same pattern this project uses for MLflow.

### How to remember it
- **`WeightedTrainer` mnemonic**: "Override `compute_loss` — three lines: pop labels, run forward, apply weighted CE." Those three steps are the entire override.
- **`TrainerCallback` mnemonic**: "`on_evaluate` = after each eval pass, not after each epoch — but with `eval_strategy='epoch'` they're the same."
- **Mock + real tensors**: "Mock the factory (from_pretrained), keep the math (softmax/argmax) real." This is the gold standard for testing ML inference code.

### Status
- [x] Done
- Next step: Build FastAPI `/predict` endpoint + Gradio demo (Day 3).

---
