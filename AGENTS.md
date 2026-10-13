# MouseLFP autoresearch

Read [program.md](program.md) and README before conducting research.

- Keep the private continuous LFP recordings, source annotation file, and day splits unchanged. Train Days 0-7, validation Days 8/11, and test Days 12/13 remain fixed.
- Research focuses on data construction, cleaning, representation, and input layout. Edit `sample_pipeline.py` and `input_pipeline.py`.
- Apply each candidate's sample definitions, window size, cleaning, and feature transformations consistently to training, validation, and test data. Fit normalization statistics on training data only and reuse them for validation and testing. Identical processing does not require fitting each split to its own statistics.
- Event eligibility, positive/negative sampling, exclusion margins, window duration, and stride are editable derived choices. The historical 978-event cohort and 28-second windows are a previous recipe, not immutable source data.
- Preserve the current CNN, its input shape `(1, 1, 55, 114)`, and training controls. `train.py`, `models/cnn.py`, and `models/__init__.py` are fixed during data experiments.
- `prepare.py`, `source_data.py`, `evaluate.py`, `protocol.lock.json`, `SETTINGS_MouseLFP.json`, and supporting legacy data helpers are fixed infrastructure during research.
- Compare seven-mouse mean validation accuracy at threshold 0.5. Record each recipe's sample counts, class ratio, window settings, and evaluation-protocol fingerprint alongside its score. Changes to labeling or exclusions change the evaluated task and must be identified in comparisons.
- `./run_research.sh restore` restores the best sampling and input code after backing up current code. Preserve unrelated user changes and previous experiment records.
- Keep runtime dependencies constant within a comparison session. Explicit environment changes require a new session and matching baseline.
- Source preparation verifies recordings and labels; it does not certify that the private signals are clean. Audit source quality and label alignment before interpreting results.
- Validation results select recipes. Test metrics are computed only when the user ends research and requests `finalize`; its samples and preprocessing come from the selected recipe.
- Documentation changes and project checks leave the experiment loop inactive. Start research when the user requests experiments and stop when asked.

See README for local execution and Docker commands.
