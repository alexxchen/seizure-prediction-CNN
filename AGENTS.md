# MouseLFP cross-day meta-learning

Read README.md before changing the training pipeline.

- Work on explicit, finite training and evaluation jobs. Building code or documentation does not authorize long training runs.
- Preserve private recordings, source annotations, historical experiment artifacts, and unrelated user changes.
- Keep train Days 0-7, validation Days 8/11, and test Days 12/13. Modeled mice are X1/X2/X4/X9/X10/X14/X15.
- Construct each meta-training episode within one mouse: all support days precede all query days, and both sets come exclusively from Days 0-7. Split entire days so overlapping windows stay together.
- Fit episode normalization on support data only. Fit final personalization statistics on that mouse's Days 0-7 and reuse them for validation and test.
- Inner optimization adapts the CNN; outer optimization learns the shared frontend and CNN initialization through the inner updates. Preserve the higher-order gradient path unless a separately named algorithm explicitly changes it.
- Query batches must not update shared normalization state. The initial implementation fixes BatchNorm running statistics while keeping affine weights trainable.
- Compare against ordinary pooled supervised training with the same architecture, episode examples, normalization, outer-step budget, and personalization budget. Report extra meta-training compute separately.
- Use seven-mouse mean validation accuracy at threshold 0.5 for checkpoint selection. Record sample counts, class ratios, window settings, source fingerprints, configuration, runtime, and selected checkpoints.
- Days 8/11 may select checkpoints but never supply gradient updates or fitted normalization statistics.
- Test evaluation requires an explicit user request and the separate test command. It loads selected adapted checkpoints without training or fitting new statistics.
- Source preparation verifies recording identities and annotation alignment. Describe signal-quality audits by their actual coverage.
- Keep dependencies stable in a comparison. Preserve source snapshots and completed/interrupted run records; resume only compatible checkpoints.
- Validate temporal isolation, hypergradients, checkpoint resumption, and inference behavior with tests. Use a separate short run for real-data smoke checks.
- Use English in code, comments, configurations, logs, and project documentation.

Primary entry points: prepare.py, train.py, evaluate.py, run_meta.sh, and the meta_learning package.
