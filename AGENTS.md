# MouseLFP cross-day meta-learning

Read README.md before changing the training pipeline.

- Work on explicit, finite training and evaluation jobs. Building code or documentation does not authorize long training runs.
- Preserve private recordings, source annotations, historical experiment artifacts, and unrelated user changes.
- Keep train Days 0-7, validation Days 8/11, and test Days 12/13. Modeled mice are X1/X2/X4/X9/X10/X14/X15.
- Construct each meta-training episode within one mouse: all support days precede all query days, and both sets come exclusively from Days 0-7. Split entire days so overlapping windows stay together.
- Every outer step includes each configured mouse exactly once (all seven in the full configuration). Average their gradients and update once after all tasks; never subsample mice or draw them with replacement. Apply the same schedule to the pooled control.
- Fit episode normalization on support data only. Fit final personalization statistics on that mouse's Days 0-7 and reuse them for validation and test.
- The only meta-parameters are the shared preprocessing frontend. Train a task-local CNN from fixed unlearned starting weights; never meta-learn CNN initialization or carry adapted classifiers between tasks. Preserve the frontend-to-inner-training-to-query gradient path.
- Train the inner CNN for complete shuffled epochs, preserving weights and all optimizer moments within a task. Default to Adam and classifier dropout 0; apply the same dropout configuration to the pooled control and final per-mouse fitting. Preserve Adam bias-correction step counts across replay segments and keep zero-variance hypergradients finite.
- Use support-only loss stability with minimum/maximum epochs; distinguish plateau stopping from budget exhaustion. Worsening loss does not count as stability. Log support accuracy as well as loss: a chance-level loss plateau does not demonstrate effective fitting. Default episodes use all eligible support/query windows. Query/official validation data must never decide the inner stopping epoch.
- Use RNG-preserving activation checkpointing for long inner trajectories; do not silently truncate their hypergradients. Final per-mouse CNN fitting uses the same epoch solver and stopping policy.
- Final per-mouse fitting freezes the shared frontend and trains a fresh CNN from a reproducible unlearned reference. The pooled control may train an auxiliary CNN to learn its representation, but must discard that CNN before final per-mouse fitting.
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
