# Current MouseLFP architecture

`architecture.tex` is an editable, three-page LaTeX/TikZ diagram of the current
working-tree implementation on `meta-learning/neural-stft-frontend`, captured on
2026-10-16. `architecture.pdf` is the compiled document.

- Page 1: fixed waveform preprocessing, shared convolution + MLP residual
  frontend, and the complete task-local CNN with tensor dimensions.
- Page 2: support/query/test roles, the differentiable inner fit, the outer
  frontend update, support-only refitting, and query-based checkpoint selection.
  Separate boxes describe the standalone log-STFT and pooled controls.
- Page 3: the inner optimizer unrolled as a recurrent computation graph. Each
  transition consumes one support minibatch; the state contains CNN parameters,
  momentum, and BatchNorm running statistics. Dashed arrows show query loss
  propagating through every update and through the query frontend directly.

The current `configs/meta.json` enables **X2 only**, so its outer objective has
one task per step. The full seven-mouse cohort is listed separately. The diagrams
show the current 50-epoch budget, SGD settings, zero classifier dropout, and
fixed inner shuffle seed; update those labels when changing the configuration.

## Build

Run from this directory with a TeX installation providing TikZ/PGF, AMS math,
and Latin Modern fonts:

```bash
make
```

Alternatively, use Tectonic:

```bash
tectonic architecture.tex
```

The figure sources are `figures/network.tex`, `figures/training.tex`, and
`figures/unrolled.tex`.
They share colors, node styles, and macros defined by `architecture.tex`.
All diagrams are vector graphics; no external images or training jobs are needed.

## Source mapping

| Diagram component | Implementation |
| --- | --- |
| Window duration | `../sample_pipeline.py`: `SAMPLE_CONFIG` |
| Fixed STFT, log map, residual frontend | `../meta_learning/frontend.py`: `fixed_spectrum_batch`, `LogSTFTFrontend`, `SpectralFrontend` |
| Native CNN input shape | `../meta_learning/model.py`: `CNN_INPUT_SHAPE` |
| CNN layers and axis BatchNorm | `../models/cnn.py`: `SeizureCNN`, `AxisBatchNorm` |
| Meta and pooled objectives | `../meta_learning/algorithms.py` |
| Full-trajectory inner solver | `../meta_learning/inner.py` |
| Outer schedule and checkpoint selection | `../meta_learning/engine.py` |
| Fresh support fitting and inference | `../meta_learning/adaptation.py` |
| Current experiment settings | `../configs/meta.json` |

Shapes include batch dimension `B`. Frontend Conv2d tensors use channel/time/
frequency axes; the CNN receives channel/electrode/time/frequency axes.
`CE_0.1` denotes the training query cross-entropy with label smoothing 0.1.
Checkpoint selection instead uses unsmoothed predictive log loss after refitting.

In the unrolled view, repeated frontend nodes share the same parameters, which
remain fixed throughout one inner fit. Support features are computed once and
indexed by minibatch in the implementation; the repeated nodes display their
logical dependencies. The optimizer transition is fixed by the configured
training rule. Its support loss depends on the CNN and frontend, so query
backpropagation differentiates through the support gradients, momentum, clipping,
and BatchNorm EMA updates. Only the shared frontend receives the outer update.
