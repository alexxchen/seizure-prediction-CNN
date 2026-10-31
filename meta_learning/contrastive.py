"""Task-local supervised contrastive regularization of frontend features."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


TEMPERATURE = 0.1
ANCHOR_CHUNK_SIZE = 256


def frontend_embedding(features):
    """Normalize each complete channel/time output without pooling or projection."""
    return F.normalize(features.flatten(start_dim=1), dim=1, eps=1e-6)


def supervised_contrastive_loss(features, labels, *, chunk_size=ANCHOR_CHUNK_SIZE):
    """Standard SupCon pairing on the supplied views (the meta support pool only).

    Every other same-class view is positive, including crops from the same event
    or background interval. Only the identical view index is excluded. Each valid
    anchor has equal weight; anchors without a same-class partner are excluded and
    counted. Chunk checkpointing bounds pairwise storage
    while preserving gradients through both anchors and all comparison features.
    """
    if len(features) != len(labels) or not len(labels):
        raise ValueError("Contrastive features and labels must align and be nonempty")
    if labels.ndim != 1 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Contrastive labels must be binary")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Contrastive anchor chunk size must be positive")
    class_counts = torch.bincount(labels, minlength=2)
    positive_counts = class_counts[labels] - 1
    anchors = torch.nonzero(positive_counts > 0, as_tuple=True)[0]
    report = dict(samples=len(labels), valid_anchors=len(anchors), skipped_anchors=len(labels) - len(anchors),
                  valid_anchors_by_class={str(label): int((labels[anchors] == label).sum()) for label in (0, 1)},
                  pairing="All other same-class views are positives; exclude self only", temperature=TEMPERATURE,
                  embedding="L2-normalized flattened full frontend output", embedding_dimensions=features[0].numel())
    if not len(anchors):
        return features.sum() * 0, report
    embedding = frontend_embedding(features)
    view_indices = torch.arange(len(labels), device=labels.device)

    def chunk_loss(z, indices):
        different_view = indices[:, None] != view_indices[None, :]
        positive = different_view & (labels[indices, None] == labels[None, :])
        logits = z[indices] @ z.T / TEMPERATURE
        denominator = torch.logsumexp(logits.masked_fill(~different_view, -torch.inf), dim=1)
        positive_mean = logits.masked_fill(~positive, 0).sum(dim=1) / positive_counts[indices]
        return (denominator - positive_mean).sum()

    total = embedding.new_zeros(())
    for indices in anchors.split(chunk_size):
        total = total + (checkpoint(chunk_loss, embedding, indices, use_reentrant=False, preserve_rng_state=True)
                         if torch.is_grad_enabled() and embedding.requires_grad else chunk_loss(embedding, indices))
    return total / len(anchors), report
