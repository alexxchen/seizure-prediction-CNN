"""Evaluate the selected adapted models on validation or explicitly requested test days."""

from meta_learning.common import prediction_metrics, resolve_device, seed_everything, validate_sample_table


if __name__ == "__main__":
    from meta_learning.cli import evaluate_main
    evaluate_main()
