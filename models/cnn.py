"""PyTorch implementation of the original patient-specific seizure CNN."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Use --gpus all in Docker or select --device cpu.")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {selected}")
    return selected


class AxisBatchNorm(nn.Module):
    """Match Keras BatchNormalization, including its biased running variance."""

    def __init__(self, features, axis, epsilon=1e-3, momentum=0.99):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))

    def forward(self, inputs):
        shape = [1] * inputs.ndim
        shape[self.axis] = -1
        if self.training:
            axes = tuple(index for index in range(inputs.ndim) if index != self.axis)
            variance, mean = torch.var_mean(inputs, dim=axes, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), 1.0 - self.momentum)
                self.running_var.lerp_(variance.detach(), 1.0 - self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (inputs - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + self.epsilon)
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)


class SeizureCNN(nn.Module):
    """Consume (N, 1, electrodes, time, frequency) and return class logits."""

    def __init__(self, input_shape, nb_classes=2):
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        if len(self.input_shape) != 4 or self.input_shape[0] != 1:
            raise ValueError("Expected input shape (1, electrodes, time, frequency)")
        if any(value < 1 for value in self.input_shape):
            raise ValueError("Input dimensions must be positive")
        electrodes, time_bins, frequency_bins = self.input_shape[1:]
        self.normal1 = AxisBatchNorm(electrodes, axis=2)
        self.conv1 = nn.Conv3d(1, 16, (electrodes, 5, 5), stride=(1, 2, 2))
        # For kernel == stride == 2, ceil_mode matches TensorFlow SAME pooling.
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)
        self.normal2 = AxisBatchNorm(16, axis=1)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.normal3 = AxisBatchNorm(32, axis=1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        def output_bins(size):
            size = (size - 5) // 2 + 1
            size = (size + 1) // 2
            size = (size - 2) // 2
            return (size - 2) // 2

        time_output, frequency_output = output_bins(time_bins), output_bins(frequency_bins)
        if min(time_output, frequency_output) < 1:
            raise ValueError("Time and frequency dimensions are too small for the CNN")
        self.dens1 = nn.Linear(64 * time_output * frequency_output, 128)
        self.dens2 = nn.Linear(128, nb_classes)
        for layer in (self.conv1, self.conv2, self.conv3, self.dens1, self.dens2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        features = self.pool1(torch.relu(self.conv1(self.normal1(inputs))))
        features = features.squeeze(2)
        features = self.pool2(torch.relu(self.conv2(self.normal2(features))))
        features = self.pool3(torch.relu(self.conv3(self.normal3(features))))
        features = self.drop1(torch.flatten(features, start_dim=1))
        features = self.drop2(torch.sigmoid(self.dens1(features)))
        # CrossEntropyLoss consumes logits; softmax is applied during prediction.
        return self.dens2(features)


class SignalDataset(Dataset):
    """Convert cached float64 features to float32 one sample at a time."""

    def __init__(self, inputs, labels=None):
        self.inputs = inputs
        self.labels = labels
        if len(inputs.shape) != 5 or inputs.shape[1] != 1:
            raise ValueError("Expected input array (N, 1, electrodes, time, frequency)")
        if labels is not None and (labels.ndim != 1 or len(labels) != len(inputs)):
            raise ValueError("Inputs and labels must have matching sample counts")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sample = np.array(self.inputs[index], dtype=np.float32, order="C", copy=True)
        if not np.isfinite(sample).all():
            raise ValueError(f"Input sample {index} contains NaN or infinity")
        tensor = torch.from_numpy(sample)
        if self.labels is None:
            return tensor
        label = int(self.labels[index])
        if label not in (0, 1):
            raise ValueError(f"Expected a binary label, received {label}")
        return tensor, torch.tensor(label, dtype=torch.long)


class ConvNN:
    """Train, checkpoint, and evaluate a SeizureCNN on CPU or CUDA."""

    def __init__(self, target, batch_size=16, nb_classes=2, epochs=2, mode="cv",
                 device="auto", checkpoint_dir="results/checkpoints", fold=None,
                 num_workers=0, patience=10, seed=0):
        if batch_size < 1 or epochs < 1 or num_workers < 0 or patience < 0:
            raise ValueError("Invalid batch size, epoch count, worker count, or patience")
        if nb_classes != 2:
            raise ValueError("This experiment uses two classes")
        self.target = str(target)
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.mode = mode
        self.device = resolve_device(device)
        self.num_workers = num_workers
        self.patience = patience
        self.seed = seed
        suffix = f"_fold{fold}" if fold is not None else ""
        self.checkpoint_path = Path(checkpoint_dir) / f"weights_{self.target}_{mode}{suffix}.pt"
        self.history = []

    def setup(self, input_shape):
        self.model = SeizureCNN(input_shape[1:], self.nb_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4,
                                          betas=(0.9, 0.999), eps=1e-8)
        self.criterion = nn.CrossEntropyLoss()
        name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        print(f"Model input: {tuple(input_shape)}, parameters: {parameters}, device: {self.device} ({name})")
        return self

    def _loader(self, inputs, labels=None, shuffle=False):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(SignalDataset(inputs, labels), batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=self.device.type == "cuda", generator=generator)

    def _run_epoch(self, loader, training):
        self.model.train(training)
        total_loss, total_correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; check features and model weights")
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"), error_if_nonfinite=True)
                    self.optimizer.step()
                size = len(inputs)
                total_loss += loss.item() * size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                count += size
        if not count:
            raise ValueError("Training and validation splits must contain samples")
        return total_loss / count, total_correct / count

    def fit(self, X_train, Y_train, X_val=None, y_val=None):
        if (X_val is None) != (y_val is None):
            raise ValueError("Validation inputs and labels must be provided together")
        if X_val is None:
            split = int(len(X_train) * 0.8)
            X_train, X_val = X_train[:split], X_train[split:]
            Y_train, y_val = Y_train[:split], Y_train[split:]
        if not len(X_train) or not len(X_val):
            raise ValueError("Training and validation splits must contain samples")
        train_loader = self._loader(X_train, Y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_score, wait = float("inf"), 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            train_loss, train_accuracy = self._run_epoch(train_loader, training=True)
            val_loss, val_accuracy = self._run_epoch(val_loader, training=False)
            score = train_loss + val_loss
            self.history.append(dict(epoch=epoch, loss=train_loss, accuracy=train_accuracy,
                                     val_loss=val_loss, val_accuracy=val_accuracy))
            print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.6f} - accuracy: {train_accuracy:.4f} "
                  f"- val_loss: {val_loss:.6f} - val_accuracy: {val_accuracy:.4f}", flush=True)
            if score < best_score:
                best_score, wait = score, 0
                checkpoint = dict(model_state_dict=self.model.state_dict(),
                                  optimizer_state_dict=self.optimizer.state_dict(),
                                  input_shape=self.model.input_shape, nb_classes=self.nb_classes,
                                  target=self.target, mode=self.mode, epoch=epoch, score=score)
                temporary = self.checkpoint_path.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(self.checkpoint_path)
            else:
                wait += 1
            # Match the original callback's combined loss and patience rule.
            if score < 1e-3 or wait > self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        self.load_trained_weights(self.checkpoint_path)
        return self

    def load_trained_weights(self, filename):
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        if tuple(checkpoint["input_shape"]) != self.model.input_shape:
            raise ValueError("Checkpoint input shape does not match this model")
        if checkpoint["nb_classes"] != self.nb_classes:
            raise ValueError("Checkpoint class count does not match this model")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {filename}")
        return self

    def predict_proba(self, inputs):
        if not len(inputs):
            return np.empty((0, self.nb_classes), dtype=np.float32)
        self.model.eval()
        predictions = []
        with torch.inference_mode():
            for batch in self._loader(inputs):
                batch = batch.to(self.device, non_blocking=True)
                probabilities = self.model(batch).softmax(dim=1)
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("Non-finite model predictions")
                predictions.append(probabilities.cpu().numpy())
        return np.concatenate(predictions)

    def evaluate(self, inputs, labels):
        predictions = self.predict_proba(inputs)[:, 1]
        auc = float(roc_auc_score(labels, predictions))
        print(f"Test AUC: {auc:.6f}")
        return auc
