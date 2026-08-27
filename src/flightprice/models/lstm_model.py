"""LSTM over per-flight fare trajectories.

**Do not import xgboost alongside this module.** PyTorch and XGBoost each bundle
their own OpenMP runtime and loading both into one process aborts it on macOS,
presenting as a kernel death with no traceback.

The unit here is a *sequence*: one itinerary (`legId`) observed across successive
search dates, in order. This is a third granularity, distinct from XGBoost's
independent rows and SARIMA's daily route series, and it is the one an LSTM
exists for -- feeding it the same flattened rows a tree sees would discard the
sequential structure that motivates the architecture at all.

The model is many-to-many: at step *t* it emits a prediction for step *t*, having
seen steps *0…t*. That is legitimate because the features are already causal --
`fareLag1`, `zLag1`, the rolling statistics and the calendar terms are all known
before the fare at *t* is observed (see :mod:`flightprice.features.build`). The
recurrent state adds trajectory history to that, which is precisely the extra
information the family is being tested for.

Comparability with XGBoost is preserved deliberately: the same 32 features, the
same folds, the same metrics. Only the model differs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightprice.config import RANDOM_SEED
from flightprice.evaluation.metrics import classification_metrics, regression_metrics
from flightprice.features.build import encode_features
from flightprice.evaluation.splitting import Fold


def select_device():
    """Prefer Apple MPS, then CUDA, then CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str = "legId",
    time_col: str = "searchDate",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split a frame into one ordered array per flight.

    Returns:
        ``(features_per_flight, targets_per_flight)``, ordered by search date
        within each flight.
    """
    ordered = frame.sort_values([group_col, time_col])
    features = encode_features(ordered, list(feature_cols)).to_numpy(dtype="float32")
    targets = ordered[target_col].to_numpy(dtype="float32")

    codes, _ = pd.factorize(ordered[group_col], sort=False)
    boundaries = np.flatnonzero(np.diff(codes)) + 1

    return (
        np.split(features, boundaries),
        np.split(targets, boundaries),
    )


def make_batches(
    sequences: list[np.ndarray],
    targets: list[np.ndarray],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator | None = None,
):
    """Yield padded batches with a validity mask.

    Sequences are sorted by length before batching so that each batch contains
    similar lengths, which keeps padding -- and therefore wasted computation and
    memory -- to a minimum. Only the batch currently in use is ever materialised
    as a dense array.
    """
    import torch

    order = np.argsort([len(s) for s in sequences])
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]

    if shuffle:
        rng = rng or np.random.default_rng(RANDOM_SEED)
        rng.shuffle(batches)

    for idx in batches:
        lengths = [len(sequences[i]) for i in idx]
        longest, n_features = max(lengths), sequences[idx[0]].shape[1]

        x = np.zeros((len(idx), longest, n_features), dtype="float32")
        y = np.zeros((len(idx), longest), dtype="float32")
        mask = np.zeros((len(idx), longest), dtype="float32")

        for row, i in enumerate(idx):
            n = lengths[row]
            x[row, :n] = sequences[i]
            y[row, :n] = targets[i]
            mask[row, :n] = 1.0

        yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def build_model(n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
    """A deliberately small recurrent model.

    One or two layers of 64 units. The architecture is kept modest because the
    comparison is between *families*, not between a tuned network and untuned
    competitors -- XGBoost and SARIMA are also run at sensible defaults, and an
    extensively tuned LSTM set against them would not answer the question the
    dissertation asks.
    """
    import torch
    from torch import nn

    class TrajectoryLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.dropout(out)).squeeze(-1)

    torch.manual_seed(RANDOM_SEED)
    return TrajectoryLSTM()


def train_model(
    model,
    sequences: list[np.ndarray],
    targets: list[np.ndarray],
    task: str,
    device,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    pos_weight: float | None = None,
    verbose: bool = True,
):
    """Train on padded batches, ignoring padded positions in the loss.

    The mask matters: without it the model would be rewarded for predicting
    zeros in the padding, which is most of a short sequence in a batch of longer
    ones.
    """
    import torch
    from torch import nn

    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(RANDOM_SEED)

    if task == "classification":
        weight = torch.tensor([pos_weight], device=device) if pos_weight else None
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=weight)
    else:
        loss_fn = nn.SmoothL1Loss(reduction="none")

    for epoch in range(epochs):
        model.train()
        total, counted = 0.0, 0.0
        for x, y, mask in make_batches(sequences, targets, batch_size, shuffle=True, rng=rng):
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimiser.zero_grad()
            loss_matrix = loss_fn(model(x), y) * mask
            loss = loss_matrix.sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total += float(loss.item()) * float(mask.sum())
            counted += float(mask.sum())

        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}  loss {total / max(counted, 1):.4f}", flush=True)

    return model


def predict(model, sequences, targets, device, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Predict over sequences, returning only the non-padded positions."""
    import torch

    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for x, y, mask in make_batches(sequences, targets, batch_size, shuffle=False):
            out = model(x.to(device)).cpu().numpy()
            valid = mask.numpy().astype(bool)
            predictions.append(out[valid])
            actuals.append(y.numpy()[valid])

    return np.concatenate(predictions), np.concatenate(actuals)


def standardise(
    train_seqs: list[np.ndarray], test_seqs: list[np.ndarray], clip: float = 4.0
):
    """Standardise features on training statistics, then clip to that range.

    Fitting the scaler on all data would leak the test distribution into
    training, so the statistics come from the training window only. Constant
    columns are left alone rather than dividing by zero.

    **The clip is not cosmetic.** Several features index time -- `weekOfYear`,
    `month`, `observationIndex` -- and rolling-origin validation guarantees the
    test window sits *later* than everything the scaler saw. Their standardised
    values therefore land beyond the training range by construction. A tree is
    unaffected, since an unseen-but-larger value simply falls in the terminal
    bin, but a network extrapolates linearly and carries the fare with it:
    measured on short-haul fold 1, the unclipped model predicted a mean fare of
    238.6 dollars against an actual 148.7, while fitting the training window
    almost exactly. Clipping to ±4 standard deviations bounds the extrapolation without
    discarding the features, keeping the feature set identical to the one the
    other families use.
    """
    stacked = np.concatenate(train_seqs, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-8] = 1.0

    scale = lambda seqs: [
        np.clip((s - mean) / std, -clip, clip).astype("float32") for s in seqs
    ]
    return scale(train_seqs), scale(test_seqs)


def evaluate_lstm_folds(
    frame: pd.DataFrame,
    folds: list[Fold],
    feature_cols: list[str],
    target_col: str,
    task: str,
    *,
    epochs: int = 8,
    hidden_size: int = 64,
    num_layers: int = 1,
    batch_size: int = 256,
    threshold: float = 0.5,
    label: str = "lstm",
    verbose: bool = True,
) -> pd.DataFrame:
    """Train and score an LSTM across folds, matching the other families' protocol.

    A fresh model is trained per fold. Features are standardised on training
    statistics only, and NaNs -- which a tree tolerates but a network does not --
    are filled with the training mean via the same statistics.
    """
    import torch

    device = select_device()
    if verbose:
        print(f"device: {device}")

    rows = []
    for fold in folds:
        train_frame = frame[fold.train_mask]
        test_frame = frame[fold.test_mask]

        train_x, train_y = build_sequences(train_frame, feature_cols, target_col)
        test_x, test_y = build_sequences(test_frame, feature_cols, target_col)

        # A network cannot ingest NaN. Fill from training statistics only.
        column_means = np.nanmean(np.concatenate(train_x, axis=0), axis=0)
        column_means = np.nan_to_num(column_means)
        fill = lambda seqs: [np.where(np.isnan(s), column_means, s).astype("float32") for s in seqs]
        train_x, test_x = fill(train_x), fill(test_x)
        train_x, test_x = standardise(train_x, test_x)

        # Standardise the regression target on training statistics. A network
        # initialised near zero, trained under a bounded-gradient loss, converges
        # impossibly slowly towards a target in the hundreds -- it is an artefact
        # of scale, not of the architecture, and leaving it in would handicap the
        # family relative to the trees, which are scale-invariant.
        target_mean, target_std = 0.0, 1.0
        if task == "regression":
            stacked = np.concatenate(train_y)
            target_mean = float(stacked.mean())
            target_std = float(stacked.std()) or 1.0
            train_y = [((t - target_mean) / target_std).astype("float32") for t in train_y]

        pos_weight = None
        if task == "classification":
            positives = float(sum(t.sum() for t in train_y))
            total = float(sum(len(t) for t in train_y))
            pos_weight = (total - positives) / max(positives, 1.0)

        if verbose:
            print(f"  fold {fold.index}: {len(train_x):,} train flights, {len(test_x):,} test flights")

        model = build_model(len(feature_cols), hidden_size, num_layers)
        model = train_model(model, train_x, train_y, task, device, epochs=epochs,
                            batch_size=batch_size, pos_weight=pos_weight, verbose=verbose)

        raw, actual = predict(model, test_x, test_y, device)

        if task == "classification":
            probability = 1.0 / (1.0 + np.exp(-raw))
            metrics = classification_metrics(actual.astype(int), probability, threshold)
        else:
            # Back to dollars before scoring, so the figures are directly
            # comparable with XGBoost and persistence.
            metrics = regression_metrics(actual, raw * target_std + target_mean)

        metrics.update({"model": label, "fold": fold.index,
                        "n_train_flights": len(train_x), "n_test_flights": len(test_x)})
        rows.append(metrics)

        if verbose:
            summary = (f"F1={metrics['f1']:.3f} AUC={metrics['roc_auc']:.3f}"
                       if task == "classification"
                       else f"RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f}")
            print(f"  fold {fold.index}: {summary}\n", flush=True)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    return pd.DataFrame(rows)
