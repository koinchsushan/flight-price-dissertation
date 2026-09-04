"""The neural network model, which reads each flight's price history in order.

WARNING BEFORE YOU RUN ANYTHING
Never import xgboost in the same notebook as this file. Both libraries ship
their own copy of a shared component, and loading both crashes the notebook on
a Mac. The symptom is nasty: the kernel simply dies with no error message,
usually nowhere near the line that caused it. This cost several hours to
diagnose. That is why the LSTM has a notebook to itself.

WHAT AN LSTM IS, IN PLAIN TERMS
Long Short-Term Memory. It is a neural network built to read things in order and
carry a running memory as it goes, like reading a sentence one word at a time
while remembering the beginning. Here the "sentence" is one flight's price
history: $312, $312, $340, $340, $389...

WHY IT NEEDS DIFFERENT INPUT TO XGBOOST
XGBoost sees each row on its own, with no idea which rows belong together.
This model sees a whole flight at once, in date order. That memory is the entire
point of the architecture. Feeding it the same disconnected rows a tree gets
would throw away the one thing it can do that a tree cannot -- and then the
comparison would not be testing what we claim it tests.

IS IT ALLOWED TO SEE THE ANSWER?
No, and this is worth being ready to defend. It reads the history up to and
including today, and predicts today. That is legitimate because every feature it
receives was already shifted back in time when it was built (fareLag1 is
yesterday's fare, and so on). The memory adds knowledge of the flight's PAST.
It never sees today's fare.

WHAT IS KEPT IDENTICAL FOR FAIRNESS
Same 32 features, same five rounds, same scoring. Only the model differs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightprice.config import RANDOM_SEED
from flightprice.evaluation.metrics import classification_metrics, regression_metrics
from flightprice.features.build import encode_features
from flightprice.evaluation.splitting import Fold


def select_device():
    """Pick the fastest available hardware to train on.

    MPS is the graphics chip in Apple Silicon Macs, CUDA is an NVIDIA graphics
    card, and CPU is the ordinary processor -- correct but far slower. On the
    machine used for this project it selects MPS.
    """
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
    """Regroup the data from a flat table into one block per flight.

    Before: 800,000 rows in a table, in no meaningful order.
    After:  63,000 separate little blocks, one per flight, each holding that
            flight's prices in date order.

    This is the shape a sequence model needs. The trick used to do the splitting
    is worth knowing: rather than looping over 63,000 flights, we number the
    flights, then find every point where that number changes. Those points are
    the boundaries, and numpy can cut the whole table at all of them at once.

    Returns:
        Two lists -- the features per flight, and the answers per flight.
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
    """Hand the model flights in batches, padded to a common length.

    THE PROBLEM
    The model wants to process many flights at once for speed, but that needs
    them to be rectangular -- all the same length. Real flights are not: one has
    9 prices, another has 30.

    THE FIX, CALLED PADDING
    Stretch every flight in the batch to match the longest one, filling the gaps
    with zeros, and keep a separate "mask" recording which entries are real:

        flight A (3 prices):  312  340  389   0    0     mask: 1 1 1 0 0
        flight B (5 prices):  145  145  160  180  210    mask: 1 1 1 1 1

    The mask is essential. Without it the model would be rewarded for correctly
    predicting the fake zeros, which is most of a short flight's row.

    ONE EFFICIENCY TRICK
    Flights are sorted by length before batching, so each batch holds flights of
    similar length. Otherwise a batch containing one 30-price flight and forty
    9-price flights would be mostly padding -- wasted memory and wasted
    computation.
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
    """Build the network. Deliberately small, and that is a defensible choice.

    One layer of 64 memory units, then a dropout step, then a single output.

    Why so modest? Because the dissertation compares model FAMILIES. XGBoost and
    SARIMA are both run at sensible untuned defaults, so tuning this one heavily
    would be comparing "a network somebody worked hard on" against "two models
    nobody touched". That would answer a different question than the one asked.

    "Dropout" randomly ignores a fraction of the network during training. It
    sounds destructive, and it is: it stops the model leaning too heavily on any
    one path, which helps it cope with data it has not seen before.
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
    """Train the network by showing it the data repeatedly and correcting it.

    Each pass over all the data is called an epoch, and we do eight. Each pass:
    make predictions, measure how wrong they were, nudge the internal settings
    slightly in the direction that would have been less wrong, repeat.

    The mask from make_batches is applied to the error, so the padding zeros are
    ignored. Without it the model would be scored on -- and would learn to
    reproduce -- entries that do not exist.
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

            # The five steps of training, in order:
            optimiser.zero_grad()                    # 1. forget the last correction
            loss_matrix = loss_fn(model(x), y) * mask  # 2. how wrong were we? (x0 on padding)
            loss = loss_matrix.sum() / mask.sum().clamp(min=1.0)  # 3. average over REAL entries only
            loss.backward()                          # 4. work out which way to adjust
            # Cap the size of any single adjustment. Occasionally one batch
            # produces a huge correction that wrecks everything learned so far
            # (known as "exploding gradients"). This keeps training stable.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()                         # 5. apply the adjustment

            total += float(loss.item()) * float(mask.sum())
            counted += float(mask.sum())

        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}  loss {total / max(counted, 1):.4f}", flush=True)

    return model


def predict(model, sequences, targets, device, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions, then throw away the padding before scoring.

    The model produces an answer for every slot including the fake padded ones.
    The mask tells us which are real, and only those are returned -- otherwise
    the scores would be diluted by predictions about entries that do not exist.
    """
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
    """Put every feature on a common scale, and cap extreme values.

    WHY RESCALE AT ALL
    Neural networks struggle when one input is measured in hundreds (fares) and
    another in single digits (day of week). Rescaling puts them all on a
    comparable footing. The scale is measured from the TRAINING data only --
    measuring it on everything would let information about the test period seep
    into training.

    WHY THE CAP MATTERS, AND THIS IS THE INTERESTING BIT
    Some features are essentially counters that only go up: week of the year,
    month, how many times this flight has been priced. And our test period is
    always LATER than our training period. So test values are guaranteed to sit
    beyond anything the model saw while learning -- week 41 when it only ever
    trained up to week 38.

    A decision tree shrugs at this. It asks "is this bigger than 38?", and the
    answer is simply yes. A neural network does not shrug: it draws a straight
    line through what it saw and extends it, and drags its fare prediction along
    for the ride.

    We measured the damage. Without the cap, on the short route in round 1, the
    model predicted an average fare of $238.60 when the real figure was $148.70
    -- while fitting the training period almost perfectly. It was not confused;
    it was confidently extrapolating off the end of the world.

    Capping at 4 standard deviations stops that, without dropping the features
    and without changing the feature set the other models get.
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
    """Run the LSTM through the same five rounds as everything else.

    A brand new model is trained for each round, so no round can benefit from
    having already seen a later round's data.

    Two preparation steps happen here that XGBoost does not need, both because a
    network is fussier than a tree: filling in missing values, and rescaling.
    Both are worked out from the training half only. See standardise() above,
    which explains the one that genuinely changed the results.
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

        # Gaps in the data must be filled before the network sees them: a tree
        # copes with a missing value, a network produces nonsense. We fill with
        # the average of the TRAINING data only -- using the overall average
        # would smuggle information about the test period into training.
        column_means = np.nanmean(np.concatenate(train_x, axis=0), axis=0)
        column_means = np.nan_to_num(column_means)
        fill = lambda seqs: [np.where(np.isnan(s), column_means, s).astype("float32") for s in seqs]
        train_x, test_x = fill(train_x), fill(test_x)
        train_x, test_x = standardise(train_x, test_x)

        # Rescale the thing we are predicting, too, and then undo it afterwards.
        #
        # A fresh network starts out guessing near zero. Asked to reach $400 in
        # small careful steps, it takes an age to get there and looks far worse
        # than it is. That is a units problem, not a modelling one -- trees do
        # not care about scale at all, so leaving it in would unfairly handicap
        # the network in a comparison that is supposed to be about the models.
        #
        # So we train it on rescaled fares and convert the predictions back to
        # pounds and dollars before scoring. All reported numbers are real fares.
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
