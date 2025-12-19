import numpy as np
from typing import Dict, Tuple
import random
import math
import os
import shutil
from datetime import datetime


def set_seed(seed=42):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed: The integer value to use as the random seed.
    """
    # Set the seed for PyTorch CPU operations
    torch.manual_seed(seed)
    # Set the seed for PyTorch CUDA operations on all GPUs
    torch.cuda.manual_seed_all(seed)
    # Set the seed for NumPy's random number generator
    np.random.seed(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Configure CuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable the CuDNN benchmark mode, which can be non-deterministic
    torch.backends.cudnn.benchmark = False


def get_experiment_dir(
    experiment_name: str, base_dir: str = "experiments", overwrite: bool = False
) -> str:
    """
    Create and return experiment directory with date-based structure: base_dir/YYYYMMDD/experiment_name

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory path (default: "experiments")
        overwrite: If True, remove existing directory before creating new one (default: False)

    Returns:
        str: Full path to the experiment directory
    """
    date_str = datetime.now().strftime("%Y%m%d")
    exp_dir = os.path.join(base_dir, date_str, experiment_name)
    if os.path.exists(exp_dir) and overwrite:
        print(f"Removing existing directory: {exp_dir}")
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def compute_statistical_features(
    data: np.ndarray, channels_first: bool = True
) -> Dict[str, np.ndarray]:
    """
    Per-channel statistical features (population definitions):
      Mean:      \bar{x} = (1/N) \sum_{t=1}^N x_t
      Min/Max:   extrema over time
      Variance:  \sigma^2 = (1/N) \sum_{t=1}^N (x_t - \bar{x})^2
      Skewness:  \gamma_1 = \mu_3 / (\mu_2)^{3/2},  \mu_k = (1/N) \sum (x_t - \bar{x})^k
      Kurtosis:  \gamma_2 = \mu_4 / (\mu_2)^2        (non-excess)
    If a channel has ~zero variance, skewness/kurtosis are returned as np.nan.
    Input shape defaults to (C, T); set channels_first=False for (T, C).
    """
    x = data.astype(np.float64, copy=False)
    if not channels_first:
        x = x.T  # (C, T)
    C, N = x.shape
    eps = 1e-15

    mean = x.mean(axis=1)  # (C,)
    centered = x - mean[:, None]  # (C, T)
    var = (centered**2).mean(axis=1)  # population variance (ddof=0)

    m3 = (centered**3).mean(axis=1)
    m4 = (centered**4).mean(axis=1)

    var_pos = var > eps
    skewness = np.full(C, np.nan, dtype=np.float64)
    kurtosis = np.full(C, np.nan, dtype=np.float64)
    skewness[var_pos] = m3[var_pos] / np.power(var[var_pos], 1.5)
    kurtosis[var_pos] = m4[var_pos] / np.power(var[var_pos], 2.0)

    vmin = x.min(axis=1)
    vmax = x.max(axis=1)

    return {
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "variance": var,
    }


def pearson_correlation_matrix(
    data: np.ndarray, channels_first: bool = True
) -> np.ndarray:
    """
    Pearson correlation across channels (population form):
        ρ_{ij} = \sum_t (x_i(t)-μ_i)(x_j(t)-μ_j) /
                 sqrt( \sum_t (x_i(t)-μ_i)^2 \sum_t (x_j(t)-μ_j)^2 )
    Returns (C, C) symmetric matrix with ones on the diagonal.
    """
    x = data.astype(np.float64, copy=False)
    if not channels_first:
        x = x.T  # (C, T)
    C, N = x.shape
    eps = 1e-15

    mu = x.mean(axis=1, keepdims=True)  # (C,1)
    xc = x - mu  # (C,T)

    ss = np.sqrt((xc**2).sum(axis=1))  # (C,)
    ss[ss < eps] = np.inf  # avoid divide-by-zero

    numer = xc @ xc.T  # (C,C)
    denom = ss[:, None] * ss[None, :]  # (C,C)
    R = numer / denom
    R = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return R


def _hann_window(M: int) -> np.ndarray:
    """Hann window (periodic form) length M."""
    if M <= 1:
        return np.ones(M, dtype=np.float64)
    n = np.arange(M, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / M)


def coherence_matrix(
    data: np.ndarray,
    fs: float = 1.0,
    coherence_ratio: str = "1/3",
    channels_first: bool = True,
    return_spectrum: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Welch-style magnitude-squared coherence (no SciPy) for all channel pairs:
        C_xy(f) = |S_xy(f)|^2 / (S_xx(f) S_yy(f))
    Steps:
      • Split into overlapping segments of length `seg_length` with `overlap`.
      • Detrend each segment (mean remove), apply Hann window, compute rFFT.
      • Average auto/cross periodograms over segments.
      • Compute C_xy(f), clip to [0,1]. Return mean across positive freqs (excl. DC/Nyq)
        as a single scalar per pair (unless `return_spectrum=True`).

    Returns:
      coh_mat : (C, C) mean coherence in [0,1]
      f       : (F,) frequency vector (Hz)
      coh_spec: (C, C, F) coherence spectra if return_spectrum=True, else None
    """
    x = data.astype(np.float64, copy=False)
    if not channels_first:
        x = x.T  # (C, T)
    C, N = x.shape

    ratio_map = {"1/3": 1 / 3, "1/2": 1 / 2, "2/3": 2 / 3}
    seg_length = int(N * ratio_map[coherence_ratio])
    coherence_overlap = 0.5  # Fixed at 50%

    step = max(1, int(seg_length * (1.0 - coherence_overlap)))
    if seg_length > N:
        seg_length = N
        step = N

    w = _hann_window(seg_length)
    F = seg_length // 2 + 1  # rfft bins

    Sxx = np.zeros((C, F), dtype=np.complex128)
    Sxy = np.zeros((C, C, F), dtype=np.complex128)

    n_segments = 0
    for start in range(0, N - seg_length + 1, step):
        seg = x[:, start : start + seg_length]  # (C, L)
        seg = seg - seg.mean(axis=1, keepdims=True)  # detrend
        segw = seg * w  # window
        X = np.fft.rfft(segw, n=seg_length, axis=1)  # (C, F)

        Sxx += X * np.conj(X)  # auto
        for i in range(C):  # cross
            Sxy[:, i, :] += X * np.conj(X[i, :])
        n_segments += 1

    if n_segments == 0:
        raise ValueError(
            "Not enough samples for a single segment; decrease seg_length."
        )

    Sxx /= n_segments
    Sxy /= n_segments

    eps = 1e-30
    Sxx = np.maximum(Sxx.real, eps)
    denom = Sxx[:, None, :] * Sxx[None, :, :]  # (C,C,F)
    coh_spec = (np.abs(Sxy) ** 2) / denom  # (C,C,F)
    coh_spec = np.clip(coh_spec.real, 0.0, 1.0)

    f = np.fft.rfftfreq(seg_length, d=1.0 / fs)

    # exclude DC & Nyquist when averaging (if present)
    if F >= 3:
        valid = slice(1, -1)
    else:
        valid = slice(None)

    coh_mean = coh_spec[..., valid].mean(axis=-1)
    np.fill_diagonal(coh_mean, 1.0)

    if return_spectrum:
        return coh_mean, f, coh_spec
    return coh_mean, f, None


# ============================================================================
# V1 Loader Functions
# ============================================================================
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


def _group_indices_by_subject(dataset) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Returns:
      subj_to_indices:  {subject_id: [dataset indices for that subject]}
      subj_to_label:    {subject_id: integer label}
    """
    subj_to_indices: Dict[str, List[int]] = defaultdict(list)
    subj_to_label: Dict[str, int] = {}
    for i in range(len(dataset)):
        g = dataset[i]
        sid = str(g.subject_id)
        lbl = int(g.y.item() if isinstance(g.y, torch.Tensor) else g.y)
        subj_to_indices[sid].append(i)
        # assume consistent label per subject:
        if sid in subj_to_label and subj_to_label[sid] != lbl:
            raise ValueError(f"Subject {sid} has inconsistent labels in dataset.")
        subj_to_label[sid] = lbl
    return subj_to_indices, subj_to_label


def _dataset_overview(dataset) -> None:
    # sample graph for shapes/dtypes
    g0 = dataset[0]
    x_shape = tuple(g0.x.shape) if hasattr(g0, "x") else None
    x_dtype = getattr(g0.x, "dtype", None)
    ea_shape = tuple(g0.edge_attr.shape) if hasattr(g0, "edge_attr") else None
    ea_dtype = getattr(g0.edge_attr, "dtype", None)
    num_graphs = len(dataset)

    subj_to_indices, subj_to_label = _group_indices_by_subject(dataset)
    n_subjects = len(subj_to_indices)

    # class balance (per-graph and per-subject):
    graph_labels = []
    for i in range(num_graphs):
        yi = int(
            dataset[i].y.item()
            if isinstance(dataset[i].y, torch.Tensor)
            else dataset[i].y
        )
        graph_labels.append(yi)
    graph_counts = Counter(graph_labels)
    subj_counts = Counter(subj_to_label.values())

    print("=== Dataset Overview ===")
    print(f"Total graphs           : {num_graphs}")
    print(f"Unique subjects        : {n_subjects}")
    if x_shape is not None:
        print(f"Node feature x shape   : {x_shape} | dtype={x_dtype}")
    if ea_shape is not None:
        print(f"Edge_attr shape        : {ea_shape} | dtype={ea_dtype}")
    print(f"Per-graph label counts : {dict(graph_counts)}")
    print(f"Per-subject label cnts : {dict(subj_counts)}")
    print("------------------------")


def _print_split_summary(
    title: str,
    dataset,
    indices: Sequence[int],
    show_subjects: bool = False,
) -> None:
    subj_to_indices, subj_to_label = _group_indices_by_subject(Subset(dataset, indices))
    # Because we passed a Subset, the inner grouping gets reindexed; we only need counts here:
    # To list original subject IDs, collect them from the original dataset using provided indices.
    subjects_in_split = set()
    label_counts_graph = Counter()
    for idx in indices:
        yi = int(
            dataset[idx].y.item()
            if isinstance(dataset[idx].y, torch.Tensor)
            else dataset[idx].y
        )
        label_counts_graph[yi] += 1
        subjects_in_split.add(str(dataset[idx].subject_id))
    # per-subject labels in this split:
    subj_labels_here = {}
    for sid in subjects_in_split:
        # consistent label per subject; find first index of that subject in split:
        # (we can scan dataset indices once)
        for idx in indices:
            if str(dataset[idx].subject_id) == sid:
                subj_labels_here[sid] = int(
                    dataset[idx].y.item()
                    if isinstance(dataset[idx].y, torch.Tensor)
                    else dataset[idx].y
                )
                break

    print(title)
    print(f"  graphs: {len(indices)} | subjects: {len(subjects_in_split)}")
    print(f"  label counts (graphs): {dict(label_counts_graph)}")
    print(f"  label counts (subjects): {dict(Counter(subj_labels_here.values()))}")
    if show_subjects:
        # Sort IDs for stable display
        print(f"  subjects: {sorted(subjects_in_split)}")


def _stratified_subject_split(
    subject_ids: List[str],
    subject_labels: List[int],
    val_ratio: float,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Returns (train_subjects, val_subjects) using stratified splitting over subjects.
    Uses sklearn if available, else a deterministic manual fallback.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=random_state
    )
    idx = np.arange(len(subject_ids))
    ((train_idx, val_idx),) = sss.split(idx, subject_labels)
    return [subject_ids[i] for i in train_idx], [subject_ids[i] for i in val_idx]


def _subject_kfold_indices(
    subject_ids: List[str],
    subject_labels: List[int],
    n_splits: int,
    random_state: int = 42,
):
    """
    Yields (train_subjects, val_subjects) per fold using stratified K-fold over subjects.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx = np.arange(len(subject_ids))
    for fold, (tr, va) in enumerate(skf.split(idx, subject_labels), start=1):
        yield fold, [subject_ids[i] for i in tr], [subject_ids[i] for i in va]


def _subjects_to_indices(dataset, subjects: Sequence[str]) -> List[int]:
    """Collect dataset indices whose .subject belongs to the provided set."""
    targets = set(map(str, subjects))
    out = []
    for i in range(len(dataset)):
        if str(dataset[i].subject_id) in targets:
            out.append(i)
    return out


def get_holdout_subject_loaders(
    dataset,
    *,
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    # holdout options:
    val_subjects: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
    show_subjects: bool = False,
):
    """
    Build subject-level holdout loaders (no data leakage).
    If `val_subjects` is given, it is used verbatim. Otherwise a stratified
    subject split with `val_ratio` is created.

    Returns
    -------
    (train_loader, val_loader)
    """
    _dataset_overview(dataset)

    # Group by subject for stratification
    subj_to_indices, subj_to_label = _group_indices_by_subject(dataset)
    subject_ids = sorted(subj_to_indices.keys())
    subject_labels = [subj_to_label[sid] for sid in subject_ids]

    if val_subjects is None:
        train_subjects, val_subjects = _stratified_subject_split(
            subject_ids, subject_labels, val_ratio, random_state
        )
    else:
        val_subjects = list(map(str, val_subjects))
        train_subjects = [sid for sid in subject_ids if sid not in set(val_subjects)]

    # Map subjects → dataset indices
    train_indices = _subjects_to_indices(dataset, train_subjects)
    val_indices = _subjects_to_indices(dataset, val_subjects)

    # Print summaries (like REFERENCES style)
    print("=== Holdout Split (by subject) ===")
    _print_split_summary("Train", dataset, train_indices, show_subjects=show_subjects)
    _print_split_summary("Val  ", dataset, val_indices, show_subjects=show_subjects)
    print("----------------------------------")

    # Build PyG DataLoaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def get_kfold_subject_loaders(
    dataset,
    *,
    n_splits: int = 5,
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_state: int = 42,
    show_subjects: bool = False,
):
    """
    Build K-fold (stratified) loaders split by subject (no leakage).
    Returns a list of (train_loader, val_loader) pairs for each fold.
    """
    _dataset_overview(dataset)

    subj_to_indices, subj_to_label = _group_indices_by_subject(dataset)
    subject_ids = sorted(subj_to_indices.keys())
    subject_labels = [subj_to_label[sid] for sid in subject_ids]

    fold_loaders: List[Tuple[DataLoader, DataLoader]] = []

    for fold_id, tr_subjects, va_subjects in _subject_kfold_indices(
        subject_ids, subject_labels, n_splits=n_splits, random_state=random_state
    ):
        train_indices = _subjects_to_indices(dataset, tr_subjects)
        val_indices = _subjects_to_indices(dataset, va_subjects)

        print(f"=== K-Fold {fold_id}/{n_splits} (by subject) ===")
        _print_split_summary(
            "Train", dataset, train_indices, show_subjects=show_subjects
        )
        _print_split_summary("Val  ", dataset, val_indices, show_subjects=show_subjects)
        print("-----------------------------------------------")

        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        fold_loaders.append((train_loader, val_loader))

    return fold_loaders


# ============================================================================
# SubsetWithTransform (for use with v2 loaders)
# ============================================================================
from torch_geometric.data import Dataset


class SubsetWithTransform(Dataset):
    """A subset of a dataset with a specific transform applied."""

    def __init__(self, subset: Subset, transform=None):
        """
        Args:
            subset: A Subset object (without transform applied)
            transform: Optional transform to apply to each graph
        """
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        """
        Retrieve a graph from the subset and apply transform if provided.

        Returns:
            graph: The graph object, optionally transformed
        """
        graph = self.subset[idx]
        if self.transform:
            graph = self.transform(graph)
        return graph


# ============================================================================
# V2 Loader Functions (with transform support)
# ============================================================================


def get_holdout_subject_loaders_v2(
    dataset,
    *,
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    # holdout options:
    val_subjects: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
    show_subjects: bool = False,
    # transform options:
    transform=None,
    train_transform=None,
    val_transform=None,
    verbose=False,
):
    """
    Build subject-level holdout loaders with transform support (no data leakage).
    If `val_subjects` is given, it is used verbatim. Otherwise a stratified
    subject split with `val_ratio` is created.

    Supports separate transforms for train and validation sets (recommended for augmentation).

    Args:
        dataset: The full fNIRS graph dataset
        batch_size: Number of graphs per batch
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        val_subjects: Optional list of subject IDs for validation set
        val_ratio: Fraction of subjects for validation (if val_subjects is None)
        random_state: Random seed for reproducibility
        show_subjects: Whether to print subject IDs in split summary
        transform: (DEPRECATED) Transform to apply to all graphs. Use train_transform and val_transform instead.
        train_transform: Transform to apply only to training set (e.g., with augmentation)
        val_transform: Transform to apply only to validation set (e.g., without augmentation)
        verbose: Whether to print dataset overview and split summaries

    Returns:
        (train_loader, val_loader): DataLoaders with transforms applied

    Example:
        # Recommended usage with separate transforms
        train_transform = get_transformations_recommended(
            mean_dict, std_dict, augment=True, feature_mask_p=0.15
        )
        val_transform = get_transformations_recommended(
            mean_dict, std_dict, augment=False
        )
        train_loader, val_loader = get_holdout_subject_loaders_v2(
            dataset,
            train_transform=train_transform,
            val_transform=val_transform
        )
    """
    # Group by subject for stratification
    subj_to_indices, subj_to_label = _group_indices_by_subject(dataset)
    subject_ids = sorted(subj_to_indices.keys())
    subject_labels = [subj_to_label[sid] for sid in subject_ids]

    if val_subjects is None:
        train_subjects, val_subjects = _stratified_subject_split(
            subject_ids, subject_labels, val_ratio, random_state
        )
    else:
        val_subjects = list(map(str, val_subjects))
        train_subjects = [sid for sid in subject_ids if sid not in set(val_subjects)]

    # Map subjects → dataset indices
    train_indices = _subjects_to_indices(dataset, train_subjects)
    val_indices = _subjects_to_indices(dataset, val_subjects)

    # Print summaries
    if verbose:
        _dataset_overview(dataset)
        print("=== Holdout Split (by subject) with Transform ===")
        _print_split_summary(
            "Train", dataset, train_indices, show_subjects=show_subjects
        )
        _print_split_summary("Val  ", dataset, val_indices, show_subjects=show_subjects)
        print("---------------------------------------------------")

    ### START CODE HERE ###

    # Handle backward compatibility: if transform is provided but not train_transform/val_transform
    if transform is not None and train_transform is None and val_transform is None:
        train_transform = transform
        val_transform = transform

    # Create subsets without transform
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Wrap subsets with separate transforms
    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    val_dataset = SubsetWithTransform(val_subset, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    ### END CODE HERE ###

    return train_loader, val_loader


def get_kfold_subject_loaders_v2(
    dataset,
    *,
    n_splits: int = 5,
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_state: int = 42,
    show_subjects: bool = False,
    transform=None,
    train_transform=None,
    val_transform=None,
):
    """
    Build K-fold (stratified) loaders split by subject with transform support (no leakage).
    Returns a list of (train_loader, val_loader) pairs for each fold.

    Supports separate transforms for train and validation sets (recommended for augmentation).

    Args:
        dataset: The full fNIRS graph dataset
        n_splits: Number of folds
        batch_size: Number of graphs per batch
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        random_state: Random seed for reproducibility
        show_subjects: Whether to print subject IDs in split summary
        transform: (DEPRECATED) Transform to apply to all graphs. Use train_transform and val_transform instead.
        train_transform: Transform to apply only to training set (e.g., with augmentation)
        val_transform: Transform to apply only to validation set (e.g., without augmentation)

    Returns:
        fold_loaders: List of (train_loader, val_loader) pairs for each fold

    Example:
        # Recommended usage with separate transforms
        train_transform = get_transformations_recommended(
            mean_dict, std_dict, augment=True, feature_mask_p=0.15
        )
        val_transform = get_transformations_recommended(
            mean_dict, std_dict, augment=False
        )
        fold_loaders = get_kfold_subject_loaders_v2(
            dataset,
            n_splits=5,
            train_transform=train_transform,
            val_transform=val_transform
        )
    """
    _dataset_overview(dataset)

    subj_to_indices, subj_to_label = _group_indices_by_subject(dataset)
    subject_ids = sorted(subj_to_indices.keys())
    subject_labels = [subj_to_label[sid] for sid in subject_ids]

    fold_loaders: List[Tuple[DataLoader, DataLoader]] = []

    for fold_id, tr_subjects, va_subjects in _subject_kfold_indices(
        subject_ids, subject_labels, n_splits=n_splits, random_state=random_state
    ):
        train_indices = _subjects_to_indices(dataset, tr_subjects)
        val_indices = _subjects_to_indices(dataset, va_subjects)

        print(f"=== K-Fold {fold_id}/{n_splits} (by subject) with Transform ===")
        _print_split_summary(
            "Train", dataset, train_indices, show_subjects=show_subjects
        )
        _print_split_summary("Val  ", dataset, val_indices, show_subjects=show_subjects)
        print("-------------------------------------------------------------")

        ### START CODE HERE ###

        # Handle backward compatibility: if transform is provided but not train_transform/val_transform
        # This needs to be done for each fold
        fold_train_transform = train_transform
        fold_val_transform = val_transform
        if transform is not None and train_transform is None and val_transform is None:
            fold_train_transform = transform
            fold_val_transform = transform

        # Create subsets without transform
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Wrap subsets with separate transforms
        train_dataset = SubsetWithTransform(
            train_subset, transform=fold_train_transform
        )
        val_dataset = SubsetWithTransform(val_subset, transform=fold_val_transform)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

        ### END CODE HERE ###

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders


import os
import pickle
import glob
from typing import Dict, List, Any, Optional, Tuple
from prettytable import PrettyTable
import numpy as np


def _load_pickle(file_path: str) -> Dict[str, Any]:
    """
    Load data from a pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        Dictionary containing the pickled data

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file is corrupted or cannot be read
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading pickle file {file_path}: {str(e)}")


def _find_pkl_files(folder_path: str, pattern: str) -> List[str]:
    """
    Find pickle files matching a pattern in a folder.

    Args:
        folder_path: Path to the folder to search
        pattern: Glob pattern to match files (e.g., '*_holdout.pkl')

    Returns:
        List of matching file paths
    """
    search_pattern = os.path.join(folder_path, pattern)
    return sorted(glob.glob(search_pattern))


def _find_best_epoch(metrics_dict: Dict[str, List[float]]) -> int:
    """
    Find the best epoch index based on validation accuracy.

    Args:
        metrics_dict: Dictionary containing training history with 'val_accuracy' key

    Returns:
        Index of the epoch with highest validation accuracy
    """
    val_accuracies = metrics_dict.get("val_accuracy", [])
    if not val_accuracies:
        return 0
    return int(np.argmax(val_accuracies))


def _format_confusion_matrix(cm: np.ndarray, title: str = "CONFUSION MATRIX") -> str:
    """
    Format a confusion matrix as a PrettyTable.

    Args:
        cm: Confusion matrix as numpy array or list
        title: Title for the table

    Returns:
        Formatted string representation of the confusion matrix
    """
    cm = np.array(cm)
    num_classes = cm.shape[0]

    table = PrettyTable()
    table.field_names = [""] + [f"P:{i}" for i in range(num_classes)]

    for i in range(num_classes):
        row = [f"A:{i}"] + [int(cm[i, j]) for j in range(num_classes)]
        table.add_row(row)

    return f"\n{title}\n{table}"


def _format_metrics_table(metrics: Dict[str, float], title: str) -> str:
    """
    Format metrics as a PrettyTable.

    Args:
        metrics: Dictionary of metric names and values
        title: Title for the table

    Returns:
        Formatted string representation of the metrics table
    """
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.align["Metric"] = "l"
    table.align["Value"] = "r"

    for metric_name, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            table.add_row([metric_name, value])
        else:
            table.add_row([metric_name, f"{value:.4f}"])

    return f"\n{title}\n{table}"


def _report_holdout_results(folder_path: str) -> None:
    """
    Generate and print report for holdout training results.

    Args:
        folder_path: Path to experiment folder containing holdout results
    """
    # Try to find holdout pickle files
    holdout_files = _find_pkl_files(folder_path, "*_holdout.pkl")
    holdout_final_files = _find_pkl_files(folder_path, "*_holdout_final.pkl")

    if not holdout_files and not holdout_final_files:
        raise ValueError(f"No holdout result files found in {folder_path}")

    # Load data - prefer full history file for complete reporting
    if holdout_files:
        result_file = holdout_files[0]
        data = _load_pickle(result_file)
        has_history = True
    else:
        result_file = holdout_final_files[0]
        data = _load_pickle(result_file)
        has_history = False

    experiment_name = os.path.basename(folder_path)
    file_name = os.path.basename(result_file)

    # Print header
    print("\n" + "=" * 80)
    print("HOLDOUT TRAINING RESULTS")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Folder: {folder_path}")
    print(f"Results File: {file_name}")

    if has_history:
        # Report training progress
        total_epochs = len(data.get("val_accuracy", []))
        best_epoch = _find_best_epoch(data)

        progress_metrics = {
            "Total Epochs": total_epochs,
            "Best Epoch": best_epoch + 1,  # Convert to 1-indexed
        }
        print(_format_metrics_table(progress_metrics, "TRAINING PROGRESS"))

        # Report best epoch metrics
        best_metrics = {
            "Accuracy": data["val_accuracy"][best_epoch],
            "F1 Score": data["val_f1"][best_epoch],
            "Precision": (
                data["precision"][best_epoch]
                if isinstance(data["precision"], list)
                else data["precision"]
            ),
            "Recall": (
                data["recall"][best_epoch]
                if isinstance(data["recall"], list)
                else data["recall"]
            ),
        }
        print(
            _format_metrics_table(
                best_metrics, f"BEST EPOCH METRICS (Epoch {best_epoch + 1})"
            )
        )

        # Report final epoch metrics
        final_metrics = {
            "Accuracy": data["val_accuracy"][-1],
            "F1 Score": data["val_f1"][-1],
            "Precision": (
                data["precision"][-1]
                if isinstance(data["precision"], list)
                else data["precision"]
            ),
            "Recall": (
                data["recall"][-1]
                if isinstance(data["recall"], list)
                else data["recall"]
            ),
        }
        print(
            _format_metrics_table(
                final_metrics, f"FINAL EPOCH METRICS (Epoch {total_epochs})"
            )
        )

        # Report confusion matrix from final epoch
        cm = data["confusion_matrix"]
        if isinstance(cm, list):
            cm = cm[-1]
        print(_format_confusion_matrix(cm, "CONFUSION MATRIX (Final Epoch)"))
    else:
        # Only have final results
        final_metrics = {
            "Accuracy": data["val_accuracy"],
            "F1 Score": data["val_f1"],
            "Precision": data["precision"],
            "Recall": data["recall"],
        }
        print(_format_metrics_table(final_metrics, "FINAL VALIDATION METRICS"))
        print(_format_confusion_matrix(data["confusion_matrix"], "CONFUSION MATRIX"))

    print()


def _report_kfold_results(folder_path: str) -> None:
    """
    Generate and print report for k-fold cross-validation results.

    Args:
        folder_path: Path to experiment folder containing k-fold results
    """
    # Find all fold files
    fold_files = _find_pkl_files(folder_path, "*_fold_*.pkl")
    overall_files = _find_pkl_files(folder_path, "*_kfold_overall.pkl")

    if not fold_files:
        raise ValueError(f"No fold result files found in {folder_path}")
    if not overall_files:
        raise ValueError(f"No overall k-fold results file found in {folder_path}")

    experiment_name = os.path.basename(folder_path)
    num_folds = len(fold_files)

    # Print header
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Folder: {folder_path}")
    print(f"Number of Folds: {num_folds}")

    # Load fold data
    fold_data = []
    for fold_file in fold_files:
        data = _load_pickle(fold_file)
        fold_num = int(fold_file.split("_fold_")[-1].split(".pkl")[0])
        fold_data.append((fold_num, data))

    # Sort by fold number
    fold_data.sort(key=lambda x: x[0])

    # Create best epoch performance table
    best_table = PrettyTable()
    best_table.field_names = [
        "Fold",
        "Best Epoch",
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
    ]
    best_table.align["Fold"] = "l"
    best_table.align["Best Epoch"] = "r"
    best_table.align["Accuracy"] = "r"
    best_table.align["F1"] = "r"
    best_table.align["Precision"] = "r"
    best_table.align["Recall"] = "r"

    for fold_num, data in fold_data:
        best_epoch = data.get("best_epoch", _find_best_epoch(data))
        best_table.add_row(
            [
                f"Fold {fold_num}",
                best_epoch + 1,
                f"{data['val_accuracy'][best_epoch]:.4f}",
                f"{data['val_f1'][best_epoch]:.4f}",
                f"{data['precision'][best_epoch] if isinstance(data['precision'], list) else data['precision']:.4f}",
                f"{data['recall'][best_epoch] if isinstance(data['recall'], list) else data['recall']:.4f}",
            ]
        )

    print(f"\nPER-FOLD BEST EPOCH PERFORMANCE")
    print(best_table)

    # Create final epoch performance table
    final_table = PrettyTable()
    final_table.field_names = ["Fold", "Accuracy", "F1", "Precision", "Recall"]
    final_table.align["Fold"] = "l"
    final_table.align["Accuracy"] = "r"
    final_table.align["F1"] = "r"
    final_table.align["Precision"] = "r"
    final_table.align["Recall"] = "r"

    for fold_num, data in fold_data:
        final_table.add_row(
            [
                f"Fold {fold_num}",
                f"{data['val_accuracy'][-1]:.4f}",
                f"{data['val_f1'][-1]:.4f}",
                f"{data['precision'][-1] if isinstance(data['precision'], list) else data['precision']:.4f}",
                f"{data['recall'][-1] if isinstance(data['recall'], list) else data['recall']:.4f}",
            ]
        )

    print(f"\nPER-FOLD FINAL EPOCH PERFORMANCE")
    print(final_table)

    # Load and display overall metrics
    overall_data = _load_pickle(overall_files[0])

    # Use the correctly calculated mean metrics from best epochs
    overall_metrics = {
        "Mean Accuracy": overall_data.get("accuracy_mean", 0),
        "Mean Precision": overall_data.get("precision_mean", 0),
        "Mean Recall": overall_data.get("recall_mean", 0),
        "Mean F1": overall_data.get("f1_mean", 0),
        "Overall Accuracy": overall_data.get("accuracy_overall", 0),
        "Overall Precision": overall_data.get("precision_overall", 0),
        "Overall Recall": overall_data.get("recall_overall", 0),
        "Overall F1": overall_data.get("f1_overall", 0),
    }

    print(_format_metrics_table(overall_metrics, "OVERALL CROSS-VALIDATION METRICS"))

    # Display aggregated confusion matrix
    if "confusion_matrix_overall" in overall_data:
        print(
            _format_confusion_matrix(
                overall_data["confusion_matrix_overall"], "AGGREGATED CONFUSION MATRIX"
            )
        )

    print()


def report_training_results(folder_path: str) -> None:
    """
    Generate and print formatted training results report.

    This function automatically detects whether the folder contains holdout
    or k-fold cross-validation results and generates the appropriate report.

    Args:
        folder_path: Path to experiment folder containing pickle files

    Raises:
        ValueError: If folder doesn't exist or contains no valid result files
        FileNotFoundError: If required pickle files are missing

    Examples:
        >>> # Report holdout training results
        >>> report_training_results('/path/to/gine_holdout_candidate_1')

        >>> # Report k-fold training results
        >>> report_training_results('/path/to/gine_kfold_candidate_1')
    """
    # Validate folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Auto-detect training type
    kfold_files = _find_pkl_files(folder_path, "*_kfold_overall.pkl")
    holdout_files = _find_pkl_files(folder_path, "*_holdout.pkl")
    holdout_final_files = _find_pkl_files(folder_path, "*_holdout_final.pkl")

    if kfold_files:
        # K-fold training detected
        _report_kfold_results(folder_path)
    elif holdout_files or holdout_final_files:
        # Holdout training detected
        _report_holdout_results(folder_path)
    else:
        raise ValueError(
            f"No valid training result files found in {folder_path}. "
            "Expected files: *_holdout.pkl, *_holdout_final.pkl, or *_kfold_overall.pkl"
        )
