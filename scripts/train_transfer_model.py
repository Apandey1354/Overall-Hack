"""
Robust training script for Transfer Learning Model
"""

import sys
from pathlib import Path
import logging
import traceback
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transfer_learning_model import create_transfer_model, TransferLearningTrainer
from src.models.transfer_data_preparer import TransferDataPreparer
from src.data_processing.driver_embedder import create_driver_embeddings
from src.data_processing.track_dna_extractor import extract_all_tracks_dna
from src.data_processing.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("train")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def _resolve_model_path(base_dir: Path, model_name: str = "transfer_model.pt") -> Path:
    models_dir = (base_dir / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / model_name

def _ensure_dataframe_not_empty(df, name: str) -> bool:
    if df is None:
        logger.error("%s is None", name)
        return False
    if getattr(df, "empty", False):
        logger.error("%s is empty", name)
        return False
    return True

def _infer_dims_from_data(driver_embeddings_df, track_dna_df, target_perf_cols=None):
    driver_dim = 0
    track_dim = 0
    perf_dim = 0

    if _ensure_dataframe_not_empty(driver_embeddings_df, "driver_embeddings_df"):
        driver_dim = driver_embeddings_df.shape[1]
    if _ensure_dataframe_not_empty(track_dna_df, "track_dna_df"):
        track_dim = track_dna_df.shape[1]
    if target_perf_cols:
        perf_dim = len(target_perf_cols)

    return driver_dim, track_dim, perf_dim

def _move_tensors_to_device(tensors, device):
    return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tensors)

def _compute_metrics(model, driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor, batch_size=32):
    model.eval()
    preds = []
    labels = []
    dataset_size = driver_tensor.size(0)
    if dataset_size == 0:
        return {}

    with torch.no_grad():
        for i in range(0, dataset_size, batch_size):
            end_idx = min(i + batch_size, dataset_size)
            batch_driver = driver_tensor[i:end_idx]
            batch_source = source_tensor[i:end_idx]
            batch_target = target_dna_tensor[i:end_idx]
            batch_labels = target_perf_tensor[i:end_idx]

            batch_preds = model(batch_driver, batch_source, batch_target)
            preds.append(batch_preds)
            labels.append(batch_labels)

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    mae = torch.mean(torch.abs(preds - labels), dim=0)
    metrics = {
        'lap_time_mae': mae[0].item(),
        'position_mae': mae[1].item(),
        'speed_mae': mae[2].item(),
        'finish_prob_mae': mae[3].item()
    }
    return metrics

def _cross_validate(driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor,
                    driver_dim, track_dim, perf_dim, device, k_folds=3,
                    epochs=20, batch_size=16):
    dataset_size = driver_tensor.size(0)
    if dataset_size < k_folds or k_folds < 2:
        logger.warning("Not enough samples for cross-validation (size=%d, folds=%d)", dataset_size, k_folds)
        return []

    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(SEED))
    fold_size = dataset_size // k_folds
    fold_metrics = []

    for fold in range(k_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k_folds - 1 else dataset_size
        val_idx = indices[val_start:val_end]
        train_idx = torch.cat([indices[:val_start], indices[val_end:]])

        train_driver = driver_tensor[train_idx]
        train_source = source_tensor[train_idx]
        train_target_dna = target_dna_tensor[train_idx]
        train_target_perf = target_perf_tensor[train_idx]

        val_driver = driver_tensor[val_idx]
        val_source = source_tensor[val_idx]
        val_target_dna = target_dna_tensor[val_idx]
        val_target_perf = target_perf_tensor[val_idx]

        model = create_transfer_model(
            driver_embedding_dim=driver_dim,
            track_dna_dim=track_dim,
            performance_features=source_tensor.shape[1]
        )
        model.to(device)
        trainer = TransferLearningTrainer(model, learning_rate=0.001)

        try:
            for epoch in range(epochs):
                trainer.train_epoch(
                    train_driver, train_source, train_target_dna, train_target_perf,
                    batch_size=batch_size
                )
        except Exception:
            logger.exception("Training failed during cross-validation fold %d", fold + 1)
            continue

        fold_metric = _compute_metrics(model, val_driver, val_source, val_target_dna, val_target_perf, batch_size)
        fold_metrics.append(fold_metric)
        logger.info("Cross-validation fold %d/%d metrics: %s", fold + 1, k_folds, fold_metric)

    return fold_metrics

def train_model():
    base_dir = Path(__file__).parent.parent.resolve()
    logger.info("Starting transfer learning training script")
    try:
        loader = DataLoader(base_path=str(base_dir))
    except Exception:
        logger.exception("Failed to initialize DataLoader")
        return 1

    try:
        logger.info("Loading driver embeddings")
        driver_embeddings_df = create_driver_embeddings(data_loader=loader)
        logger.info("Loading track DNA")
        track_dna_df = extract_all_tracks_dna(loader)
    except Exception:
        logger.exception("Failed to load embeddings/DNA")
        return 1

    if not (_ensure_dataframe_not_empty(driver_embeddings_df, "driver_embeddings_df")
            and _ensure_dataframe_not_empty(track_dna_df, "track_dna_df")):
        logger.error("Required data missing; aborting.")
        return 1

    logger.info("Preparing training pairs")
    preparer = TransferDataPreparer(data_loader=loader)
    try:
        training_pairs_df = preparer.prepare_training_pairs(
            driver_embeddings_df=driver_embeddings_df,
            track_dna_df=track_dna_df,
            min_races_per_driver=2
        )
    except Exception:
        logger.exception("prepare_training_pairs failed")
        return 1

    if not _ensure_dataframe_not_empty(training_pairs_df, "training_pairs_df"):
        logger.error("No training pairs created; aborting.")
        return 1

    logger.info("Converting to tensors")
    try:
        driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor = preparer.prepare_tensors(
            training_pairs_df=training_pairs_df,
            driver_embeddings_df=driver_embeddings_df,
            track_dna_df=track_dna_df
        )
    except Exception:
        logger.exception("prepare_tensors failed")
        return 1

    if not (isinstance(driver_tensor, torch.Tensor) and driver_tensor.numel() > 0):
        logger.error("Driver tensor invalid or empty; aborting.")
        return 1

    dataset_size = driver_tensor.size(0)
    if dataset_size < 2:
        logger.error("Dataset too small for train/validation split (size=%d). Need at least 2 samples.", dataset_size)
        return 1

    logger.info("Inferred shapes: driver=%s, source=%s, target_dna=%s, target_perf=%s",
                getattr(driver_tensor, "shape", None),
                getattr(source_tensor, "shape", None),
                getattr(target_dna_tensor, "shape", None),
                getattr(target_perf_tensor, "shape", None))

    driver_dim, track_dim, perf_dim = _infer_dims_from_data(driver_embeddings_df, track_dna_df,
                                                           target_perf_cols=(None if target_perf_tensor is None else list(range(target_perf_tensor.shape[1]))))

    if driver_dim and driver_dim != driver_tensor.shape[1]:
        logger.warning("Driver embedding dimension inferred from df (%d) does not match tensor dim (%d). Using tensor dim.",
                       driver_dim, driver_tensor.shape[1])
        driver_dim = driver_tensor.shape[1]

    if track_dim and track_dim != target_dna_tensor.shape[1]:
        logger.warning("Track DNA dimension inferred from df (%d) does not match tensor dim (%d). Using tensor dim.",
                       track_dim, target_dna_tensor.shape[1])
        track_dim = target_dna_tensor.shape[1]

    if perf_dim == 0 and isinstance(target_perf_tensor, torch.Tensor):
        perf_dim = target_perf_tensor.shape[1] if target_perf_tensor.dim() > 1 else 1

    logger.info("Building model with dims: driver=%d, track_dna=%d, perf=%d", driver_dim, track_dim, perf_dim)

    try:
        model = create_transfer_model(
            driver_embedding_dim=driver_dim,
            track_dna_dim=track_dim,
            performance_features=source_tensor.shape[1]
        )
    except Exception:
        logger.exception("Failed to create model - check create_transfer_model signature and dims")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    try:
        model.to(device)
    except Exception:
        logger.exception("Failed to move model to device")
        return 1

    trainer = TransferLearningTrainer(model, learning_rate=0.001)
    logger.info("Model created; params: %s", sum(p.numel() for p in model.parameters()))

    # Move tensors to device
    try:
        driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor = _move_tensors_to_device(
            (driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor), device)
    except Exception:
        logger.exception("Failed to move tensors to device")
        return 1

    # train/val split
    logger.info("Creating train/validation split")
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(SEED))
    train_size = int(0.8 * dataset_size)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_driver = driver_tensor[train_idx]
    train_source = source_tensor[train_idx]
    train_target_dna = target_dna_tensor[train_idx]
    train_target_perf = target_perf_tensor[train_idx]

    val_driver = driver_tensor[val_idx]
    val_source = source_tensor[val_idx]
    val_target_dna = target_dna_tensor[val_idx]
    val_target_perf = target_perf_tensor[val_idx]

    logger.info("Train samples: %d, Val samples: %d", len(train_driver), len(val_driver))

    num_epochs = 50
    batch_size = 16

    logger.info("Beginning training: epochs=%d, batch_size=%d", num_epochs, batch_size)
    try:
        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(
                train_driver, train_source, train_target_dna, train_target_perf,
                batch_size=batch_size
            )
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_loss = trainer.validate(
                    val_driver, val_source, val_target_dna, val_target_perf,
                    batch_size=batch_size
                )
                logger.info("Epoch %3d/%d  Train Loss: %.6f  Val Loss: %.6f", epoch + 1, num_epochs, train_loss, val_loss)
    except Exception:
        logger.exception("Training failed")
        return 1

    val_metrics = _compute_metrics(model, val_driver, val_source, val_target_dna, val_target_perf, batch_size)
    logger.info("Validation metrics: %s", val_metrics)

    # Cross-validation (optional)
    cross_val_metrics = _cross_validate(
        driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor,
        driver_dim, track_dim, perf_dim, device,
        k_folds=3, epochs=20, batch_size=batch_size
    )

    # Save summary report
    report_path = base_dir / "transfer_learning_training_report.txt"
    try:
        with report_path.open("w", encoding="utf-8") as report_file:
            report_file.write("Transfer Learning Training Report\n")
            report_file.write("=" * 80 + "\n\n")
            report_file.write(f"Total samples: {dataset_size}\n")
            report_file.write(f"Train samples: {len(train_driver)}\n")
            report_file.write(f"Validation samples: {len(val_driver)}\n\n")
            if trainer.train_losses:
                report_file.write(f"Final Train Loss: {trainer.train_losses[-1]:.6f}\n")
            if trainer.val_losses:
                report_file.write(f"Final Val Loss: {trainer.val_losses[-1]:.6f}\n")
            report_file.write(f"Validation metrics: {val_metrics}\n\n")
            if cross_val_metrics:
                report_file.write("Cross-validation metrics per fold:\n")
                for idx, metrics in enumerate(cross_val_metrics, start=1):
                    report_file.write(f"  Fold {idx}: {metrics}\n")
                # Aggregate metrics
                avg_metrics = {}
                for key in cross_val_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in cross_val_metrics])
                report_file.write(f"\nAverage cross-validation metrics: {avg_metrics}\n")
            else:
                report_file.write("Cross-validation not performed or no metrics available.\n")
    except Exception:
        logger.exception("Failed to write training report")

    # Save model
    model_path = _resolve_model_path(base_dir, "transfer_model.pt")
    try:
        if hasattr(trainer, "save_model"):
            trainer.save_model(str(model_path))
        else:
            torch.save(model.state_dict(), str(model_path))
        logger.info("Model saved to %s", model_path)
    except Exception:
        logger.exception("Failed to save model")
        return 1

    logger.info("Training complete")
    return 0

if __name__ == "__main__":
    exit_code = train_model()
    if exit_code != 0:
        logger.error("Script exited with code %d", exit_code)
    raise SystemExit(exit_code)
