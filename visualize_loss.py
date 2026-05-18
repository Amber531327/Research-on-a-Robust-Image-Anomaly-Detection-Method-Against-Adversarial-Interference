import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LOSS_LINE_PATTERN = re.compile(r"Epoch\s+(\d+)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _default_log_path(dataset, class_name, split="train"):
    if split == "val":
        return Path(f"./{dataset}_{class_name}_val_loss_log")
    return Path(f"./{dataset}_{class_name}_loss_log")


def _default_output_path(dataset, class_name, output_dir="./vis_output/loss_curve"):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root / f"{dataset}_{class_name}_loss_curve.png"


def parse_loss_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Loss log file not found: {log_path}")

    epochs = []
    losses = []
    skipped_lines = 0

    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = LOSS_LINE_PATTERN.fullmatch(line)
            if match is None:
                skipped_lines += 1
                print(f"[Loss Curve] Skipping malformed line {line_number} in {log_path}: {line}")
                continue

            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))

    if not epochs:
        raise ValueError(f"No valid loss records were found in {log_path}")

    if skipped_lines > 0:
        print(f"[Loss Curve] Parsed {len(epochs)} records from {log_path}, skipped {skipped_lines} malformed lines.")

    return epochs, losses


def _extract_latest_training_segment(epochs, losses):
    # Logs can contain multiple runs/classes appended together. We keep the latest
    # monotonic epoch segment so the curve reflects one training run only.
    segments = []
    current_epochs = []
    current_losses = []

    for epoch, loss in zip(epochs, losses):
        if current_epochs and epoch <= current_epochs[-1]:
            segments.append((current_epochs, current_losses))
            current_epochs = []
            current_losses = []
        current_epochs.append(epoch)
        current_losses.append(loss)

    if current_epochs:
        segments.append((current_epochs, current_losses))

    if not segments:
        return epochs, losses

    latest_epochs, latest_losses = segments[-1]
    if len(segments) > 1:
        print(
            f"[Loss Curve] Detected {len(segments)} training segments in one log; "
            f"using latest segment with {len(latest_epochs)} points "
            f"(epoch {latest_epochs[0]} to {latest_epochs[-1]})."
        )
    return latest_epochs, latest_losses


def _load_curve_from_log(log_path):
    epochs, losses = parse_loss_log(log_path)
    return _extract_latest_training_segment(epochs, losses)


def plot_loss_curve(log_path, dataset, class_name, output_path=None, output_dir="./vis_output/loss_curve", val_log_path=None):
    log_path = Path(log_path)
    output_path = Path(output_path) if output_path is not None else _default_output_path(dataset, class_name, output_dir)

    train_epochs, train_losses = _load_curve_from_log(log_path)
    val_epochs = None
    val_losses = None

    if val_log_path is not None:
        val_log_path = Path(val_log_path)
        if val_log_path.exists():
            val_epochs, val_losses = _load_curve_from_log(val_log_path)
        else:
            print(f"[Loss Curve] Validation log not found, plotting train only: {val_log_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
    if val_epochs is not None and val_losses is not None:
        plt.plot(val_epochs, val_losses, label="Validation Loss", color="#d62728", linewidth=2)
    plt.title(f"Training Loss Curve - {dataset}/{class_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Loss Curve] Saved loss curve to {output_path}")
    return output_path


def _resolve_log_path(dataset, class_name, split="train"):
    default_log_path = _default_log_path(dataset, class_name, split=split)
    if default_log_path.exists():
        return default_log_path

    if split == "val":
        return None

    legacy_candidates = sorted(Path(".").glob(f"{dataset}_*_loss_log"))
    scored_candidates = []
    for candidate in legacy_candidates:
        candidate_name = candidate.name
        if class_name not in candidate_name:
            continue

        # Parse possible legacy list style: dataset_['a', 'b']_loss_log
        score = 1
        payload = candidate_name[len(f"{dataset}_") : -len("_loss_log")]
        try:
            parsed = ast.literal_eval(payload)
            if isinstance(parsed, str):
                parsed = [parsed]
            if isinstance(parsed, list) and class_name in parsed:
                score = 10
                if parsed and parsed[-1] == class_name:
                    score = 100
        except (ValueError, SyntaxError):
            pass

        scored_candidates.append((score, candidate.stat().st_mtime, candidate))

    if scored_candidates:
        scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = scored_candidates[0][2]
        print(f"[Loss Curve] Using legacy loss log for class {class_name}: {chosen}")
        return chosen

    raise FileNotFoundError(
        f"Loss log file not found for class '{class_name}'. Expected {default_log_path}"
    )


def visualize_loss_curves(args):
    class_names = args.class_name if isinstance(args.class_name, list) else [args.class_name]
    output_paths = []

    for class_name in class_names:
        train_log_path = _resolve_log_path(args.dataset, class_name, split="train")
        val_log_path = _resolve_log_path(args.dataset, class_name, split="val")
        output_paths.append(
            plot_loss_curve(train_log_path, args.dataset, class_name, val_log_path=val_log_path)
        )

    return output_paths
