import csv
import json
import re
import time
from pathlib import Path, PurePosixPath

TEST_KEYS = ("pngfile_class0_test", "pngfile_class1_test")


def rebase_image_path(image_path, image_root):
    original = str(image_path)
    direct_path = Path(original)
    if direct_path.is_file():
        return direct_path

    normalized = PurePosixPath(original.replace("\\", "/"))
    parts = normalized.parts
    if "pill_data_croped" in parts:
        relative_parts = parts[parts.index("pill_data_croped") + 1:]
    else:
        pill_index = next(
            (index for index, part in enumerate(parts) if re.fullmatch(r"K-[A-Za-z0-9-]+", part)),
            None,
        )
        relative_parts = parts[pill_index:] if pill_index is not None else (normalized.name,)
    return Path(image_root).joinpath(*relative_parts)


def evaluate_test_set(
    predictor,
    class_list_path,
    image_root,
    output_dir,
):
    with Path(class_list_path).open("r", encoding="utf-8") as file:
        class_list = json.load(file)

    entries = []
    for key in TEST_KEYS:
        for image_path in class_list.get(key, []):
            entries.append((key, image_path))

    label_map = predictor.label_map
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    total_count = len(entries)
    missing_count = 0
    failed_count = 0
    evaluated_count = 0
    misclassified_count = 0
    correct_counts = {1: 0, 3: 0, 5: 0}
    total_inference_seconds = 0.0

    columns = [
        "source_split",
        "image_path",
        "true_class_id",
        "true_pill_id",
        "predicted_top1_class_id",
        "predicted_top1_pill_id",
        "predicted_top1_pill_name",
        "predicted_top1_score_percent",
        "predictions_top5_json",
        "is_top1_correct",
        "is_top3_correct",
        "is_top5_correct",
        "status",
        "error",
    ]

    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for split, source_path in entries:
            resolved_path = rebase_image_path(source_path, image_root)
            pill_id = resolved_path.parent.name
            true_class_id = label_map.pill_to_class.get(pill_id)
            base_row = {
                "source_split": split,
                "image_path": str(resolved_path),
                "true_class_id": true_class_id,
                "true_pill_id": pill_id,
                "predictions_top5_json": "[]",
                "is_top1_correct": False,
                "is_top3_correct": False,
                "is_top5_correct": False,
                "status": "ok",
                "error": "",
            }

            if not resolved_path.is_file():
                missing_count += 1
                base_row.update(status="missing_file", error="image file not found")
                writer.writerow(base_row)
                continue
            if true_class_id is None:
                failed_count += 1
                base_row.update(status="missing_label", error="pill ID is absent from label map")
                writer.writerow(base_row)
                continue

            try:
                if predictor.device.type == "cuda":
                    import torch

                    torch.cuda.synchronize(predictor.device)
                started = time.perf_counter()
                result = predictor.predict(resolved_path, top_k=5)
                if predictor.device.type == "cuda":
                    import torch

                    torch.cuda.synchronize(predictor.device)
                elapsed = time.perf_counter() - started
            except (OSError, ValueError) as exc:
                failed_count += 1
                base_row.update(status="invalid_image", error=str(exc))
                writer.writerow(base_row)
                continue

            candidates = result["predictions"]
            predicted_ids = [candidate["class_id"] for candidate in candidates]
            top1_correct = bool(predicted_ids and predicted_ids[0] == true_class_id)
            top3_correct = true_class_id in predicted_ids[:3]
            top5_correct = true_class_id in predicted_ids[:5]
            evaluated_count += 1
            misclassified_count += not top1_correct
            correct_counts[1] += top1_correct
            correct_counts[3] += top3_correct
            correct_counts[5] += top5_correct
            total_inference_seconds += elapsed

            top1 = candidates[0] if candidates else {}
            base_row.update(
                predicted_top1_class_id=top1.get("class_id"),
                predicted_top1_pill_id=top1.get("pill_id"),
                predicted_top1_pill_name=top1.get("pill_name"),
                predicted_top1_score_percent=top1.get("score_percent"),
                predictions_top5_json=json.dumps(candidates, ensure_ascii=False),
                is_top1_correct=top1_correct,
                is_top3_correct=top3_correct,
                is_top5_correct=top5_correct,
            )
            writer.writerow(base_row)

    metrics = {
        "model": "resnet152_dataclass01",
        "test_list_keys": list(TEST_KEYS),
        "total_test_images": total_count,
        "evaluated_image_count": evaluated_count,
        "missing_file_count": missing_count,
        "failed_image_count": failed_count,
        "misclassified_image_count": misclassified_count,
        "top1_accuracy_percent": 100 * correct_counts[1] / evaluated_count if evaluated_count else None,
        "top3_accuracy_percent": 100 * correct_counts[3] / evaluated_count if evaluated_count else None,
        "top5_accuracy_percent": 100 * correct_counts[5] / evaluated_count if evaluated_count else None,
        "mean_inference_time_seconds": total_inference_seconds / evaluated_count if evaluated_count else None,
        "softmax_scores_are_medically_validated_probabilities": False,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return metrics