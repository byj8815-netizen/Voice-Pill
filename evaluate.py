import argparse
import json
from pathlib import Path

from src.evaluation import evaluate_test_set
from src.inference import PillPredictor, PROJECT_ROOT


def _default_path(project_path, adjacent_path):
    return project_path if project_path.exists() else adjacent_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate the existing Test split only.")
    parser.add_argument(
        "--class-list",
        default=str(_default_path(
            PROJECT_ROOT / "data" / "pill_class_list.json",
            PROJECT_ROOT.parent / "pill_class_list.json",
        )),
    )
    parser.add_argument(
        "--image-root",
        default=str(_default_path(PROJECT_ROOT / "data" / "test_images", PROJECT_ROOT.parent)),
    )
    parser.add_argument("--metadata-root")
    parser.add_argument("--label-map")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    predictor = PillPredictor(
        checkpoint_path=args.checkpoint,
        label_path=args.label_map,
        metadata_root=args.metadata_root,
        device=args.device,
    )
    metrics = evaluate_test_set(
        predictor,
        class_list_path=args.class_list,
        image_root=args.image_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()