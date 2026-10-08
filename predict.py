import argparse
import json

from src.inference import PillPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict pill candidates from a cropped RGB image.")
    parser.add_argument("image", help="Path to an already cropped 224x224 RGB image")
    parser.add_argument("--checkpoint")
    parser.add_argument("--label-map")
    parser.add_argument("--metadata-root")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictor = PillPredictor(
        checkpoint_path=args.checkpoint,
        label_path=args.label_map,
        metadata_root=args.metadata_root,
        device=args.device,
    )
    print(json.dumps(predictor.predict(args.image, top_k=args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()