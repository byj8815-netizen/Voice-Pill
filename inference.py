from pathlib import Path

import torch

from .label_map import PillLabelMap
from .model_loader import MODEL_NAME, load_model
from .preprocessing import load_image_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _existing_path(*candidates):
    return next((path for path in candidates if path.exists()), candidates[0])


def _metadata_root(*candidates):
    for candidate in candidates:
        if candidate.is_dir() and any(
            child.is_dir() and child.name.startswith("K-") for child in candidate.iterdir()
        ):
            return candidate
    return _existing_path(*candidates)


class PillPredictor:
    def __init__(self, checkpoint_path=None, label_path=None, metadata_root=None, device=None):
        data_dir = PROJECT_ROOT / "data"
        self.checkpoint_path = Path(
            checkpoint_path or PROJECT_ROOT / "models" / "pill_resnet152_dataclass01_aug0.pt"
        )
        label_path = Path(label_path) if label_path else _existing_path(
            data_dir / "pill_label_path_sharp_score.json",
            PROJECT_ROOT.parent / "pill_label_path_sharp_score.json",
        )
        metadata_root = Path(metadata_root) if metadata_root else _metadata_root(
            data_dir / "test_images",
            PROJECT_ROOT.parent,
        )

        self.model, self.device = load_model(self.checkpoint_path, device)
        self.label_map = PillLabelMap(label_path, metadata_root)

    def predict(self, image_path, top_k=3):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 1000:
            raise ValueError("top_k must be an integer between 1 and 1000.")

        image = load_image_tensor(image_path).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(image), dim=1)[0]
            scores, class_ids = torch.topk(probabilities, k=top_k)

        predictions = []
        for rank, (class_id, score) in enumerate(zip(class_ids.cpu().tolist(), scores.cpu().tolist()), 1):
            pill = self.label_map.resolve(class_id)
            predictions.append(
                {
                    "rank": rank,
                    "class_id": class_id,
                    **pill,
                    "score_percent": round(score * 100, 4),
                }
            )

        return {
            "status": "prediction_ready",
            "model": MODEL_NAME,
            "model_info": {
                "architecture": "ResNet152",
                "checkpoint": self.checkpoint_path.name,
                "device": str(self.device),
            },
            "predictions": predictions,
            "requires_user_confirmation": True,
            "score_interpretation": "Uncalibrated softmax score; not a medically validated probability.",
        }


_default_predictor = None


def predict(image_path, top_k=3):
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = PillPredictor()
    return _default_predictor.predict(image_path, top_k=top_k)