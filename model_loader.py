from pathlib import Path

import torch
from torchvision import models


MODEL_NAME = "resnet152_dataclass01"
NUM_CLASSES = 1000


def select_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return selected


def load_model(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Required checkpoint was not found: {checkpoint_path}. "
            "No randomly initialized model will be used."
        )

    selected_device = select_device(device)
    model = models.resnet152(weights=None, num_classes=NUM_CLASSES)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("Checkpoint must contain a state dictionary under the 'model' key.")

        state_dict = checkpoint["model"]
        normalized_state = {}
        for key, value in state_dict.items():
            normalized_key = key[len("module."):] if key.startswith("module.") else key
            if normalized_key in normalized_state:
                raise ValueError(f"Duplicate state-dictionary key after prefix removal: {normalized_key}")
            normalized_state[normalized_key] = value

        model.load_state_dict(normalized_state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not strictly load ResNet152 weights from {checkpoint_path}: {exc}"
        ) from exc

    model.to(selected_device)
    model.eval()
    return model, selected_device