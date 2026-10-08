from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


IMAGE_SIZE = (224, 224)
PREPROCESS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_image_tensor(image_path):
    image_path = Path(image_path)
    try:
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                raise ValueError(
                    f"Expected an RGB image, got mode {image.mode!r}: {image_path}"
                )
            if image.size != IMAGE_SIZE:
                raise ValueError(
                    f"Expected a 224x224 image, got {image.width}x{image.height}: {image_path}"
                )
            tensor = PREPROCESS(image.copy())
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read image: {image_path}: {exc}") from exc

    if tuple(tensor.shape) != (3, 224, 224):
        raise ValueError(f"Expected a 3x224x224 tensor, got {tuple(tensor.shape)}")
    return tensor