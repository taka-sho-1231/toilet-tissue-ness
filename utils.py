from enum import Enum
from io import BytesIO
from pathlib import Path
import random
from typing import Optional, Tuple, Union
import time
import warnings

import numpy as np
from PIL import Image
import requests
import torch
from torchvision import models, transforms


def get_random_image(dir_path: Optional[Union[str, Path]] = None, size: Tuple[int, int] = (224, 224), online: bool = False) -> Image.Image:
    """Get random image from directory, URL, or online service."""
    width, height = size

    if dir_path and isinstance(dir_path, str) and dir_path.startswith(("http://", "https://")):
        # Try a few times before falling back. Network errors / 5xx may be transient.
        retries = 3
        backoff = 0.5
        last_exc = None
        for attempt in range(retries):
            try:
                resp = requests.get(dir_path, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                if img.size != (width, height):
                    img = img.resize((width, height), Image.LANCZOS)
                return img
            except requests.RequestException as e:
                last_exc = e
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                warnings.warn(f"Failed to fetch image from URL after {retries} attempts: {e}")
            except Exception as e:
                # PIL might raise OSError when opening/decoding
                last_exc = e
                warnings.warn(f"Failed to open image from URL: {e}")

        # Fallback: return a neutral placeholder image instead of raising.
        placeholder = Image.new("RGB", (width, height), color=(128, 128, 128))
        return placeholder

    if online:
        # Use retries + backoff because remote image service can return 5xx intermittently.
        retries = 3
        backoff = 0.5
        last_exc = None
        # Add a cache-busting query to reduce chance of cached errors
        url = f"https://picsum.photos/{width}/{height}?random={random.randint(0, 2**31 - 1)}"
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                if img.size != (width, height):
                    img = img.resize((width, height), Image.LANCZOS)
                return img
            except requests.RequestException as e:
                last_exc = e
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                warnings.warn(f"Failed to fetch image online after {retries} attempts: {e}")
            except Exception as e:
                last_exc = e
                warnings.warn(f"Failed to open online image: {e}")

        # Fallback placeholder if remote service unavailable.
        placeholder = Image.new("RGB", (width, height), color=(128, 128, 128))
        return placeholder

    if dir_path:
        p = Path(dir_path)

        if p.is_file():
            try:
                img = Image.open(p).convert("RGB")
                if img.size != (width, height):
                    img = img.resize((width, height), Image.LANCZOS)
                return img
            except Exception as e:
                raise ValueError(f"Failed to load image from file: {e}")

        if not p.is_dir():
            raise ValueError(f"Path is not a directory or file: {dir_path}")

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
        files = []
        for e in exts:
            files.extend(p.glob(e))
        files = [f for f in files if f.is_file()]
        if not files:
            raise ValueError(f"No image files found in directory: {dir_path}")
        chosen = random.choice(files)
        img = Image.open(chosen).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        return img

    raise ValueError("Failed to get image. Specify dir_path or set online=True.")


class ModelName(Enum):
    RESNET50 = 'resnet50'
    RESNET18 = 'resnet18'
    ALEXNET = 'alexnet'
    VGG16 = 'vgg16'
    MOBILENET_V2 = 'mobilenet_v2'
    DENSENET121 = 'densenet121'
    INCEPTION_V3 = 'inception_v3'


def load_imagenet_model(model: ModelName | str = ModelName.RESNET50, device: str | None = None):
    """Load ImageNet pretrained model with preprocessing."""
    name = model.value if isinstance(model, ModelName) else str(model).lower()
    torch_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    torch_mapping = {
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'alexnet': (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
        'vgg16': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
        'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
        'densenet121': (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
        'inception_v3': (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),
    }

    if name not in torch_mapping:
        raise ValueError(f"Unknown model name: {model}")

    model_fn, weights = torch_mapping[name]
    model = model_fn(weights=weights)
    model.to(torch_device)
    model.eval()

    input_size = 299 if name == 'inception_v3' else 224
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess, input_size


def acquire_images(num_images: int, size: int, online: bool = True) -> list[Image.Image]:
    """Acquire multiple images."""
    images = []
    for i in range(num_images):
        try:
            img = get_random_image(online=online, size=(size, size))
            images.append(img)
            print(f"Image {i+1}/{num_images} acquired")
        except Exception as e:
            # If something unexpected happens, append a placeholder and continue
            warnings.warn(f"Failed to acquire image {i+1}: {e} -- adding placeholder image and continuing.")
            placeholder = Image.new("RGB", (size, size), color=(128, 128, 128))
            images.append(placeholder)
            print(f"Image {i+1}/{num_images} -- placeholder used")
    return images


def run_inference(model, preprocess, images: list[Image.Image], class_idx: int):
    """Run batch inference on images."""
    import torch.nn.functional as F
    
    batch = torch.stack([preprocess(img) for img in images])
    device = next(model.parameters()).device
    batch = batch.to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        scores = probs[:, class_idx].cpu().numpy()
    
    return scores


def display_results(tissue_scores: np.ndarray, selected_idx: int, num_images: int):
    """Display results for selected image."""
    if not (0 <= selected_idx < num_images):
        print(f"Invalid choice. Please enter a number between 1 and {num_images}.")
        return

    selected_score = tissue_scores[selected_idx]
    actual_max_idx = tissue_scores.argmax()
    actual_max_score = tissue_scores[actual_max_idx]
    
    print("\n" + "="*50)
    print("=== RESULTS ===")
    print("="*50)
    print(f"\nYour choice: Image {selected_idx + 1}")
    print(f"Toilet Tissue-ness: {selected_score:.6f} ({selected_score*100:.4f}%)")
    
    print(f"\n--- All Scores ---")
    for i, score in enumerate(tissue_scores):
        marker = " ← YOUR CHOICE" if i == selected_idx else ""
        marker += " ★ HIGHEST" if i == actual_max_idx else ""
        print(f"Image {i+1}: {score:.6f} ({score*100:.4f}%){marker}")
    
    print(f"\nActual highest: Image {actual_max_idx + 1} ({actual_max_score*100:.4f}%)")
    
    if selected_idx == actual_max_idx:
        print("\n🎉 Correct! You predicted the AI's top choice!")
    else:
        print(f"\n💡 The AI rated Image {actual_max_idx + 1} highest.")