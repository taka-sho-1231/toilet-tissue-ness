# Toilet Tissue-ness Game 🧻

[![Latest Release](https://img.shields.io/github/v/release/taka-sho-1231/toilet-tissue-ness?display_name=tag)](https://github.com/taka-sho-1231/toilet-tissue-ness/releases/latest)

An interactive game where you predict which random image the AI model thinks looks most like toilet tissue.

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: This installs the default PyTorch version. For GPU support, see [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Play

Open `play.ipynb` and run all cells. Predict which image the AI rates highest for toilet tissue-ness!

## How It Works

- Uses ResNet50 (ImageNet pre-trained) to score random images
- Target class: 999 (toilet tissue, toilet paper, bathroom tissue)
- Your goal: Predict which image the AI model will score highest

## Future Ideas

- **Pygame Implementation**: Create a standalone game with GUI using pygame
- **Other Categories**: Try different categories like cats vs dogs
- **Other Models**: Try different models (VGG16, MobileNet, etc.) and compare predictions
