# Toilet Tissue-ness Game 🧻

[![Latest Release](https://img.shields.io/github/v/release/taka-sho-1231/toilet-tissue-ness?display_name=tag)](https://github.com/taka-sho-1231/toilet-tissue-ness/releases/latest)
[![Release Date](https://img.shields.io/github/release-date/taka-sho-1231/toilet-tissue-ness)](https://github.com/taka-sho-1231/toilet-tissue-ness/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/taka-sho-1231/toilet-tissue-ness/total)](https://github.com/taka-sho-1231/toilet-tissue-ness/releases)
[![Stars](https://img.shields.io/github/stars/taka-sho-1231/toilet-tissue-ness)](https://github.com/taka-sho-1231/toilet-tissue-ness/stargazers)
[![Open Issues](https://img.shields.io/github/issues/taka-sho-1231/toilet-tissue-ness)](https://github.com/taka-sho-1231/toilet-tissue-ness/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/taka-sho-1231/toilet-tissue-ness)](https://github.com/taka-sho-1231/toilet-tissue-ness/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/taka-sho-1231/toilet-tissue-ness)](https://github.com/taka-sho-1231/toilet-tissue-ness/commits)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#setup)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)
[![Contributing](https://img.shields.io/badge/Contributing-Guide-informational)](./CONTRIBUTING.md)

[![Check PR Source](https://github.com/taka-sho-1231/toilet-tissue-ness/actions/workflows/check-pr-source.yaml/badge.svg)](https://github.com/taka-sho-1231/toilet-tissue-ness/actions/workflows/check-pr-source.yaml)
[![Auto Tag](https://github.com/taka-sho-1231/toilet-tissue-ness/actions/workflows/auto-tag.yaml/badge.svg)](https://github.com/taka-sho-1231/toilet-tissue-ness/actions/workflows/auto-tag.yaml)
[![Create Release](https://github.com/taka-sho-1231/toilet-tissue-ness/actions/workflows/create-release.yaml/badge.svg)](https://github.com/taka-sho-1231/toilet-tissue-ness/actions/workflows/create-release.yaml)

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

## Contributing

See the contribution guide: [CONTRIBUTING.md](./CONTRIBUTING.md) (JP/EN). PRs should target the `dev` branch, and issues are welcome.

## Future Ideas

- **Pygame Implementation**: Create a standalone game with GUI using pygame
- **Other Categories**: Try different categories like cats vs dogs
- **Other Models**: Try different models (VGG16, MobileNet, etc.) and compare predictions
