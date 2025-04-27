# SCGNet: Gender and Age Detection

SCGNet (Spatial Cross-scale Guided Network) is a Vision Transformer-based neural network designed for accurate gender and age detection from facial images.

## Features

- **Hierarchical Vision Transformer Architecture**: Efficiently processes multi-scale features
- **Spatial Autocorrelation Token Analysis (SATA)**: Analyzes spatial relationships between tokens
- **Cross-Scale Communication (CSC)**: Enhances feature representation across different scales
- **Local Feature Guided (LFG) Module**: Focuses on important local facial features

## Dataset

This project uses the [UTKFace dataset](https://susanqq.github.io/UTKFace/), which contains over 20,000 face images with annotations of age, gender, and ethnicity.

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
timm>=0.5.4
numpy>=1.19.5
matplotlib>=3.4.3
scikit-learn>=0.24.2
pandas>=1.3.3
pillow>=8.3.2
tqdm>=4.62.3
```

## Project Structure

```
SCGNet/
├── data/                  # Data handling utilities
│   ├── dataset.py         # UTKFace dataset loader
│   └── transforms.py      # Data augmentation transforms
├── models/                # Model architecture
│   ├── net.py             # SCGNet architecture
│   ├── patchhance.py      # Patch enhancement modules
│   └── SCCA.py            # Spatial correlation modules
├── utils/                 # Utility functions
│   ├── logger.py          # Logging utilities
│   ├── metrics.py         # Evaluation metrics
│   └── visualization.py   # Visualization tools
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference script for real-world usage
├── config.py              # Configuration parameters
└── README.md              # Project documentation
```

## Usage

### Training

```bash
python train.py --data_path /path/to/utkface --batch_size 32 --epochs 100
```

### Evaluation

```bash
python evaluate.py --data_path /path/to/utkface --checkpoint /path/to/model.pth
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.


