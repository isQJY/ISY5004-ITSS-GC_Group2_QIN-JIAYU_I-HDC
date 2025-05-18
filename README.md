# i-HDC: Hybrid Dilated Convolutional Architecture for Traffic Sign Recognition

This repository contains the official implementation of the **i-HDC** model described in the paper *"i-HDC: A Hybrid Dilated Convolutional Architecture for Multi-Scale Feature Learning in Traffic Sign Recognition"* by Jiayu Qin, NUS-ISS, National University of Singapore. The i-HDC model achieves state-of-the-art performance in traffic sign recognition, with 98.7% accuracy on the GTSRB dataset and 94.2% on CIFAR-10, by leveraging a hybrid architecture that combines standard and dilated convolutions for multi-scale feature learning.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview
Traffic sign recognition is critical for autonomous driving and intelligent transportation systems but poses challenges due to high inter-class similarity, complex backgrounds, and varying environmental conditions. Traditional Convolutional Neural Networks (CNNs) are limited by small receptive fields, while Dilated CNNs (DCNNs) may sacrifice fine-grained details. The i-HDC model addresses these issues with a novel three-stage architecture:
1. **Mixed Receptive Field (MRF) Module**: Combines 3x3 and 5x5 dilated convolutions to capture both local and global features.
2. **Adaptive Feature Extraction Module**: Uses standard convolutions and adaptive pooling to balance detail retention and contextual aggregation.
3. **Dynamic Classification Head**: Adapts to varying class counts across datasets without structural modifications.

This repository provides a PyTorch implementation, including training and evaluation scripts, dataset preprocessing utilities, and configuration files to replicate the results reported in the paper.

## Installation
Follow these steps to set up the i-HDC project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/i-HDC.git
   cd i-HDC
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   - `torch>=1.9.0`
   - `torchvision>=0.10.0`
   - `numpy>=1.19.0`
   - `PyYAML>=5.4.0`
   - `tqdm>=4.62.0`
   - `pandas>=1.3.0`
   - `Pillow>=8.3.0`
   - `scikit-learn>=0.24.0`

4. **Verify Installation**:
   Ensure PyTorch is correctly installed with GPU support (if available):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Datasets
The i-HDC model was evaluated on two datasets:
1. **GTSRB (German Traffic Sign Recognition Benchmark)**:
   - Contains 51,839 RGB images across 43 classes, with resolutions ranging from 15x15 to 250x250 pixels.
   - Download: [GTSRB Dataset](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)
   - Expected directory structure after download:
     ```
     GTSRB/
     ├── Final_Training/
     │   ├── Images/
     │   └── GT-final_train.csv
     ├── Final_Test/
     │   ├── Images/
     │   └── GT-final_test.csv
     ```

2. **CIFAR-10**:
   - Contains 60,000 RGB images (32x32 resolution) across 10 classes.
   - Automatically downloaded via `torchvision.datasets.CIFAR10`.

**Preprocessing**:
Run the preprocessing script to resize images to 64x64 and organize splits:
```bash
python data/prepare_data.py --dataset gtsrb --data_dir /path/to/GTSRB
```
For CIFAR-10, the script handles downloading and preprocessing automatically:
```bash
python data/prepare_data.py --dataset cifar10 --output_dir ./data
```

## Usage
### Training
Train the i-HDC model using the provided configuration file:
```bash
python train.py --config configs/config.yaml
```
The `config.yaml` specifies hyperparameters, dataset paths, and training settings, including:
- `dataset`: `gtsrb` or `cifar10`
- `batch_size`: 64
- `epochs`: 60
- `learning_rate`: 0.001
- `optimizer`: AdamW
- `weight_decay`: 0.001

### Evaluation
Evaluate a trained model:
```bash
python evaluate.py --model_path /path/to/checkpoint.pth --dataset gtsrb --data_dir /path/to/GTSRB
```
This computes accuracy and confusion rates, replicating the paper’s results.

### Example
To train on GTSRB:
```bash
python train.py --config configs/config.yaml
```
To evaluate:
```bash
python evaluate.py --model_path checkpoints/i_hdc_gtsrb.pth --dataset gtsrb --data_dir ./data/GTSRB
```

## Model Architecture
The i-HDC model consists of three stages, designed to balance local and global feature extraction:
1. **Mixed Receptive Field Module**:
   - Parallel 3x3 and 5x5 dilated convolutions (dilation rate=2, 32 channels each).
   - GELU activation, BatchNorm, and stochastic depth regularization (0.2 probability).
   - Captures fine details (e.g., numeral shapes) and broader context (e.g., sign contours).
2. **Adaptive Feature Extraction Module**:
   - 3x3 standard convolutions (64 and 128 channels).
   - Adaptive max pooling (to 16x16) and average pooling (to 8x8).
   - Balances local detail retention and global feature aggregation.
3. **Dynamic Classification Head**:
   - Flattening, fully connected layer (512 units), LayerNorm, and Dropout (0.5).
   - Output layer adjusts to dataset classes (43 for GTSRB, 10 for CIFAR-10).

The model, implemented in `models/i_hdc.py`, uses approximately 2.3 million parameters, ensuring efficiency for real-time applications.

## Results
The i-HDC model achieves the following performance:
- **GTSRB**:
  - Accuracy: 98.7%
  - Speed Limit 60/80 confusion rate: 1.8% (vs. 7.2% for baseline CNN)
  - Convergence: Reaches 95% training accuracy in 8 epochs
- **CIFAR-10**:
  - Accuracy: 94.2%
  - Robustness: Maintains 73.6% accuracy under FGSM attack (ε=0.03)
- **Efficiency**:
  - Parameters: ~2.3 million
  - Inference latency: 23.1ms on NVIDIA Jetson AGX Xavier

These results demonstrate i-HDC’s superior performance over traditional CNNs (95.1% on GTSRB) and standard DCNNs (97.3% on GTSRB), with significant reductions in misclassification rates for similar signs.

| Model           | GTSRB Accuracy (%) | Speed Limit 60/80 Confusion (%) | Parameters (M) |
|-----------------|--------------------|----------------------------------|----------------|
| Traditional CNN | 95.1               | 7.2                              | ~2.5           |
| Standard DCNN   | 97.3               | 2.9                              | ~2.5           |
| ResNet18        | 95.1               | 5.1                              | 11             |
| i-HDC           | **98.7**           | **1.8**                          | 2.3            |

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Please submit issues for bug reports, feature requests, or documentation improvements. Ensure code follows PEP 8 style guidelines and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this code or the i-HDC model in your research, please cite:
```bibtex
@article{qin2025ihdc,
  title={i-HDC: A Hybrid Dilated Convolutional Architecture for Multi-Scale Feature Learning in Traffic Sign Recognition},
  author={Qin, Jiayu},
  journal={TBD},
  year={2025},
  institution={NUS-ISS, National University of Singapore}
}
```

---
