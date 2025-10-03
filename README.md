# Hepatic Vessel Segmentation using EfficientNet-B3

A deep learning pipeline for accurate segmentation of hepatic vessels from CT medical images using a 2.5D U-Net architecture with an EfficientNet-B3 encoder.

## Overview

This project addresses the challenge of segmenting hepatic vessels in medical images—a task complicated by complex anatomical structures, extreme class imbalance, and variable image quality. The pipeline achieves a peak Dice score of 79.34%, representing a substantial improvement over the baseline performance of 51.94%.

## Key Features

- **Efficient Architecture**: 2.5D U-Net with EfficientNet-B3 encoder pretrained on ImageNet
- **Robust Preprocessing**: Intensity clipping, isotropic resampling, z-score normalization, and CLAHE enhancement
- **Custom Loss Function**: Combined Tversky and Binary Cross-Entropy loss to handle severe class imbalance
- **Comprehensive Validation**: 3-fold cross-validation ensuring robust and reproducible results
- **Advanced Augmentation**: Extensive data augmentation using Albumentations library

## Project Structure

```
.
├── g062coursework4.py          # Main training pipeline implementation
├── G062COURSEWORK4.ipynb       # Main training pipeline jupyter file
├── MainResearchPaper.pdf       # Base Reseach paper used for the project 
├── _Report_courseworkG062.pdf  # Detailed technical report
├── fold_1_metrics.csv          # Training metrics for fold 1
├── fold_2_metrics.csv          # Training metrics for fold 2
├── fold_3_metrics.csv          # Training metrics for fold 3
├── visualisation_stats.csv     # Visualization statistics across epochs
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore configuration
└── README.md                   # This file
```

## Dataset

This project uses the **Medical Segmentation Decathlon Challenge Task 8 (MSDC-T8)** hepatic vessel dataset, which contains contrast-enhanced CT images with annotations for liver structures including hepatic vessels and tumors.

### Dataset Access

The dataset can be obtained from the Medical Segmentation Decathlon:

**Official Website**: [http://medicaldecathlon.com/](http://medicaldecathlon.com/)

### Dataset Structure

- **Training Images (imagesTr)**: Multi-dimensional CT scans for model training
- **Testing Images (imagesTs)**: Held-out set for final evaluation
- **Annotated Labels (labelsTr)**: Ground truth segmentation masks

## Requirements

### System Requirements

- NVIDIA GPU with CUDA support (tested on Tesla V100)
- Python 3.7+
- Minimum 16GB RAM recommended

### Python Dependencies

```bash
pip install SimpleITK
pip install torchio
pip install segmentation-models-pytorch
pip install torch torchvision
pip install albumentations
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install opencv-python
pip install tqdm
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hepatic-vessel-segmentation.git
cd hepatic-vessel-segmentation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the MSDC-T8 dataset from the Medical Segmentation Decathlon website and place the `Task08_HepaticVessel.tar` file in your working directory.

## Usage

### Preprocessing

The pipeline automatically handles preprocessing including:

1. **Orientation Standardization**: Converts images to RAS (Right-Anterior-Superior) orientation
2. **Intensity Windowing**: Clips CT intensities to [-100, 400] HU range
3. **Voxel Resampling**: Resamples to isotropic spacing of (1.0, 1.0, 1.0) mm
4. **Z-score Normalization**: Per-volume normalization with mean=0, std=1
5. **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization with grid size 8×8

### Training

Run the complete training pipeline:

```bash
python g062coursework4.py
```

The script will:
- Extract and preprocess the dataset
- Split data into train/validation sets (85/15 ratio)
- Perform 3-fold cross-validation
- Train for 30 epochs per fold
- Save best models and metrics
- Generate visualizations

### Model Architecture

- **Encoder**: EfficientNet-B3 with ImageNet pretraining
- **Decoder**: U-Net architecture with skip connections
- **Input**: 2.5D slices (3 adjacent slices: target ± 1)
- **Output**: Binary segmentation mask (vessel vs. background)

### Training Configuration

- **Optimizer**: AdamW with learning rate 5e-4
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Batch Size**: 8
- **Image Size**: 128×128 (resized from 256×256×256 volumes)
- **Loss Function**: Tversky Loss (α=0.7, β=0.3) + Weighted BCE (pos_weight=20)

## Results

### Performance Metrics

| Fold | Best Validation Dice (%) | Final Training Loss | Final Validation Loss |
|------|--------------------------|---------------------|----------------------|
| 1    | 76.06                    | 0.94                | 1.51                 |
| 2    | 72.01                    | 0.91                | 1.63                 |
| 3    | 79.34                    | 0.86                | 1.41                 |

**Average Dice Score**: 75.8%

This represents a **46% improvement** over the baseline Dice score of 51.94% reported in the original research.

### Key Improvements Over Baseline

1. **Enhanced Feature Extraction**: EfficientNet-B3 encoder vs. conventional ConvNet
2. **Better Normalization**: Z-score normalization vs. min-max scaling
3. **Improved Contrast**: CLAHE preprocessing for vessel boundary detection
4. **Specialized Loss**: Tversky + weighted BCE to address class imbalance
5. **Regularization**: AdamW optimizer with dynamic learning rate scheduling
6. **Extensive Augmentation**: Rotation, flipping, elastic deformation, gamma shifts, grid distortion

## Methodology Highlights

### Preprocessing Pipeline

Unlike traditional approaches, this implementation uses:
- Z-score normalization for consistent intensity distributions across diverse patient data
- CLAHE for enhanced local contrast, making vessel boundaries more distinguishable
- 2.5D approach to balance computational efficiency with spatial context

### Loss Function Design

The combined loss function specifically addresses vessel segmentation challenges:

```
L_total = L_Tversky(α=0.7, β=0.3) + L_BCE(pos_weight=20)
```

- **Tversky Loss**: Penalizes false negatives more heavily (critical for detecting small vessels)
- **Weighted BCE**: Strongly penalizes missed vessel pixels to counter class imbalance

### Data Augmentation

Comprehensive augmentation strategy using Albumentations:
- Horizontal flipping (p=0.5)
- Rotation up to 15° (p=0.5)
- Elastic transformation (p=0.3)
- Random gamma adjustment (p=0.3)
- Grid distortion (p=0.2)
- Random resized crop (p=0.4)

## Visualization

The pipeline generates extensive visualizations including:
- Predicted masks at multiple probability thresholds (0.3, 0.5, 0.7)
- Soft prediction heatmaps
- Training and validation loss curves
- Dice score progression across epochs
- Ground truth comparisons

Visualizations are automatically saved to `/content/drive/MyDrive/SegmentationVis/`

## Limitations and Future Work

### Current Limitations

1. **2.5D Approach**: Does not fully capture volumetric continuity
2. **Single Dataset**: Trained and validated on MSDC-T8 only
3. **Computational Requirements**: Requires GPU for practical training times

### Future Directions

1. **Architecture Extensions**:
   - Full 3D U-Net or hybrid 2.5D-3D architectures
   - Transformer-based components (TransUNet, Swin UNet)
   - Attention mechanisms for improved feature focusing

2. **Training Enhancements**:
   - Semi-supervised learning with unlabeled data
   - Self-supervised pretraining on medical datasets
   - Multi-dataset training for better generalization

3. **Deployment Optimization**:
   - Model pruning and quantization
   - Real-time inference optimization
   - Interactive clinical tools with uncertainty visualization

4. **Domain Extension**:
   - Application to other anatomical structures
   - Multi-organ segmentation capabilities
   - Cross-modality transfer learning

## Citation

If you use this work in your research, please cite:

```bibtex
@article{hepaticvesselseg2025,
  title={Hepatic Vasculature Segmentation via EfficientNet-B3},
  author={Group G062},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Medical Segmentation Decathlon Challenge for providing the dataset
- Segmentation Models PyTorch library for model architectures
- Original research team whose baseline work informed this project

## Contact

For questions, issues, or collaboration opportunities, please open an issue in this repository.

---

**Note**: This project was developed as part of the Machine Learning Practical coursework. The implementation prioritizes reproducibility, clinical applicability, and computational efficiency while achieving state-of-the-art segmentation performance.
