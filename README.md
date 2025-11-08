# CheXpert Chest X-Ray Classification with DenseNet121

A deep learning project implementing DenseNet121 for chest radiograph pathology classification using the CheXpert dataset. This project explores transfer learning, uncertainty label handling, and GUI-based inference.

## Project Overview

This project replicates and extends the methodology from the CheXpert paper, which introduces a large-scale chest radiograph dataset with uncertainty labels. The goal is to build an efficient medical imaging classifier using DenseNet121, one of the top-performing architectures for this task.

### Key Features
- **Transfer Learning Implementation**: Uses pretrained ImageNet weights for efficient training
- **Multiple Framework Support**: TensorFlow/Keras and PyTorch implementations
- **Streamlit GUI**: User-friendly interface for model inference on new chest X-rays
- **Uncertainty Handling**: Explores different approaches to manage uncertain labels
- **Checkpoint Management**: Supports training resumption and model state preservation

## Dataset

The CheXpert dataset contains 224,316 chest radiographs from 65,240 patients with 14 distinct pathologies labeled as positive (1), negative (0), or uncertain (u). For computational efficiency, this project uses a smaller sample of images and a subset of pathologies.

Download the dataset from Kaggle: https://www.kaggle.com/datasets/willarevalo/chexpert-v10-small

## Architecture: DenseNet121

DenseNet121 comprises three key block types that work together to build an efficient deep network:

**Bottleneck Blocks**: Reduce the number of feature maps within dense blocks, making computation more efficient while maintaining representational capacity.

**Dense Blocks**: The core innovation where all layers connect to every other layer in a feed-forward manner. Each layer concatenates feature maps from previous layers as input, transforms them to higher dimensions, and passes concatenated outputs to subsequent layers. This dense connectivity pattern enables efficient feature reuse and gradient flow.

**Transition Blocks**: Bridge dense blocks by reducing spatial dimensions and feature map counts while preserving relevant information, creating a hierarchical feature representation.

**Feature Maps** (Activation Maps): The output of applying a filter/kernel across an input, representing learned features at different levels of abstraction.

## Project Approach

### Development Strategy
1. **Foundation Building**: Started with practice implementations of DenseNet121 from scratch using PyTorch
2. **Framework Exploration**: Compared PyTorch and TensorFlow implementations to understand framework differences
3. **Production Implementation**: Adopted transfer learning for efficiency due to 121 layers and computational constraints
4. **Practical Adaptations**:
   - Reduced training/testing dataset size for feasible training time
   - Subset of pathology labels used instead of all 14
   - Limited epochs in training phase
   - Transfer learning chosen over training from scratch

### Why Transfer Learning?
Given the computational cost of training 121-layer networks and the limited number of epochs, leveraging ImageNet pretrained weights provides strong initial feature representations and dramatically reduces training time while maintaining high accuracy.

## File Structure

- `docs/` - Documentation and research papers
  - `CheXpert.pdf` - Chrome PDF documentation
  - `CheXpertPaper.pdf` - Research paper reference
- `models/` - Main model implementations and weights
  - `DenseNet121_ChExpert.py` - PyTorch implementation
  - `DenseNet121_TF.py` - TensorFlow/Keras implementation
- `practice/` - Practice projects for learning
  - `CIFAR_10UsingDenseNet121/` - Building DenseNet from scratch
  - `CIFAR_10UsingDenseNet121TL/` - Transfer learning approach
- `gui/` - Streamlit GUI application
  - `gui.py` - Interactive inference interface
- `.gitignore` - Git ignore file
- `README.md` - This file

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- TensorFlow 2.6+ (for TF implementation)
- Streamlit
- NumPy, Pandas, scikit-learn, Matplotlib

### Setup
```bash
# Clone repository
git clone https://github.com/mdarshanrshaikh/chexpert-densenet121.git
cd chexpert-densenet121

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install tensorflow
pip install streamlit pandas numpy scikit-learn matplotlib pillow
```

## Usage

### Training the Model

**PyTorch Implementation:**
```bash
python models/DenseNet121_ChExpert.py
```

**TensorFlow Implementation:**
```bash
python models/DenseNet121_TF.py
```

### GUI Inference

Run the Streamlit application for interactive predictions:
```bash
streamlit run gui/gui.py
```

The GUI allows you to:
- Upload chest X-ray images
- Get model predictions with confidence scores
- Visualize model attention/feature activations
- Batch process multiple radiographs

### Practice Projects

Explore the CIFAR-10 implementations to understand DenseNet121 mechanics:
- `practice/CIFAR_10UsingDenseNet121/` - Building DenseNet from scratch
- `practice/CIFAR_10UsingDenseNet121TL/` - Transfer learning approach

## Key Learnings

1. **Dense Connectivity Benefits**: The dense connections between layers enable efficient gradient propagation and feature reuse, allowing deeper networks with fewer parameters.

2. **Transfer Learning Efficiency**: Pretrained ImageNet weights provide strong initial representations. Even with limited training epochs, transfer learning significantly outperforms training from scratch on small datasets.

3. **Uncertainty Label Handling**: Different pathologies benefit from different label strategies (ignore uncertain, combine with positive/negative, or multiclass classification).

4. **Computational Trade-offs**: Balancing model capacity, training time, and accuracy requires careful data sampling and hyperparameter tuning, especially with 224K+ image datasets.

5. **Framework Comparison**: PyTorch and TensorFlow have different abstractions but comparable final performance. Framework choice depends on workflow preferences and deployment requirements.

## Model Performance

Despite computational constraints limiting training epochs, the transfer learning approach achieved strong accuracy on the CheXpert pathology classification task, demonstrating the effectiveness of leveraging pretrained representations for medical imaging.

## References

**Paper**: CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison
- ResNet Explanation: https://www.youtube.com/watch?v=woEs7UCaITo
- DenseNet Explanation: https://www.youtube.com/watch?v=y81RrUHMRSA

## Contact

Questions or feedback? Feel free to open an issue or reach out.
