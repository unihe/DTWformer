

# DTWformer

DTWformer is a Transformer-based model designed for multivariate time series forecasting. It leverages a dynamic time warping (DTW)-based attention mechanism to overcome the limitations of traditional Transformer models in handling noisy and misaligned time series data. Additionally, it employs adaptive patch selection and multi-scale modeling to capture both short-term local patterns and long-term global dependencies.

## Key Features

- **DTW-Attention Mechanism**  
  Replaces traditional dot-product attention with a Sakoe-Chiba constrained SCsoft-DTW, enabling dynamic sequence alignment and increased robustness to noise and temporal misalignments.

- **Adaptive Patch Selection**  
  Utilizes FFT to extract dominant frequency components, allowing for automatic selection of patch sizes that capture multi-scale temporal features.

- **Multi-scale Modeling**  
  Combines intra-patch (local) and inter-patch (global) attention to effectively model both fine-grained and long-range temporal dependencies.

- **Dynamic Hybrid Attention Training**  
  Uses DTW-attention during early training epochs for precise alignment and switches to the more efficient dot-product attention in later stages, balancing performance and computational cost.

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Additional dependencies are listed in [requirements.txt](requirements.txt).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/unihe/DTWformer.git
   cd DTWformer
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The project supports several widely used multivariate time series datasets such as Weather, Traffic, Electricity, ILI, Exchange, and ETT. Use the provided data preprocessing scripts to download and format the datasets. Ensure that the data is split into training, validation, and testing sets (e.g., a 70%/10%/20% ratio) as required by the model.

## Code Structure

```
DTWformer/
├── models/              # Implementation of the model architecture including DTW-attention and adaptive patch selection modules
├── scripts/             # Training, evaluation, and testing scripts
├── utils/               # Data loading, preprocessing, and utility functions
├── requirements.txt     # List of project dependencies
└── README.md            # This file
```

## Usage

### Training

To start training using a predefined configuration file (e.g., for the ETTh1 dataset), run:
```bash
python main.py --config configs/ETTh1.yaml
```

### Evaluation & Forecasting

After training, evaluate the model or perform forecasting with:
```bash
python evaluate.py --config configs/ETTh1.yaml --checkpoint path/to/checkpoint.pth
```

## Experimental Results

The paper evaluates DTWformer on multiple datasets, including Weather, Traffic, Electricity, ILI, Exchange, and ETT. The experiments demonstrate that DTWformer significantly outperforms existing methods in terms of both MSE and MAE, especially on datasets with high noise levels and temporal misalignments. For detailed experimental results and ablation studies, please refer to the corresponding sections in the paper.

## Citation

If you use this code in your research, please cite the paper as follows:
```

```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact & Feedback

For questions or suggestions, please open an issue on GitHub or contact us via email.

---

