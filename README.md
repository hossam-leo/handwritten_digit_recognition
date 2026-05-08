# Handwritten Digit Recognition using MLP (PyTorch)

This project implements a Multilayer Perceptron (MLP) for recognizing handwritten digits using the MNIST dataset, built with PyTorch. It includes data preprocessing, model building, and multiple experiments to compare the effects of different hyperparameters.

## Project Structure
- `data/`: Stores the downloaded MNIST dataset.
- `src/`: Contains modular source code.
    - `data_loader.py`: Script for loading and preprocessing the MNIST dataset for PyTorch.
    - `model_builder.py`: Script for defining the MLP model architecture using PyTorch.
    - `trainer.py`: Script for training the model and plotting results.
- `results/`: Contains experiment results, including plots and a comparison CSV.
- `main.py`: The main script to run all experiments and generate results.
- `requirements.txt`: List of required Python packages.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the main script: `python main.py`.

## Model Architecture
The MLP models consist of:
- Input Layer: 784 neurons (28x28 pixels).
- Hidden Layers: Configurable number of dense layers with ReLU activation (or other specified activation).
- Output Layer: 10 neurons, Softmax activation (implicitly handled by `CrossEntropyLoss`).

## Experiments
The project includes four experiments:
1. **Exp1 - Baseline**: Standard MLP with two hidden layers ([512, 256]), ReLU activation, and a learning rate of 0.001.
2. **Exp2 - Deep Model**: A deeper network with three hidden layers ([256, 128, 64]), ReLU activation, and a learning rate of 0.001.
3. **Exp3 - High LR**: Baseline architecture but with a higher learning rate of 0.01.
4. **Exp4 - Enhanced**: Baseline architecture with Dropout and Batch Normalization layers.

## Results
After running `main.py`, you can find the performance comparison in `results/experiment_comparison.csv` and the corresponding training plots in the `results/` folder.

### Experiment Comparison Table
| Experiment        | Hidden Layers  | LR    | Test Accuracy (%) |
|-------------------|----------------|-------|-------------------|
| Exp1 - Baseline   | [512, 256]     | 0.001 | 97.44             |
| Exp2 - Deep Model | [256, 128, 64] | 0.001 | 97.44             |
| Exp3 - High LR    | [512, 256]     | 0.010 | 94.42             |
| Exp4 - Enhanced   | [512, 256]     | 0.001 | 97.89             |

### Visualizations
Individual training and validation loss/accuracy plots for each experiment are saved in the `results/` directory (e.g., `exp1_-_baseline_results.png`). A comparison plot of training loss across all experiments is saved as `comparison_loss.png`.
