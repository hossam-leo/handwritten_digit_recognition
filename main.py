import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data_loader import load_and_preprocess_data
from src.model_builder import MLP
from src.trainer import train_model, plot_results

def main():
    # 1. & 2. DATA LOADING & PREPROCESSING
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 5. EXPERIMENTS
    experiments = [
        {"name": "Exp1 - Baseline", "hidden": [512, 256], "lr": 0.001, "dropout": False, "bn": False},
        {"name": "Exp2 - Deep Model", "hidden": [256, 128, 64], "lr": 0.001, "dropout": False, "bn": False},
        {"name": "Exp3 - High LR", "hidden": [512, 256], "lr": 0.01, "dropout": False, "bn": False},
        {"name": "Exp4 - Enhanced", "hidden": [512, 256], "lr": 0.001, "dropout": True, "bn": True},
    ]

    results_summary = []
    all_histories = []

    for exp in experiments:
        print(f"\n Starting: {exp['name']}")
        model = MLP(input_size=784, hidden_layers=exp["hidden"], use_dropout=exp["dropout"], use_batch_norm=exp["bn"])
        
        # 4. TRAINING
        history = train_model(model, train_loader, val_loader, lr=exp["lr"], epochs=3)
        all_histories.append((exp["name"], history))
        
        # Evaluation on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        print(f" {exp['name']} Finished. Test Acc: {test_acc:.2f}%")
        
        results_summary.append({
            "Experiment": exp["name"],
            "Hidden Layers": str(exp["hidden"]),
            "LR": exp["lr"],
            "Test Accuracy (%)": test_acc
        })
        
        # Save individual plots
        plot_results(history, results_dir, exp["name"])

    # 6. RESULTS
    print("\n FINAL RESULTS")
    df = pd.DataFrame(results_summary)
    print(df)
    df.to_csv(os.path.join(results_dir, "experiment_comparison.csv"), index=False)

    # 7. PLOTS (Comparison)
    plt.figure(figsize=(10, 6))
    for name, history in all_histories:
        plt.plot(history['train_loss'], label=f"{name} (Train)")
    
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "comparison_loss.png"))
    print(f"Comparison plot saved to {results_dir}/comparison_loss.png")

if __name__ == "__main__":
    main()
