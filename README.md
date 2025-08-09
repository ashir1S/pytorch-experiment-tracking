# 📈 PyTorch Experiment Tracking

A professional Jupyter Notebook demonstrating best practices for tracking machine learning experiments with PyTorch.  
This guide covers fine-tuning a pre-trained computer vision model for a custom image classification task, with detailed TensorBoard logging for monitoring progress.

## ✨ Features

- Transfer Learning – Fine-tune a pre-trained EfficientNet_B0 model from torchvision for high performance on a small dataset.
- Custom Dataset Handling – Automatic download, extraction, and preparation of a custom image dataset.
- Experiment Logging – Track key metrics like loss and accuracy with torch.utils.tensorboard.SummaryWriter.
- Model Checkpointing – Automatically save the best-performing model based on validation performance.
- Reproducible Workflow – Well-structured notebook for reuse in your own projects.

## 📁 Project Workflow

1. Setup – Install required dependencies and set up the environment.
2. Data Acquisition – Download and prepare the pizza_steak_sushi dataset.
3. Data Preparation – Create PyTorch Dataset and DataLoader objects.
4. Model Configuration – Load EfficientNet_B0 and modify the classifier head.
5. Training & Evaluation – Train the model, log results to TensorBoard, and save the best model.
6. Visualization – Launch TensorBoard to explore experiment results.

## 📦 Dependencies

Install the following packages:

pip install torch>=1.12 torchvision>=0.13 matplotlib tqdm requests torchinfo tensorboard

## 🚀 Getting Started

1. Clone the Repository

git clone https://github.com/<your-username>/pytorch-experiment-tracking.git
cd pytorch-experiment-tracking

2. Open the Notebook  
   Use Google Colab (recommended) or run locally in JupyterLab / VS Code.

3. Run the Cells Sequentially from top to bottom.

## 📊 Using TensorBoard

After training, start TensorBoard with:

tensorboard --logdir runs

Then open your browser at http://localhost:6006 to view training curves and metrics.

## 🙏 Acknowledgements

This project builds upon the excellent resources and tutorials from the PyTorch ecosystem.
