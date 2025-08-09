PyTorch Experiment Tracking
This repository contains a Jupyter Notebook that demonstrates a professional workflow for tracking machine learning experiments. The project uses PyTorch to fine-tune a pre-trained computer vision model for a custom image classification task and leverages TensorBoard's SummaryWriter for logging and visualization.

📝 Project Description
The primary goal of this project is to classify images into one of three distinct classes: pizza, steak, or sushi. The Jupyter Notebook walks through the entire process, from data acquisition to model training and performance tracking. Key steps include:

Setup and Dependencies: Installing necessary libraries like torch, torchvision, and torchinfo.

Data Handling: Automating the download and preparation of a custom image dataset.

Model Initialization: Using a pre-trained EfficientNet_B0 model from torchvision to leverage transfer learning.

Model Customization: Freezing the feature extraction layers and modifying the classifier head to align with the three-class problem.

Training & Logging: Implementing a training loop that utilizes torch.utils.tensorboard.SummaryWriter to log metrics such as loss and accuracy for both training and testing datasets.

Model Checkpointing: The training loop is configured to save the model's state dictionary for the epoch with the best validation performance, ensuring you keep the most effective version of the model.

🚀 Key Sections of the Notebook
The notebook is logically structured to guide you through the process:

Getting Setup: Ensures all required dependencies are installed and essential utility scripts are available.

Get Data: Defines and uses a download_data function to automatically retrieve and extract the "pizza_steak_sushi" dataset.

Create Datasets and DataLoaders: Demonstrates how to prepare the data for training by creating datasets and data loaders, including using automated transforms from the pre-trained model's weights.

Get a Pre-trained Model: Details the process of loading EfficientNet_B0, freezing its base layers, and adjusting the final classification layer.

Train Model and Track Results: The core of the project, showcasing a training loop with CrossEntropyLoss and Adam optimizer, and integrating SummaryWriter for logging. This section also includes the logic for saving the model with the best validation accuracy.

View Our Model's Results in TensorBoard: Provides the command to launch TensorBoard, allowing for a visual inspection of the logged metrics and a deeper understanding of the model's performance over time.

📦 Dependencies
To run this notebook, you will need the following dependencies:

torch (>= 1.12)

torchvision (>= 0.13)

matplotlib

tqdm

requests

zipfile

pathlib

torchinfo

tensorboard

▶️ How to Run
Download the Notebook: Clone this repository or download the PyTorch_Experiment_Tracking.ipynb file.

Setup Environment: This notebook is configured for a Google Colab environment with GPU access, but can be adapted for a local setup.

Install Dependencies: The notebook includes pip commands to install the necessary packages. Simply run the cells sequentially.

Execute: Open the notebook in a Jupyter environment (JupyterLab, VS Code, or Google Colab) and run all cells in order.
