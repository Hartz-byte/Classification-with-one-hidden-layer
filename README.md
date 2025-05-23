![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-%23FA0F00.svg?logo=Jupyter&logoColor=white)
![From Scratch](https://img.shields.io/badge/Framework-From%20Scratch-critical)
![Neural Network](https://img.shields.io/badge/Model-1%20Hidden%20Layer%20Neural%20Network-blueviolet)
![Binary Classification](https://img.shields.io/badge/Task-Binary%20Classification-red)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25-success)
![NumPy](https://img.shields.io/badge/NumPy-%23113f8c.svg?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Science%20Plotting-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Planar%20Moons-orange)
![Tested](https://img.shields.io/badge/Tests-Passed-brightgreen)


# Planar Data Classification from Scratch (Without TensorFlow/PyTorch)
A neural network implementation in NumPy to classify non-linear planar data. This project walks through building and training a simple neural network with one hidden layer—step-by-step—without using any deep learning libraries like TensorFlow or PyTorch.

---

## Project Overview
This project implements a basic neural network capable of classifying planar 2D data. It explores how a neural network learns, forward and backward propagation, cost calculation, and decision boundaries—all from scratch using NumPy.
It is ideal for beginners looking to strengthen their intuition for how neural networks work under the hood.

---

## Dataset
The planar dataset is synthetic and generated using sklearn.datasets.make_moons() to create two interleaving half circles (a common test case for non-linear classification).
Each point has two input features, and the labels are binary (0 or 1).

---

## Neural Network Architecture
The model is a two-layer feedforward neural network:
- Input layer: 2 neurons (since input features are 2D)
- One Hidden layer: Variable neurons (1 to 5 in experiments)
- Output layer: 1 neuron (binary classification with sigmoid)

---

## Steps:
- Initialize parameters
- Forward propagation
- Compute cost
- Backward propagation
- Update parameters
- Repeat for multiple iterations

---

## Activation Functions:
- Hidden layer: tanh
- Output layer: sigmoid

---

## Requirements:
- Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- Scikit-learn

```bash
pip install numpy matplotlib scikit-learn
```

## Run the notebook:
```bash
jupyter notebook planar_classification.ipynb
```
---

## Results:
The network reaches ~90% accuracy with just 3-5 hidden units!

---

## Technologies Used
- NumPy – Numerical operations
- Matplotlib – Visualization
- Scikit-learn – Dataset generation
- Jupyter Notebook – Interactive exploration

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
