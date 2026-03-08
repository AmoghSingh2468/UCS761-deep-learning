# UCS761-deep-learning
# Neural Networks Lab Exercises

This repository contains implementations of fundamental neural network algorithms using **Perceptrons and Multilayer Perceptrons (MLP)**.
All models are implemented **from scratch in Python without using built-in machine learning libraries** such as Scikit-learn or TensorFlow.

The purpose of these exercises is to understand the **mathematical foundations and working of neural networks**.

---

# Technologies Used

* Python
* NumPy
* Pandas
* Jupyter Notebook

---

# Lab Exercise 1

## Design Perceptron for Logic Gates

Implemented a **single layer perceptron** to simulate basic digital logic gates.

### Implemented Gates

* AND Gate
* OR Gate
* NAND Gate
* NOR Gate

### Concept

A perceptron calculates:

y = f(w₁x₁ + w₂x₂ + b)

Where

* x₁, x₂ = inputs
* w₁, w₂ = weights
* b = bias
* f = step activation function

By selecting appropriate weights and bias, a perceptron can model **linearly separable logic gates**.

---

# Lab Exercise 2

## Logistic Regression Using Single Layer Perceptron

Implemented **logistic regression using a single layer perceptron neural network** to classify glass types based on chemical composition.

### Dataset

Glass Identification Dataset

Kaggle Link
https://www.kaggle.com/datasets/uciml/glass

### Features

* Refractive Index
* Sodium
* Magnesium
* Aluminum
* Silicon
* Potassium
* Calcium
* Barium
* Iron

### Objective

Predict the **type of glass** using physical and chemical attributes.

### Reference

https://medium.com/analytics-vidhya/logistic-regression-using-single-layer-perceptron-neural-network-slpnn-31757c792d5d

---

# Lab Exercise 3

## Multiple Linear Regression Using Perceptron

Implemented **multiple linear regression from scratch using a perceptron model and gradient descent**.

### Dataset

Multiple Linear Regression Dataset

Kaggle Link
https://www.kaggle.com/datasets/hussainnasirkhan/multiple-linear-regression-dataset

### Objective

Predict the output variable based on multiple input features using a linear model.

### Key Concepts

* Weighted sum of inputs
* Gradient descent optimization
* Weight and bias updates

---

# Lab Exercise 4

## Multilayer Perceptron (MLP) Using Backpropagation

Implemented a **Multilayer Perceptron neural network with backpropagation** to predict the **age of abalone**.

### Dataset

Abalone Dataset

UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/1/abalone

### Input Features

Physical measurements of abalone including:

* Length
* Diameter
* Height
* Whole Weight
* Shucked Weight
* Viscera Weight
* Shell Weight

### Output

Number of **Rings**, which is used to estimate the age of abalone.

Age ≈ Rings + 1.5

### Model Architecture

Input Layer → Hidden Layer → Output Layer

Backpropagation is used to update weights and minimize prediction error.

---

# Repository Structure

```
Neural-Network-Lab
│
├── perceptron_logic_gates.ipynb
├── glass_classification_perceptron.ipynb
├── multiple_linear_regression_perceptron.ipynb
├── abalone_mlp_backpropagation.ipynb
└── README.md
```

---

# Learning Outcomes

Through these lab exercises, the following concepts are implemented and understood:

* Single Layer Perceptron
* Logistic Regression
* Multiple Linear Regression
* Gradient Descent
* Multilayer Perceptron (MLP)
* Backpropagation Algorithm

---

# Author

Amogh Singh
