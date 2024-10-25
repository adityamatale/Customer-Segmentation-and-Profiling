# Customer Segmentation and Profiling System

A web application for customer segmentation and profiling using advanced clustering algorithms. This project includes an interactive interface that allows users to manually input data or upload CSV files to determine the segmentation of customers based on financial parameters. The system was developed using Python and includes a machine learning pipeline for clustering and analyzing customer profiles.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Accuracies](#model-accuracies)
- [Clustering Implementation](#clustering-implementation)

## Project Overview
This application performs customer segmentation using K-means, DBSCAN, and Agglomerative clustering techniques, with accuracy scores of **0.9292**, **0.9847**, and **0.9549**, respectively. Principal Component Analysis (PCA) was used for dimensionality reduction and visualization, enhancing the interpretability of high-dimensional customer data.

The app's primary purpose is to segment customers based on behavioral and financial attributes, providing organizations with data-driven insights into customer profiles, allowing for personalized marketing and resource allocation.

## Features
- **Clustering Algorithms**: Supports K-means, DBSCAN, and Agglomerative clustering models.
- **Data Input**: Choose between manual input fields or uploading a CSV file.
- **Interactive Visualizations**: View histograms for each customer attribute within each cluster.
- **Easy Integration**: Web-based interface built with Streamlit for user-friendly interaction.

## Technologies Used
- **Python Libraries**: Pandas, Seaborn, Matplotlib, TensorFlow, Scikit-learn
- **Clustering Models**: K-means, DBSCAN, Agglomerative Clustering
- **Visualization**: Seaborn and Matplotlib for plotting distributions within clusters
- **Web Framework**: Streamlit for the front-end interface

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityamatale/customer-segmentation-profiling.git
   cd customer-segmentation-profiling
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the app:**
   ```bash
   streamlit run app.py

## Usage
Launch the Streamlit app as described above. Choose the input method:

- **Manual Input**: Enter individual customer details.
- **CSV Upload**: Upload a CSV file containing customer data.

Select a clustering model (K-means, Agglomerative, or DBSCAN) and click "Submit" to view segmentation results.

## Model Accuracies
- **K-means**: 0.9292
- **DBSCAN**: 0.9847
- **Agglomerative Clustering**: 0.9549

## Clustering Implementation
This project implements clustering techniques, including K-means, DBSCAN, and Agglomerative Clustering. The implementation follows these steps:

### Data Preprocessing:
- Load the dataset and handle missing values appropriately.
- Drop unnecessary columns that do not contribute to the clustering process.

### Data Visualization:
- Generate visualizations (e.g., KDE plots, correlation heatmaps) to understand the data distribution and relationships.

### Scaling and Dimensionality Reduction:
- Scale the data using `StandardScaler` to ensure all features contribute equally to the distance calculations.
- Apply dimensionality reduction techniques (e.g., PCA) if necessary, to enhance visualization and reduce computational complexity.

### Clustering:
1. **K-Means Clustering**:
   - Determine the optimal number of clusters using the elbow method based on inertia.
   - Fit the K-means model and assign clusters to each data point.

2. **DBSCAN**:
   - Set appropriate parameters (e.g., `eps` and `min_samples`) to define the density threshold.
   - Fit the DBSCAN model to identify clusters and noise points.

3. **Agglomerative Clustering**:
   - Choose a linkage criterion (e.g., single, complete, average) and the number of clusters.
   - Fit the Agglomerative model to create a hierarchy of clusters.

### Visualization of Clusters:
- Visualize the resulting clusters using scatter plots or other relevant visualizations to illustrate the distribution and characteristics of each cluster.

### Model Persistence:
- Save the trained clustering models (K-means, DBSCAN, Agglomerative) using `joblib` or another serialization method.

### Classification (if applicable):
- Split the dataset into training and testing sets if predicting cluster labels.
- Train a classification model (e.g., Decision Tree Classifier) to predict the cluster labels based on features.

### Model Evaluation:
- Evaluate the performance of the classification model using metrics such as confusion matrix and classification report.
- Save the final classification model using `pickle`.
