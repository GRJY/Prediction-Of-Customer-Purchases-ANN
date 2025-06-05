# Walmart Customer Purchase Prediction

## Overview
This project develops an Artificial Neural Network (ANN) model to predict customer purchasing behavior using Walmart's e-commerce sales data. The model classifies whether a customer's purchase amount exceeds $8,000 based on demographic and product-related features. The project achieves an accuracy of 80.52% and an AUC-ROC score of 0.81, providing actionable insights for e-commerce marketing strategies.

## Features
- **Data Preprocessing**: Handles missing data with mean/mode imputation, applies one-hot encoding for categorical variables, and normalizes features using Min-Max scaling.
- **ANN Model**: Implements a two-layer ANN with 128 and 64 neurons, using ReLU and sigmoid activation functions, optimized with Adam optimizer and binary cross-entropy loss.
- **Performance Evaluation**: Evaluates model performance with accuracy, loss curves, AUC-ROC, confusion matrix, and feature importance analysis using Permutation Importance.
- **Key Insights**: Identifies product categories as the most influential predictors of purchase behavior.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Keras, TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Dataset**: Walmart sales dataset from Kaggle (550,068 rows, 10 columns)
- **Development Environment**: Visual Studio Code, MacBook M1 (16 GB RAM)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/walmart-purchase-prediction.git
   cd walmart-purchase-prediction
   ```
2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the Dataset**: Obtain the Walmart sales dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset) and place it in the `data/` folder as `walmart_sales.csv`.

## Usage
1. Ensure the dataset (`walmart_sales.csv`) is in the `data/` folder.
2. Run the main script to preprocess data, train the model, and evaluate performance:
   ```bash
   python main.py
   ```
3. Outputs include:
   - Model accuracy and loss curves (`plots/loss_accuracy.png`)
   - Confusion matrix (`plots/confusion_matrix.png`)
   - Feature importance rankings (`plots/feature_importance.png`)

## Project Structure
```
walmart-purchase-prediction/
├── data/                    # Dataset folder
├── plots/                   # Generated plots (loss, accuracy, etc.)
├── main.py                  # Main script for preprocessing, training, and evaluation
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## Results
- **Accuracy**: 80.52%
- **AUC-ROC**: 0.81
- **Precision**: 86.9%
- **Key Finding**: Product categories significantly influence purchase predictions, enabling targeted marketing strategies.

## Future Improvements
- Experiment with advanced models like LSTM or Transformer architectures.
- Incorporate additional datasets to enhance model robustness.
- Develop a web interface to visualize predictions and insights.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
- Inspired by research from Elma (2014), Sharma et al. (2021), and Kumar et al. (2023)