# Airline Passengers LSTM Forecasting with Hyperparameter Tuning

This project implements a Long Short-Term Memory (LSTM) neural network to forecast airline passenger counts. It includes **Random Search** and **Bayesian Optimization** for hyperparameter tuning, with automatic logging of all trials to CSV for reproducibility and analysis.

---

## Features

- **Automatic Hyperparameter Tuning:**

  - Supports both **Random Search** and **Bayesian Optimization** to optimize LSTM architecture and training parameters.
  - Finds the best combination of:
    - Number of LSTM units
    - Batch size
    - Number of epochs
    - Look-back window
    - Optimizer type
    - Learning rate

- **Data Preprocessing:**

  - Normalizes the dataset using MinMaxScaler.
  - Splits data into training and test sets while maintaining sequential order.

- **Flexible LSTM Model:**

  - Easily adjustable number of LSTM units and input shape.
  - Supports multiple optimizers with configurable learning rates.

- **Automatic Logging:**

  - Random Search and Bayesian Optimization trials are logged to CSV (`random_search_results.csv` or `bayesian_trials.csv`) for easy analysis.

- **Visualization:**

  - Plots original dataset alongside train and test predictions.
  - Clearly shows LSTM model performance.

- **Reproducibility:**
  - Sets TensorFlow random seed for consistent results.
  - Fully self-contained notebook for direct execution.

---

## Installation

1. **Clone the Repository:**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. **Create a Virtual Environment (Optional but Recommended):**

```bash
python -m venv venv
venv\Scripts\activate # On Mac: source venv/bin/activate
```

3. **Install Required Packages:**

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Open the Notebook**

   - For **Random Search tuning**, open `Task3-agentic-LSTM-enhanced.ipynb`.
   - For **Bayesian Optimization tuning**, open `Task3-agentic-LSTM-Bayesian-Logging.ipynb`.
   - Use **Jupyter Notebook** or **Google Colab**.

2. **Run Cells Sequentially**

   - Execute all cells in order. The notebook will automatically:
     - Load and preprocess the dataset.
     - Normalize and split the data into training and testing sets.
     - Perform hyperparameter tuning (Random Search or Bayesian Optimization) to find the optimal LSTM configuration.
     - Evaluate RMSE for each trial and log results in a CSV file:
       - `random_search_results.csv` for Random Search
       - `bayesian_trials.csv` for Bayesian Optimization
     - Retrain the best LSTM model on the full training set.
     - Plot predictions against the original passenger data.

3. **Adjust Optional Settings**

   - Modify the number of search trials, epochs, or hyperparameter ranges directly in the notebook cells if needed.

4. **View Results**
   - After running the notebook, you will see:
     - Original passenger counts.
     - LSTM predictions for both training and test sets.
     - Best hyperparameters along with their RMSE.
     - A CSV file logging all trial hyperparameters and corresponding RMSE values for further analysis.

---
