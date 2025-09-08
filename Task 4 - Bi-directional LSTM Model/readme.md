# Airline Passengers **Bi-Directional LSTM** Forecasting with Random Search Hyperparameter Tuning

This project implements a **Bi-Directional Long Short-Term Memory (Bi-LSTM)** neural network to forecast airline passenger counts.  
It uses **Random Search** for hyperparameter tuning, with automatic logging of all trials to CSV for reproducibility and analysis.

---

## ✨ Features

- **Automatic Hyperparameter Tuning (Random Search):**

  - Optimizes Bi-LSTM architecture and training parameters.
  - Finds the best combination of:
    - Number of Bi-LSTM units
    - Batch size
    - Number of epochs
    - Look-back window
    - Optimizer type
    - Learning rate

- **Data Preprocessing:**

  - Normalizes the dataset using MinMaxScaler.
  - Splits data into training and test sets while maintaining sequential order.

- **Flexible Bi-LSTM Model:**

  - Uses **Bidirectional LSTM layers** to capture dependencies from both past and future contexts in the sequence.
  - Easily adjustable number of LSTM units and input shape.
  - Supports multiple optimizers with configurable learning rates.

- **Automatic Logging:**

  - All Random Search trials are logged to a CSV file (`random_search_results.csv`) for easy analysis.

- **Forecasting:**

  - Multi-step forecasting by iteratively predicting future passenger counts.
  - Forecast horizon is customizable (e.g., next 12 months).

- **Visualization:**

  - Plots historical data alongside model forecasts.
  - Clearly shows Bi-LSTM performance and future predictions.

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

## 🚀 Usage

1. **Open the Notebook**

   - Open `Task3-agentic-BiLSTM-RandomSearch.ipynb`.
   - Use **Jupyter Notebook** or **Google Colab**.

2. **Run Cells Sequentially**

   - The notebook will automatically:
     - Load and preprocess the dataset.
     - Normalize and split the data into training and testing sets.
     - Perform **Random Search** to find the optimal Bi-LSTM configuration.
     - Evaluate RMSE for each trial and log results in `random_search_results.csv`.
     - Retrain the best Bi-LSTM model on the full training set.
     - Forecast the next 12 months of passenger data.
     - Plot historical and forecasted values.

3. **Adjust Optional Settings**

   - Modify the number of search trials, epochs, or hyperparameter ranges directly in the notebook cells if needed.

4. **View Results**
   - After running the notebook, you will see:
     - Original passenger counts.
     - Bi-LSTM predictions for both training and test sets.
     - Next 12 months forecasted values.
     - Best hyperparameters along with their RMSE.
     - A CSV file (`random_search_results.csv`) logging all trial hyperparameters and corresponding RMSE values for further analysis.

---
