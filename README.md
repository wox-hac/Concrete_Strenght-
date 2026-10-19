# AI-Based Concrete Compressive Strength Predictor

Beginner-friendly student project using a public dataset from the UCI Machine Learning Repository.

## Run on your laptop

1. Install Python 3.10 or newer from https://www.python.org/downloads/
   - On Windows, tick **Add Python to PATH** during installation.
2. Open Command Prompt (Windows) or Terminal (Mac/Linux).
3. Change directory to this project folder. Example on Windows:
   `cd Downloads\concrete_strength_ai_project`
4. Install dependencies:
   `python -m pip install -r requirements.txt`
5. Run the app:
   `python -m streamlit run app.py`
6. A browser tab should open. If not, copy the local URL printed in the terminal into your browser.

## Dataset

The app retrieves the Concrete Compressive Strength dataset from UCI using `ucimlrepo` (dataset ID 165). Internet access is required on first load.

Official dataset page:
https://archive.ics.uci.edu/dataset/165/concrete%2Bcompressive%2Bstrength

The dataset contains 1,030 records, eight input variables and one compressive-strength target.

## What to explain

- Inputs: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age.
- Target: measured concrete compressive strength in MPa.
- Model: Random Forest regression.
- Evaluation: an 80/20 train-test split; report the actual MAE, RMSE and R² shown by your app.
- Limitation: educational prediction only; not a substitute for laboratory tests, mix design approval, or structural engineering judgment.

## Troubleshooting

- If `python` is not recognized, try `py` instead of `python` on Windows.
- If installation fails, run `python -m pip install --upgrade pip`, then repeat the requirements command.
- If the dataset cannot be loaded, check your internet connection and try again.
- Keep the terminal open while the app runs. Press Ctrl+C in the terminal to stop it.
