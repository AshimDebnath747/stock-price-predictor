import joblib
from train import load_data, FEATURES

ticker = input("Enter stock ticker (e.g., AAPL, MSFT, TSLA): ").upper()

model = joblib.load("models/AAPL_model.pkl")

data = load_data(ticker)

latest = data[FEATURES].iloc[-1:]

prediction = model.predict(latest)

print(f"Predicted next close: ${prediction[0]:.2f}")