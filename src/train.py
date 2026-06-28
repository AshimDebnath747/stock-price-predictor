import os
import joblib
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from evaluate import evaluate

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA20",
    "Return",
    "Volatility",
]


def load_data(ticker):
    data = yf.download(
        ticker,
        start="2020-01-01",
        end="2026-01-01",
        auto_adjust=True,
    )

    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()

    data["Return"] = data["Close"].pct_change()

    data["Volatility"] = (
        data["High"] - data["Low"]
    ) / data["Close"]

    data["Target"] = data["Close"].shift(-1)

    return data.dropna()


def train_model(ticker):

    print(f"traning {ticker} the model!")
    data = load_data(ticker)

    X = data[FEATURES]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #predict
    predictions = model.predict(X_test)

    # Evaluate
    evaluate(y_test, predictions)


    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{ticker}_model.pkl")

    print(f"{ticker} model trained successfully!")


if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, MSFT, TSLA): ").upper()
    train_model(ticker)