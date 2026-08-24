import pandas as pd
import gradio as gr
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

RANDOM_STATE = 42


def load_and_clean(path="cardekho.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")
    km_cap = df["km_driven"].quantile(0.99)
    df["km_driven"] = df["km_driven"].clip(upper=km_cap)
    df["brand"] = df["name"].str.split().str[0]
    df["car_age"] = 2024 - df["year"]
    return df.drop(columns=["name", "year"])


def train_model(df):
    X = df.drop(columns=["selling_price"])
    y = df["selling_price"]

    num_cols = ["km_driven", "mileage(km/ltr/kg)", "engine", "max_power", "seats", "car_age"]
    cat_cols = ["fuel", "seller_type", "transmission", "owner", "brand"]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    model = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    model.fit(X, y)
    return model


print("Loading data and training model (a few seconds)...")
df = load_and_clean()
model = train_model(df)
print("Model ready.")

brands = sorted(df["brand"].unique())
fuels = sorted(df["fuel"].unique())
seller_types = sorted(df["seller_type"].unique())
transmissions = sorted(df["transmission"].unique())
owners = sorted(df["owner"].unique())


def predict_price(brand, car_age, km_driven, fuel, seller_type, transmission,
                   owner, mileage, engine, max_power, seats):
    row = pd.DataFrame([{
        "km_driven": km_driven,
        "mileage(km/ltr/kg)": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
        "car_age": car_age,
    }])
    price = model.predict(row)[0]
    return f"$ {price:,.0f}"


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(brands, value=brands[0], label="Brand"),
        gr.Slider(0, 30, value=5, step=1, label="Car Age (years)"),
        gr.Number(value=50000, label="Km Driven"),
        gr.Dropdown(fuels, value=fuels[0], label="Fuel"),
        gr.Dropdown(seller_types, value=seller_types[0], label="Seller Type"),
        gr.Dropdown(transmissions, value=transmissions[0], label="Transmission"),
        gr.Dropdown(owners, value=owners[0], label="Owner"),
        gr.Number(value=18.0, label="Mileage (km/ltr/kg)"),
        gr.Number(value=1200.0, label="Engine (cc)"),
        gr.Number(value=85.0, label="Max Power (bhp)"),
        gr.Number(value=5, label="Seats"),
    ],
    outputs=gr.Textbox(label="Predicted Selling Price"),
    title="🚗 Car Selling Price Predictor",
    description="Predicts a used car's selling price with a tuned XGBoost model trained on the CarDekho dataset.",
)


demo.launch(
    server_name="127.0.0.1",
    server_port=7861,
    share=True
)