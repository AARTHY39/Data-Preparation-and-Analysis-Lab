import pandas as pd
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    category: str
    price: float

products = [
    Product("Laptop", "Electronics", 1000),
    Product("T-shirt", "Clothing", 20),
    Product("Book", "Books", 15),
    Product("Headphones", "Electronics", 100),
    Product("Jeans", "Clothing", 50),
    Product("Smartphone", "Electronics", 800),
    Product("Sunglasses", "Accessories", 30),
    Product("Watch", "Accessories", 50),
    Product("Shoes", "Footwear", 80),
]

dates = ["2023-05-01", "2023-05-02", "2023-05-03"]
countries = ["USA", "UK", "Germany"]

data = []

for product in products:
    for date in dates:
        for country in countries:
            data.append({
                "Category": product.category,
                "Product": product.name,
                "Date": date,
                "Country": country,
                "Price": product.price
            })

df = pd.DataFrame(data)

data_cube = pd.pivot_table(
    df,
    values="Price",
    index=["Category", "Product"],
    columns=["Date", "Country"],
    aggfunc="first"
)

print("3D Data Cube (Product x Date x Country):\n")
print(data_cube)