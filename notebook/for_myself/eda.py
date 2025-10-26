# eda.py
import pandas as pd

def simple_eda(df: pd.DataFrame, n_rows: int = 5) -> None:
    """データの簡易的なEDAを行う"""
    print("\n🔹 Data Overview")
    print(df.head(n_rows))

    print("\n🔹 Missing Values")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n🔹 Data Types")
    print(df.dtypes.value_counts())

    print("\n🔹 Numeric Summary")
    print(df.describe().T)
