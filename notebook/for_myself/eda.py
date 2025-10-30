# eda.py
import pandas as pd
import plotly.express as px
from config import CFG, Config

def simple_eda(df: pd.DataFrame, n_rows: int = 5) -> None:
    """データの簡易的なEDAを行う"""
    """
    データの基本的な構造と概要を表示する簡易EDA関数
    """
    print("===== データの概要（有効性） =====")
    print(df.info())
    print("\n===== 欠損値の数（完全性） =====")
    print(df.isnull().sum())
    print("\n===== カテゴリの分布確認（完全性、一貫性、妥当性） =====")
    print(df['area'].value_counts())
    print("\n===== 統計量確認（最新性、妥当性） =====")
    print(df.describe().T)
    print("\n===== データ中身（妥当性） =====")
    print(df.head(3))
    print(df.tail(3))
    print(df.sample(5, random_state=CFG.SEED))
    print("\n===== 重複確認（一意性） =====")
    print(df.duplicated().sum())
    print(df.duplicated(subset=['area', 'date']).sum())
    print("\n===== 分布確認（妥当性、一貫性） =====")
    fig = px.histogram(df['num_trip'], log_y=True, nbins=100)
    fig.update_layout(xaxis_title="タクシー利用回数")
    fig.show()
    print("\n===== 分布確認_年探知（妥当性、一貫性） =====")
    df['year'] = df['date'].dt.year
    
    fig = px.histogram(
        df,
        x='num_trip',
        facet_col='year',
        nbins = 100,
        log_y=True,
        labels = {'num_trip': "タクシー利用回数"}
        )
    # タイトル調整
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] + "年"))
    fig.show()
