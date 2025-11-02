import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def _plot_prediction(df: pd.DataFrame) -> go.Figure:
    """データフレーム形式の予測結果をグラフ化する"""
    fig = px.line(
        df,
        x='date',
        y = 'num_trip',
        color = 'area',
        markers=True,
        line_dash='label',
    )
    fig.update_layout(
        title="乗客数の推移",
        xaxis_title="日付",
        yaxis_title="乗車数",
        legend_title="エリア, ラベル",
    )
    return fig

# 適当なデータフレーム
df = pd.DataFrame(
    {
        'area': [1,1,1,3,3,3],
        'date': [
            datetime.date(2019,12,1),
            datetime.date(2019,12,2),
            datetime.date(2019,12,3),
            datetime.date(2019,12,1),
            datetime.date(2019,12,2),
            datetime.date(2019,12,3),
        ],
        'num_trip':[100,200,300,400,500,600],
        'label': ["real","real","predict","real","real","predict"]
    }
)

fig = _plot_prediction(df)
st.plotly_chart(fig)