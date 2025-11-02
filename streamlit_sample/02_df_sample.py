import datetime
import pandas as pd
import streamlit as st

df = pd.DataFrame(
    {
        'area': ["Sendai", "Nara", "Shiga", "Kyoto", "Shizuoka"],
        'population': [100, 50, 20, 200, 60],
        'date': [
            datetime.date(2025,11,2),
            datetime.date(2025,11,2),
            datetime.date(2025,11,2),
            datetime.date(2025,11,2),
            datetime.date(2025,11,2),
        ],
        'StarCity': [1,0,0,1,0],
    }
)

st.dataframe(df)