import pandas as pd
import streamlit as st

upload_file = st.file_uploader(
    "予測用ファイル（.csv）をアップロードしてください", type='csv'
)

if upload_file is not None:
    df_upload = pd.read_csv(upload_file, parse_dates=['date'])
    df_upload['area'] = df_upload['area'].astype('category')
    # アップロードされたファイルの内容を表示
    st.dataframe(df_upload)