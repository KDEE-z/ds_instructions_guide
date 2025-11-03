import datetime
import time
from pathlib import Path

import pandas as pd
import pandera as pa
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandera.typing import DataFrame

from kri_simu.schema import KriDatasetSchema
from taxi_prediction.model import LGBModel
from taxi_prediction.process import postprocess, preprocess_for_infer


@st.cache_resource
def load_model(model_path: str | Path) -> LGBModel:
    """学習済みモデルを読み込む"""
    return LGBModel.load(model_path)


@st.cache_data
@pa.check_types
def inference_usecase(
    df: DataFrame[KriDatasetSchema],
    model_path: str | Path,
    predict_start_date: datetime.date,
) -> DataFrame[KriDatasetSchema]:
    """推論のワークフロー。前処理から予測までの一連の処理を行う"""
    df_processed = preprocess_for_infer(df)
    model = load_model(model_path)
    df_pred = model.predict(df_processed)
    return postprocess(df_pred, predict_date=predict_start_date)


def _filter_by_area(df: pd.DataFrame, list_selected_area: list[str]) -> pd.DataFrame:
    """ユーザーが入力したエリアでデータをフィルタリングする

    Note: list_selected_areaが空の場合はすべてのエリアを選択する
    """
    if len(list_selected_area) == 0:
        return df
    return df[df["area"].isin(list_selected_area)]


def _filter_by_display_period(df: pd.DataFrame, display_period: int) -> pd.DataFrame:
    """最新の日付から指定された表示期間（日数）分のデータを抽出する"""
    return df[df["date"] > df["date"].max() - pd.Timedelta(days=display_period)]


def _plot_prediction(df: pd.DataFrame) -> go.Figure:
    """予測結果をグラフ化する"""
    fig = px.line(
        df, x="date", y="num_trip", color="area", markers=True, line_dash="label"
    )
    fig.update_layout(
        title="乗車数の推移",
        xaxis_title="日付",
        yaxis_title="乗車数",
        legend_title="エリア, ラベル",
    )
    return fig


# 元の df（ユーザー提示のもの）
DEFAULT_DF = pd.DataFrame(
    {
        "area": ["Sendai", "Nara", "Shiga", "Kyoto", "Shizuoka"],
        "population": [100, 50, 20, 200, 60],
        "date": [
            datetime.date(2025, 11, 2),
            datetime.date(2025, 11, 2),
            datetime.date(2025, 11, 2),
            datetime.date(2025, 11, 2),
            datetime.date(2025, 11, 2),
        ],
        "StarCity": [1, 0, 0, 1, 0],
    }
)


def ensure_step_column(df: pd.DataFrame) -> pd.DataFrame:
    """'step' カラムを 1 からの昇順で自動付番"""
    df = df.reset_index(drop=True)
    df["step"] = range(1, len(df) + 1)
    # カラム順を step を最初にする
    cols = ["step"] + [c for c in df.columns if c != "step"]
    return df[cols]


def main() -> None:
    st.header("シミュレーションシーケンス設定")
    st.write("表に任意の値を入力してください（`+` ボタンで行を追加します）")

    # 初期化
    if "df" not in st.session_state:
        st.session_state.df = ensure_step_column(DEFAULT_DF.copy())

    # -------------------------------
    # ＋ ボタンで空行追加
    # -------------------------------
    add_col, editor_col = st.columns([1, 10])
    with add_col:
        if st.button("+", key="add_row_button"):
            empty_row = {
                col: (
                    pd.NaT
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col])
                    else pd.NA
                )
                for col in st.session_state.df.columns
                if col != "step"  # step は自動で再計算するため除外
            }
            new_row_df = pd.DataFrame([empty_row])
            st.session_state.df = pd.concat(
                [st.session_state.df, new_row_df], ignore_index=True
            )
            st.session_state.df = ensure_step_column(st.session_state.df)
            st.experimental_rerun()

    # -------------------------------
    # 編集可能テーブル
    # -------------------------------
    area_options = ["Sendai", "Nara", "Shiga", "Kyoto", "Shizuoka", "Tokyo", "Osaka"]
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "area": st.column_config.SelectboxColumn(
                "area（地域）", options=area_options
            ),
        },
        key="data_editor",
    )

    # 編集結果を反映
    if edited_df is not None:
        st.session_state.df = ensure_step_column(edited_df)

    # -------------------------------
    # シーケンス登録ボタン
    # -------------------------------
    st.divider()
    register_btn = st.button("🗂️ シーケンス登録", type="primary")

    if register_btn:
        with st.spinner("シーケンス登録中..."):
            # 5秒間処理中に見せる
            for i in range(3):
                time.sleep(1)
        st.success("✅ シーケンス登録完了")

    st.caption(
        "セルを直接編集 → '+' ボタンで空行追加 → 'シーケンス登録' で登録。"
    )

    # ==================
    # ファイルアップロード
    # ==================
    st.header("シーケンスファイルのアップロード")
    uploaded_file = st.file_uploader(
        "シーケンスファイルをアップロードすることも可能です。", type="csv"
    )

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file, parse_dates=["date"])
        df_upload["area"] = df_upload["area"].astype("category")
        st.dataframe(df_upload)


if __name__ == "__main__":
    main()
