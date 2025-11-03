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
    st.header("劣化予測シミュレーション")
    st.write("シミュレーションする条件を設定してください")

    list_selected_area: list[str] = st.multiselect(
            "対象を選択",  ["test_0", "test_1", "test_2"],
            )
    list_selected_area: list[str] = st.multiselect(
            "IDを選択", ["ID_0", "ID_1", "ID_2"],
            )
    list_selected_area: list[str] = st.multiselect(
            "シーケンスを選択", ["シーケンス_0", "シーケンス_1", "シーケンス_2"]
            )
    # -------------------------------
    # シーケンス登録ボタン
    # -------------------------------
    st.divider()
    register_btn = st.button("📈 シミュレーション実行", type="primary")

    if register_btn:
        with st.spinner("シミュレーション実行中..."):
            # 5秒間処理中に見せる
            for i in range(5):
                time.sleep(1)
        st.success("✅ シミュレーション実行完了")


if __name__ == "__main__":
    main()
