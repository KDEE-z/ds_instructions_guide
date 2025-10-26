from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd


class Config(BaseModel):
    """
    実験設定やデータパスを管理するクラス
    """

    # __file__ はこのファイル(config.py)の場所を指すので、
    # そこから2階層上の"data"ディレクトリを指定する
    data_dir: Path = Field(
        Path(__file__).resolve().parents[2] / "data",
        description="データディレクトリへのパス"
    )

    train_file: str = Field("taxi_dataset.csv", description="学習データのファイル名")
    test_file: str = Field("taxi_dataset_for_upload.csv", description="テストデータのファイル名")
    random_seed: int = Field(42, description="乱数シード値")

    @property
    def train_path(self) -> Path:
        """学習データのフルパス"""
        return self.data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        """テストデータのフルパス"""
        return self.data_dir / self.test_file

    def load_train_data(self) -> pd.DataFrame:
        """学習用データを読み込む"""
        df = pd.read_csv(self.train_path)
        print(f"✅ Loaded training data: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def load_test_data(self) -> pd.DataFrame:
        """テスト用データを読み込む"""
        df = pd.read_csv(self.test_path)
        print(f"✅ Loaded test data: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
