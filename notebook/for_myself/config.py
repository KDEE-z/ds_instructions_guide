from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, field_validator, computed_field
import chardet

class CFG:
    SEED = 55

class Config(BaseModel):
    """
    実験設定やデータパスを管理するクラス
    """
    # __file__ はこのファイル(config.py)の場所を指すので、
    # そこから2階層上の"data"ディレクトリを指定する
    data_dir: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2] / "data" 
            if '__file__' in globals() else Path.cwd().parents[1] / 'data'
            ),
        description="データディレクトリへのパス"
        )

    train_file: str = Field("taxi_dataset.csv", description="学習データのファイル名")
    test_file: str = Field("taxi_dataset_for_upload.csv", description="テストデータのファイル名")
    random_seed: int = Field(CFG.SEED, description="乱数シード値")

    @computed_field
    @property
    def train_path(self) -> Path:
        """学習データのフルパス"""
        return self.data_dir / self.train_file

    @computed_field
    @property
    def test_path(self) -> Path:
        """テストデータのフルパス"""
        return self.data_dir / self.test_file

    # === バリデーション ===
    @field_validator('data_dir')
    @classmethod
    def validate_data_dir(cls, v :Path) -> Path:
        """存在確認(存在しなければ警告のみ"""
        if not v.exists():
            print(f"Warning: data_dir {v} が存在しません。")
        return v
    
    # === データロード ===    
    def load_train_data(self) -> pd.DataFrame:
        """学習用データを読み込む"""
        with open(self.train_path, "rb") as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']
            print(f"Detected encoding: {encoding}")
            
        df = pd.read_csv(
            self.train_path,
            dtype={'area': str, 'num_trip': int},
            parse_dates=['date'],
            )
        print(f"✅ Loaded training data: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def load_test_data(self) -> pd.DataFrame:
        """テスト用データを読み込む"""
        with open(self.test_path, "rb") as f:
            raw_data=f.read()
            encoding=chardet.detect(raw_data)['encoding']
            print(f"Detected encoding: {encoding}")
            
        df = pd.read_csv(
            self.test_path,
            dtype={'area':str, 'num_trip':str},
            parse_dates=['date'],
            )
        print(f"✅ Loaded test data: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    
    # === 設定表示 ===
    def show_summary(self):
        """設定内容を整形して表示"""
        print("=== Config Summary ===")
        print(f"📁 Data Dir : {self.data_dir}")
        print(f"📄 Train File : {self.train_file}")
        print(f"📄 Test File  : {self.test_file}")
        print(f"🎲 Random Seed : {self.random_seed}")
        print("=======================")
        return print("it's done.")