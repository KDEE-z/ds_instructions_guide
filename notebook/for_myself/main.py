# main.py
from config import Config
from eda import simple_eda

def main():
    cfg = Config()
    train_df = cfg.load_train_data()
    test_df = cfg.load_test_data()
    print(train_df.head())
    simple_eda(train_df)

if __name__ == "__main__":
    main()
    print("\nit's done !")
