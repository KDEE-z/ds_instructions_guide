# main.py
from config import Config
from eda import simple_eda

def main():
    cfg = Config()
    cfg.show_summary()
    
    train_df = cfg.load_train_data()
    test_df = cfg.load_test_data()
    print(train_df.head(3))
    print()
    print(test_df.head(3))
    simple_eda(train_df)

if __name__ == "__main__":
    main()
    print("\nit's done !")
