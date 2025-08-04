import pandas as pd
import numpy as np
from src.__00__paths import raw_data_dir, curated_data_dir, processed_data_dir, model_dir
from src.__01_data_setup import *
from src.__02__model_creation import *


def main():
    print(f"Creating Needed Directories...")

    print(f"⬇️ Downloading Data...")
    download_dataset()

    # Load Raw Datasets
    raw_train_df = load_dataset(raw_data_dir / "train.csv")
    raw_store_df = load_dataset(raw_data_dir / "stores.csv")

    print(f"⬇️ Processing Data...")
    # Declare Processed Datasets
    processed_data_df = add_date_features(raw_train_df)

    # Merge Store Features
    processed_data_df = merge_datasets(processed_data_df, raw_store_df, on="Store")

    # One-hot encode
    processed_data_df = pd.get_dummies(processed_data_df, columns=["Type"], prefix="Type")

    # Reorder Dataset Columns and Rows
    processed_data_df = reorder_data_frame(processed_data_df)

    # Split in order on Date 13 Apr 2012
    train_df, test_df = split_data(processed_data_df, onDate="2012-04-13")

    # Save Data
    save_data(processed_data_df, processed_data_dir / "processed_data.csv")
    save_data(train_df, curated_data_dir / "train.csv")
    save_data(test_df, curated_data_dir / "test.csv")

    # Declare features & Labels
    features = train_df.drop(columns=["Weekly_Sales"])
    labels = train_df["Weekly_Sales"]

    print(f"⬇️ Model Training...")
    rf_model = return_rf_model()
    rf_model.fit(features, labels)
    save_model(rf_model)


if __name__ == "__main__":
    main()
