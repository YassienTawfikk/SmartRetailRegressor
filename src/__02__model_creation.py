from sklearn.ensemble import RandomForestRegressor
import joblib
from src.__00__paths import model_dir


def return_rf_model(max_depth=38, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=170,
                    n_jobs=-1, random_state=42):
    """
    Hyperparameters based on RandomSearch in `02_model_baseline_rf.ipynb`
    """
    rf_model = RandomForestRegressor(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        n_jobs=-n_jobs,
        random_state=random_state
    )

    return rf_model


def save_model(model, file_name=model_dir / "random_forest_model.joblib"):
    joblib.dump(model, file_name)
    print(f"✔️ Model Saved at {'/'.join(file_name.parts[-2:])}")
