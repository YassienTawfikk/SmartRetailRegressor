# SmartRetailRegressor

> Sales Forecasting for Walmart Stores Using Regression Models (Random Forest & XGBoost)

<p align='center'>
<img width="800" alt="SmartRetailRegressor Poster" src="https://github.com/user-attachments/assets/b9e05c58-8fd7-4e47-b21e-1e19aa52c5fe" />  
</p>

---

## Problem Statement

Retail giants like Walmart need accurate sales forecasting to:

* Plan inventory efficiently
* Anticipate seasonal spikes
* Set realistic revenue expectations
* Optimize staff, logistics, and marketing

This project aims to predict future weekly sales across Walmart stores using historical data and machine learning regression models.

---

## Dataset Overview

The dataset comes from Kaggle: Walmart Sales Forecasting, and includes:

* `train.csv`: Weekly sales for each store/department/date
* `features.csv`: Contains temperature, fuel price, holidays, etc.
* `stores.csv`: Store type (A/B/C), and size

---

## Features

| Feature                | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `Dept`                 | Department number within a store                    |
| `isHoliday`            | Whether the date is a holiday (True/False)          |
| `store`                | Store number ID                                     |
| `type`                 | Store type (A, B, or C)                             |
| `size`                 | Size of the store (square footage)                  |
| `time_index`           | Index of the sample's order in time                 |
| `woy_sin`, `woy_cos`   | Week of year as sine/cosine (captures periodicity)  |
| `mon_sin`, `mon_cos`   | Month as sine/cosine (captures monthly seasonality) |
| `is_month_start`       | Flag if the date is the start of a month            |
| `is_month_end`         | Flag if the date is the end of a month              |
| `is_quarter_end`       | Flag if the date is at quarter end                  |
| `is_year_end`          | Flag if the date is at year end                     |
| `week_of_month`        | Week number within the month                        |
| `dist_thanksgiving_wk` | Distance in weeks to Thanksgiving                   |
| `dist_black_friday_wk` | Distance in weeks to Black Friday                   |
| `dist_xmas_peak_wk`    | Distance in weeks to Christmas peak                 |
| `dist_easter_wk`       | Distance in weeks to Easter                         |
| `dist_memorial_day_wk` | Distance in weeks to Memorial Day                   |
| `dist_july4_wk`        | Distance in weeks to Independence Day               |
| `dist_labor_day_wk`    | Distance in weeks to Labor Day                      |
| `dist_super_bowl_wk`   | Distance in weeks to Super Bowl                     |
| `is_black_friday_wk`   | Flag if it’s Black Friday week                      |
| `is_thanksgiving_wk`   | Flag if it’s Thanksgiving week                      |
| `is_xmas_peak_wk`      | Flag if it’s Christmas week                         |
| `is_back_to_school`    | Flag if it’s the back-to-school season              |

### Label

* `weekly_sales`: Total sales of that store-department-week

---

## Data Engineering

We combined `train.csv`, `features.csv`, and `stores.csv` on the following keys:

* `Store`
* `Date`

Then we performed extensive feature engineering:

### Categorical & Structural Features

* `Store`, `Dept`, `Type` (One-hot encoded)
* `Size`

### Date-Based Features

From `Date`, we extracted:

* `month`, `week_of_year`, `week_of_month`
* `quarter`
* `time_index` (for modeling trend)
* Sinusoidal encodings: `woy_sin/cos`, `mon_sin/cos` for periodicity
* Position flags: start/end of month, quarter, and year

### Holiday Features

We generated:

* Binary flags for key holidays (`is_thanksgiving`, `is_black_friday`, etc.)
* Distance (in weeks) from holiday peaks (e.g. `dist_xmas_peak_wk`)

### Domain Observations

* Holiday weeks, especially around Christmas, Thanksgiving, and Black Friday, show peak sales performance.
* Christmas sales typically occur during the 51st week, not in the final days of December as marked in the dataset.
* Post-holiday months like January see a drop in sales, likely due to reduced consumer spending after November and December.
* Features such as CPI, temperature, unemployment rate, and fuel price were excluded as they exhibited no significant patterns or correlation with weekly sales.

### Final Step

* Encoded booleans, dropped raw `Date`
* Aligned dummy features between train/test sets

---

## Model Comparison: Random Forest vs. XGBoost

| Metric   | Random Forest | XGBoost |
| -------- | ------------- | ------- |
| MAE      | 1642.07       | 2267.61 |
| WMAE     | 1676.58       | 2300.03 |
| RMSE     | 3335.64       | 3978.45 |
| R² Score | 97.69%        | 96.72%  |

### Performance Insights

* The **Random Forest** model underwent **exhaustive hyperparameter tuning** with a wide search space, leading to a substantial boost in accuracy.
* As a result, Random Forest achieved the **highest R² score** (97.69%) and lower error metrics (MAE, WMAE, RMSE) compared to XGBoost.
* This tuning allowed Random Forest to generalize better while maintaining strong accuracy on unseen data.

### Key Visuals

#### Actual vs Predicted

| Random Forest                                                                                             | XGBoost                                                                                                        |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| ![RF\_actual\_predicted](https://github.com/user-attachments/assets/cd0dad01-d8f0-4185-8d29-659600ea538c) | ![XGBoost\_actual\_predicted](https://github.com/user-attachments/assets/20be3609-9293-4d81-a35b-11ed91b2d97a) |

#### Feature Importance

| Random Forest                                                                                               | XGBoost                                                                                                          |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ![RF\_feature\_importance](https://github.com/user-attachments/assets/b6fc6053-8871-469d-bba3-6bd6e9b7baa2) | ![XGBoost\_feature\_importance](https://github.com/user-attachments/assets/28b6dd64-8246-48dc-8f6a-f20375ea2cb4) |

#### Tree Snapshot

* Random Forest (depth=3 view):
  ![RF\_tree\_visualization](https://github.com/user-attachments/assets/1c205f1d-ac1a-4f89-9bb1-80b0b45efe10)

---

## Conclusion

* Both models captured seasonal and structural sales patterns effectively.
* After exhaustive hyperparameter tuning, **Random Forest** outperformed XGBoost across all metrics, including R² score and error measures.
* Feature engineering around holidays and date encodings was crucial for the performance of both models.
  
---

## Project Structure

```
SmartRetailRegressor/
├── data/              # Raw and processed datasets
├── notebooks/         # Jupyter notebooks for each stage
├── src/               # Scripts for paths, utils, model logic
├── outputs/           # Saved model (.joblib), figures, data
├── docs/              # Evaluation metrics
└── requirements.txt   # Full environment dependencies
```

---

## Submission

This project was developed as part of the **Elevvo AI Internship**, showcasing hands-on application of recommendation systems — from data curation to evaluation — with clear insights into model limitations and comparative strengths.

---

## Author

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br>
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>
