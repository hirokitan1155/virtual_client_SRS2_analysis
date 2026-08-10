import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

# ==========================
# Load data
# ==========================

X = pd.read_csv(
    "nlp_features.csv"
)

outcomes = pd.read_csv(
    "outcomes.csv"
)

y = outcomes["K6_total"]  # SRS2_total or K6_total


# ==========================
# Models
# ==========================

models = {

    "Random Forest":
        RandomForestRegressor(
            n_estimators=1000,
            random_state=42
        ),

    "Ridge":
        Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=0.1))
        ]),

    "Elastic Net":
        Pipeline([
            ("scaler", StandardScaler()),
            ("elastic", ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=10000,
                random_state=42
            ))
        ]),

    "XGBoost":
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            objective="reg:squarederror"
        )
}


# ==========================
# Leave-One-Out CV
# ==========================

loo = LeaveOneOut()

results = []

for name, model in models.items():

    print("\n====================")
    print(name)
    print("====================")

    pred = cross_val_predict(
        model,
        X,
        y,
        cv=loo
    )

    rho, p = spearmanr(
        y,
        pred
    )

    mae = mean_absolute_error(
        y,
        pred
    )

    r2 = r2_score(
        y,
        pred
    )

    print("Spearman rho =", rho)
    print("p =", p)
    print("MAE =", mae)
    print("R2 =", r2)

    results.append({
        "Model": name,
        "Spearman_rho": rho,
        "p": p,
        "MAE": mae,
        "R2": r2
    })


# ==========================
# Elastic Net coefficients
# ==========================

print("\n====================")
print("Elastic Net Coefficients")
print("====================")

elastic_model = models["Elastic Net"]

coef_list = []

for train_idx, test_idx in loo.split(X):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    # Fit Elastic Net on training data
    elastic_model.fit(
        X_train,
        y_train
    )

    # Extract coefficients
    coef = elastic_model.named_steps["elastic"].coef_

    coef_list.append(coef)


# Convert coefficients to DataFrame

coef_df = pd.DataFrame(
    coef_list,
    columns=X.columns
)


# ==========================
# Summarize coefficients
# ==========================

coef_summary = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Weight": coef_df.mean(),
    "SD": coef_df.std(),
    "Mean_Abs_Weight": coef_df.abs().mean(),
    "Nonzero_Count": (coef_df != 0).sum(),
    "Nonzero_Ratio": (coef_df != 0).mean()
})

# Sort by absolute importance

coef_summary = coef_summary.sort_values(
    "Mean_Abs_Weight",
    ascending=False
)


print("\n===== Elastic Net Coefficients =====")
print(coef_summary)


# ==========================
# Save coefficients
# ==========================

coef_summary.to_csv(
    "elastic_net_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================
# Fit final Elastic Net
# using all data
# ==========================

final_elastic = models["Elastic Net"]

final_elastic.fit(
    X,
    y
)

final_coef = final_elastic.named_steps["elastic"].coef_

final_coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Final_Weight": final_coef
})

final_coef_df["Abs_Final_Weight"] = (
    final_coef_df["Final_Weight"].abs()
)

final_coef_df = final_coef_df.sort_values(
    "Abs_Final_Weight",
    ascending=False
)


print("\n===== Final Elastic Net Coefficients =====")
print(final_coef_df)


final_coef_df.to_csv(
    "elastic_net_final_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================
# Save results
# ==========================

results_df = pd.DataFrame(results)

print("\n===== Model Comparison =====")
print(results_df)

results_df.to_csv(
    "prediction_results.csv",
    index=False,
    encoding="utf-8-sig"
)

