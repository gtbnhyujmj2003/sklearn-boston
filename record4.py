# --- 1. 載入資料與自動 EDA ---
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['TARGET'] = boston.target

# 自動 EDA（如果沒裝，先 pip install pandas-profiling）
try:
    from pandas_profiling import ProfileReport
    ProfileReport(df, title="Boston Data Report", explorative=True)
except:
    print("未安裝 pandas-profiling，請使用 `!pip install pandas-profiling` 安裝")

# --- 2. 訓練/測試集切分 ---
from sklearn.model_selection import train_test_split
X = df.drop("TARGET", axis=1)
y = df["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 用現代的回歸模型 HistGradientBoostingRegressor ---
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 4. 多指標評估（MAE, RMSE, R2） ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# --- 5. 特徵重要度視覺化 ---
import matplotlib.pyplot as plt

importances = model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,4))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), feat_names[indices], rotation=90)
plt.show()

# --- 6. 用 SHAP 解釋預測原因 ---
!pip install shap -q
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
