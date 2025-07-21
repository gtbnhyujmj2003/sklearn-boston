# --- 1. 載入資料與初步 EDA ---
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['TARGET'] = boston.target
display(df.head())

# --- 2. 分割訓練集與測試集 ---
from sklearn.model_selection import train_test_split
X = df.drop("TARGET", axis=1)
y = df["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 一般 HistGradientBoostingRegressor 模型 ---
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"[HistGBR] MAE: {mean_absolute_error(y_test, y_pred):.3f}  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}  R2: {r2_score(y_test, y_pred):.3f}")

# --- 4. 用 AutoML (TPOT) 找最佳模型 ---
!pip install tpot -q
from tpot import TPOTRegressor

tpot = TPOTRegressor(verbosity=2, generations=5, population_size=30, random_state=42, disable_update_check=True)
tpot.fit(X_train, y_train)
tpot_pred = tpot.predict(X_test)
print(f"[TPOT-AutoML] MAE: {mean_absolute_error(y_test, tpot_pred):.3f}  RMSE: {np.sqrt(mean_squared_error(y_test, tpot_pred)):.3f}  R2: {r2_score(y_test, tpot_pred):.3f}")

# 輸出 TPOT 找到的最佳流程（自動推薦 pipeline）
print("\n[TPOT Pipeline]\n", tpot.fitted_pipeline_)

# --- 5. 進階：TPOT pipeline 匯出成 .py，給你作為後續開發基底 ---
tpot.export('tpot_boston_best_pipeline.py')
print("已匯出 TPOT pipeline 到 tpot_boston_best_pipeline.py")
