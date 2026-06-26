import pandas as pd
import xgboost as xgb
import pickle

print("Start training XGBoost model...")

# 1. 读取我们提取的画像数据
df = pd.read_parquet('driver_profiles.parquet')

# 2. 构造特征 (X) 和 预测目标 (y)
# 我们预测的目标是：每次行程的平均能耗 (total_energy_proxy / total_trips)
df['energy_per_trip'] = df['total_energy_proxy'] / df['total_trips']

# 选取影响能耗的特征：平均速度、急加速频率、急刹车频率、外部气温
features = ['avg_speed_kmh', 'rapid_accel_per_trip', 'harsh_brake_per_trip', 'avg_out_temp']
X = df[features].fillna(0)
y = df['energy_per_trip'].fillna(0)

# 3. 训练 XGBoost 回归模型
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X, y)
print("Model training completed!")

# 4. 导出模型文件 (离线训练，在线推理的精髓)
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to xgb_model.pkl")