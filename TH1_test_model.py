import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split

# Hàm tính "accuracy" tùy chỉnh (dự đoán trong khoảng ±10% giá thực tế)
def custom_accuracy(y_true, y_pred, tolerance=0.1):
    return np.mean(np.abs(y_pred - y_true) / y_true <= tolerance) * 100

# Đọc dữ liệu train để tạo tập validation
train_data = pd.read_csv('train.csv')
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

# Xử lý dữ liệu train
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].mean())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])
X = pd.get_dummies(X)

# Chia tập train thành train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tải mô hình
lr_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Dự đoán trên tập validation
y_pred_lr = lr_model.predict(X_val)
y_pred_rf = rf_model.predict(X_val)

# Đánh giá mô hình
print("Kết quả trên tập validation:")
for model_name, y_pred in [("Linear Regression", y_pred_lr), ("Random Forest", y_pred_rf)]:
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    acc = custom_accuracy(y_val, y_pred)
    print(f"\n{model_name}:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"Custom Accuracy (±10%): {acc:.2f}%")

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))

# Biểu đồ cho Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_lr, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('Linear Regression')


# Biểu đồ cho Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_rf, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('Random Forest')

plt.tight_layout()
plt.savefig('prediction_plot.png')
print("\nBiểu đồ đã được lưu vào 'prediction_plot.png'")

# Dự đoán trên tập test
test_data = pd.read_csv('test.csv')
X_test = test_data

# Xử lý dữ liệu test
for col in X_test.columns:
    if X_test[col].dtype in ['int64', 'float64']:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_test[col] = X_test[col].fillna(X_test[col].mean())
    else:
        X_test[col] = X_test[col].fillna(X_test[col].mode()[0])
X_test = pd.get_dummies(X_test)

# Đồng bộ cột
train_columns = joblib.load('train_columns.pkl')
X_test = X_test.reindex(columns=train_columns, fill_value=0)

# Dự đoán
print("\nDự đoán trên tập test...")
y_pred_lr_test = lr_model.predict(X_test)
y_pred_rf_test = rf_model.predict(X_test)

# Lưu dự đoán
predictions = pd.DataFrame({
    'Predicted_Price_LR': y_pred_lr_test,
    'Predicted_Price_RF': y_pred_rf_test
})
predictions.to_csv('predictions.csv', index=False)
print("Dự đoán đã được lưu vào 'predictions.csv'")