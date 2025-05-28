import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Đọc dữ liệu train
train_data = pd.read_csv(r'D:\Do_an_mon_hoc\train.csv')

# Tách đặc trưng và giá nhà
X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice']

# Kiểm tra và chuyển đổi kiểu dữ liệu
for col in X_train.columns:
    # Nếu cột là số (int, float), chuyển thành float và xử lý giá trị thiếu
    if X_train[col].dtype in ['int64', 'float64']:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')  # Chuyển chuỗi không hợp lệ thành NaN
        X_train[col] = X_train[col].fillna(X_train[col].mean())  # Điền giá trị trung bình
    # Nếu cột là chuỗi (object), để nguyên và sẽ mã hóa sau
    else:
        X_train[col] = X_train[col].fillna(X_train[col].mode()[0])  # Điền giá trị phổ biến nhất cho cột chuỗi

# Mã hóa biến phân loại (cột chuỗi)
X_train = pd.get_dummies(X_train)

# Huấn luyện mô hình Linear Regression
print("Đào tạo mô hình Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Huấn luyện mô hình Random Forest
print("Đào tạo mô hình Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Lưu mô hình đã huấn luyện
joblib.dump(lr_model, 'linear_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Mô hình đã được lưu: linear_regression_model.pkl và random_forest_model.pkl")

# Lưu X_train columns để đảm bảo test có cùng cột
joblib.dump(X_train.columns, 'train_columns.pkl')
print("Cột đặc trưng đã được lưu: train_columns.pkl")