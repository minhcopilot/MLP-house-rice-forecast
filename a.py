import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.ensemble import RandomForestRegressor
# Load the data
x_train  = pd.read_csv('data/X_train.csv')
y_train  = pd.read_csv('data/Y_train.csv')
x_test  = pd.read_csv('data/X_test.csv')
# Check data shapes
print("X_train shape:", x_train.shape)#X_train shape: (255080, 21)
print("Y_train shape:", y_train.shape)#Y_train shape: (255080, 2)
print("X_test shape:", x_test.shape)#X_test shape: (63771, 20)
# Merge x_train and y_train
train = pd.merge(x_train, y_train, on='ID')
# Data preprocessing
train.drop('ID', axis=1, inplace=True)
if 'ID' in x_test.columns:
    x_test.drop('ID', axis=1, inplace=True)
# Process 'tradeTime'
train['tradeTime'] = pd.to_datetime(train['tradeTime'])
x_test['tradeTime'] = pd.to_datetime(x_test['tradeTime'])
train['tradeYear'] = train['tradeTime'].dt.year
train['tradeMonth'] = train['tradeTime'].dt.month
x_test['tradeYear'] = x_test['tradeTime'].dt.year
x_test['tradeMonth'] = x_test['tradeTime'].dt.month
train.drop('tradeTime', axis=1, inplace=True)
x_test.drop('tradeTime', axis=1, inplace=True)
# Process 'floor' column

def process_floor(x):
    if '低' in x:
        return 0
    elif '中' in x:
        return 1
    elif '高' in x:
        return 2
    elif '顶' in x:
        return 3
    elif '底' in x:
        return 4
    elif '未知' in x:
        return 5
    elif '混合' in x:
        return 6
    else:
        return -1

train['floor'] = train['floor'].apply(process_floor)
x_test['floor'] = x_test['floor'].apply(process_floor)

# Xử lý các biến phân loại
categorical_features = ['renovationCondition', 'buildingStructure', 'district', 'buildingType']
for col in categorical_features:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].astype(str)) + list(x_test[col].astype(str)))
    train[col] = lbl.transform(train[col].astype(str))
    x_test[col] = lbl.transform(x_test[col].astype(str))
    
# Xử lý các giá trị thiếu
train.fillna(-999, inplace=True)
x_test.fillna(-999, inplace=True)
# xoá đi các hàng livingRoom có giá trị #NAME?
train = train[train['livingRoom'] != '#NAME?']
x_test = x_test[x_test['livingRoom'] != '#NAME?']
# xoá đi các hàng constructionTime có giá trị 未知
train = train[train['constructionTime'] != '未知']
x_test = x_test[x_test['constructionTime'] != '未知']
# Convert columns to numeric, coercing errors to NaN
train['livingRoom'] = pd.to_numeric(train['livingRoom'], errors='coerce')
train['drawingRoom'] = pd.to_numeric(train['drawingRoom'], errors='coerce')
train['kitchen'] = pd.to_numeric(train['kitchen'], errors='coerce')
train['bathRoom'] = pd.to_numeric(train['bathRoom'], errors='coerce')

x_test['livingRoom'] = pd.to_numeric(x_test['livingRoom'], errors='coerce')
x_test['drawingRoom'] = pd.to_numeric(x_test['drawingRoom'], errors='coerce')
x_test['kitchen'] = pd.to_numeric(x_test['kitchen'], errors='coerce')
x_test['bathRoom'] = pd.to_numeric(x_test['bathRoom'], errors='coerce')

# Perform calculations after conversion
train['room_sum'] = train['livingRoom'] + train['drawingRoom'] + train['kitchen'] + train['bathRoom']
x_test['room_sum'] = x_test['livingRoom'] + x_test['drawingRoom'] + x_test['kitchen'] + x_test['bathRoom']

train['area_per_room'] = train['square'] / train['room_sum']
x_test['area_per_room'] = x_test['square'] / x_test['room_sum']

# Handling NaN values by filling with the mean
train['livingRoom'].fillna(train['livingRoom'].mean(), inplace=True)
train['drawingRoom'].fillna(train['drawingRoom'].mean(), inplace=True)
train['kitchen'].fillna(train['kitchen'].mean(), inplace=True)
train['bathRoom'].fillna(train['bathRoom'].mean(), inplace=True)

x_test['livingRoom'].fillna(x_test['livingRoom'].mean(), inplace=True)
x_test['drawingRoom'].fillna(x_test['drawingRoom'].mean(), inplace=True)
x_test['kitchen'].fillna(x_test['kitchen'].mean(), inplace=True)
x_test['bathRoom'].fillna(x_test['bathRoom'].mean(), inplace=True)
# Chuẩn hóa dữ liệu
scaler = StandardScaler()
numerical_feats = train.select_dtypes(include=[np.number]).columns.tolist()
numerical_feats.remove('TARGET')

train[numerical_feats] = scaler.fit_transform(train[numerical_feats])
x_test[numerical_feats] = scaler.transform(x_test[numerical_feats])
# Tách dữ liệu train và validation
X = train.drop('TARGET', axis=1)
y = train['TARGET']
X_train_part, X_val, y_train_part, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Huấn luyện RandomForest để lấy kết quả dự đoán
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_part, y_train_part)
rf_train_pred = rf.predict(X_train_part)
rf_val_pred = rf.predict(X_val)
# Huấn luyện mô hình MLP
def build_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model
# Thêm kết quả RandomForest vào dữ liệu huấn luyện MLP
X_train_mlp = X_train_part.copy()
X_train_mlp['rf_pred'] = rf_train_pred
X_val_mlp = X_val.copy()
X_val_mlp['rf_pred'] = rf_val_pred
# Xây dựng mô hình MLP
mlp_model = build_mlp_model(X_train_mlp.shape[1])

# Sử dụng ReduceLROnPlateau
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
# Đánh giá mô hình trên tập validation
val_pred = mlp_model.predict(X_val_mlp)
mae = mean_absolute_error(y_val, val_pred)
print(f"Mean Absolute Error trên tập validation: {mae}")
# Dự đoán trên tập test
# Dự đoán kết quả từ RandomForest trên x_test
rf_test_pred = rf.predict(x_test)

# Thêm kết quả rf_pred vào x_test để dùng cho MLP
x_test_mlp = x_test.copy()
x_test_mlp['rf_pred'] = rf_test_pred

# Dự đoán kết quả cuối cùng bằng mô hình MLP
test_pred = mlp_model.predict(x_test_mlp)
x_test_ids = np.arange(len(x_test))  

# Chuẩn bị file kết quả
output = pd.DataFrame({'ID': x_test_ids, 'TARGET': test_pred.flatten()})
print(output.head())

# Lưu kết quả dự đoán ra file CSV
output.to_csv('submission.csv', index=False)