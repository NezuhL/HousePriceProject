
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


######STARTING DATA PROCESSING ########

house_data = pd.read_csv("C:/Users/USER/PycharmProjects/pythonProject/data/data_house.csv")
print(f"size data \n , {house_data.shape} \n ,the 5 first lines: \n , {house_data.head()} \n"
      f" , general info \n, {house_data.describe()}\n")

# Part 1 cleaning data:

# checking for missing values
a = house_data.isnull().any()  # האם קיימים ערכים חסרים פר עמודה
total_missing = house_data.isnull().sum().sum()  # סכום הערכים הכולל חסרים בכל עמודה
size_data = house_data.size
missing_percent = (total_missing / size_data) * 100
print(f"the present of the missing data: {missing_percent:.2f}%")
# checking the distribution of missing values
column_missing_data = house_data.isnull().sum()
missing_percent_column = (column_missing_data / len(house_data)) * 100  # אחוז הערכים החסרים
missing_info = pd.DataFrame({'missing_values': column_missing_data, 'percent': missing_percent_column})
missing_info = missing_info[missing_info['missing_values'] > 0]  # סינון ערכים שאין בהם חוסר
print(missing_info)

# Deleting missing data:

# above 50% - deleting the entire column because there is not enough data
col_drop = ["Alley", "MasVnrType", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
house_data = house_data.drop(columns=col_drop)

# Values around 5% are related to the garage,
# so we will check whether it is possible that there is no garage at all in the house and therefore it is empty.
c = house_data[house_data["GarageType"].isnull()][["GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]].head()

# The conclusion is that there is no garage in the house,
# so we will not delete the rows but fill them with the value 0/NONE.

house_data[["GarageType", "GarageFinish", "GarageQual", "GarageCond"]] = \
    house_data[["GarageType", "GarageFinish", "GarageQual", "GarageCond"]].fillna("None")

house_data["GarageYrBlt"] = house_data["GarageYrBlt"].fillna(0)

# Values below 5% are related to the basement, check if all rows are empty and if the basement area is 0
d = house_data[house_data["BsmtQual"].isnull()][["BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]].head()
length_basement = house_data.loc[[17, 39, 90, 102, 156], "TotalBsmtSF"]
# Conclusion: There is no basement, all rows are empty and the area of the basement is also 0

house_data[["BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual"]] = \
    house_data[["BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual"]].fillna("None")

# In the Electrical column, only one data is missing, so we will fill in the most common value.
house_data["Electrical"] = house_data["Electrical"].fillna(house_data["Electrical"].mode()[0])

# When testing MasVnrArea, there is a low correlation with SalePrice, we will fill in 0
correlation = house_data["MasVnrArea"].corr(house_data["SalePrice"])
print(f"Correlation between MasVnrArea and SalePrice: {correlation:.2f}")
house_data["MasVnrArea"] = house_data["MasVnrArea"].fillna(0)

# 17% of the missing values in the LotFrontage column by filling in the median
correlation2 = house_data["LotFrontage"].corr(house_data["SalePrice"])
print(f"Correltion2: {correlation2:.2f}")
house_data["LotFrontage"] = house_data["LotFrontage"].fillna(house_data["LotFrontage"].median())

# Check that there is no missing data
missing_info2 = house_data.isnull().any().any()
print(missing_info2)

##### Part 2: Converting Categorical Data #####

# 1. Convert categorical data to separate columns (One-Hot Encoding)
house_data = pd.get_dummies(house_data, columns=[
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual',
    'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual',
    'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'SaleType', 'SaleCondition'
])

# 2. Converting ordered categorical columns to numeric values (Label Encoding) for ordered columns
label_columns = [
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical'
]

encoder = LabelEncoder()

for column in label_columns:
    if column in house_data.columns:
        house_data[column] = encoder.fit_transform(house_data[column])

########## End of data processing #########

# Tracking MLflow
mlflow.set_tracking_uri("file:///C:/MLflow_Experiments")
test_size = 0.20
run_name = f"model_LinearRegression_testSize_{test_size}"

mlflow.start_run(run_name=run_name)
mlflow.log_param("test_size", test_size)

x = house_data.drop(columns=["SalePrice"])
y = house_data["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

# Scale the data
scaler = StandardScaler()
x_train_sta = scaler.fit_transform(x_train)
x_test_sta = scaler.transform(x_test)

# Model training and prediction
model = LinearRegression()
model.fit(x_train_sta, y_train)
y_predict = model.predict(x_test_sta)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mse)
print(f"mae: {mae: .2f} \nmse: {mse: .2f} \n r2: {r2: .2f} \n rmse: {rmse: .2f}")

# Documenting results for MLflow
mlflow.log_metric("mae", mae)
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("rmse", rmse)

#Documenting and saving the model to MLflow
mlflow.sklearn.log_model(model, f"model_LinearRegression_testSize_{test_size}")
mlflow.log_artifact("C:/Users/USER/PycharmProjects/pythonProject/HousePrice.py")  # שמור את קובץ הקוד


mlflow.end_run()
