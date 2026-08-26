import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel("Data/HousePricePrediction.xlsx")

print(dataset.head())
print("\nDataset shape:")
print(dataset.shape)



#Data Preprocessing: Examine Dataset
print("\nDataset information:")
print(dataset.info())

print("\nMissing values:")
print(dataset.isnull().sum())

object_cols = dataset.select_dtypes(include=["object"]).columns
integer_cols = dataset.select_dtypes(include=["int64"]).columns
float_cols = dataset.select_dtypes(include=["float64"]).columns

print("\nCategorical variables:", len(object_cols))
print("Integer variables:", len(integer_cols))
print("Float variables:", len(float_cols))



#Exploratory Data Analysis (Correlation heatmap)
numerical_dataset = dataset.select_dtypes(
    include=["int64", "float64"]
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    numerical_dataset.corr(),
    cmap="BrBG",
    fmt=".2f",
    linewidths=2,
    annot=True
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()

plt.savefig(
    "images/correlation_heatmap.png",
    dpi=300
)

plt.show()



#Explore categorical variables
unique_values = []

for col in object_cols:
    unique_values.append(dataset[col].unique().size)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=object_cols,
    y=unique_values
)

plt.title("Number of Unique Values of Categorical Features")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    "images/categorical_unique_values.png",
    dpi=300
)

plt.show()



#Data Cleaning
dataset.drop(
    ["Id"],
    axis=1,
    inplace=True
)


#Fill missing SalePrice values:
dataset["SalePrice"] = dataset["SalePrice"].fillna(
    dataset["SalePrice"].mean()
)


#Remove the remaining rows containing missing values:
new_dataset = dataset.dropna().copy()

print("\nMissing values after cleaning:")
print(new_dataset.isnull().sum())



#OneHotEncoder - For Label categorical features
from sklearn.preprocessing import OneHotEncoder

object_cols = new_dataset.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print("\nCategorical variables:")
print(object_cols)

print(
    "Number of categorical variables:",
    len(object_cols)
)
#Encode
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

OH_cols = pd.DataFrame(
    OH_encoder.fit_transform(new_dataset[object_cols])
)

OH_cols.index = new_dataset.index

OH_cols.columns = OH_encoder.get_feature_names_out(object_cols)

df_final = new_dataset.drop(object_cols, axis=1)

df_final = pd.concat([df_final, OH_cols], axis=1)

print("\nFinal encoded dataset:")
print(df_final.head())



#Splitting Dataset into Training and Testing: Separate X and y
X = df_final.drop(
    ["SalePrice"],
    axis=1
)

y = df_final["SalePrice"]




#Model Training and Accuracy
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)

print("\nTraining rows:", X_train.shape[0])
print("Validation rows:", X_valid.shape[0])


#1. SVM - Support vector Machine
from sklearn import svm
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()

model_SVR.fit(
    X_train,
    y_train
)

svr_predictions = model_SVR.predict(
    X_valid
)

svr_mape = mean_absolute_percentage_error(
    y_valid,
    svr_predictions
)

print("\nSVR MAPE:")
print(svr_mape)

#2. Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(
    n_estimators=10,
    random_state=0
)

model_RFR.fit(
    X_train,
    y_train
)

rf_predictions = model_RFR.predict(
    X_valid
)

rf_mape = mean_absolute_percentage_error(
    y_valid,
    rf_predictions
)

print("\nRandom Forest MAPE:")
print(rf_mape)

#3. Linear Regression
from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()

model_LR.fit(
    X_train,
    y_train
)

lr_predictions = model_LR.predict(
    X_valid
)

lr_mape = mean_absolute_percentage_error(
    y_valid,
    lr_predictions
)

print("\nLinear Regression MAPE:")
print(lr_mape)


#Proper Model Comparison
results = pd.DataFrame({
    "Model": [
        "Support Vector Regression",
        "Random Forest Regression",
        "Linear Regression"
    ],

    "MAPE": [
        svr_mape,
        rf_mape,
        lr_mape
    ]
})

results["Accuracy Approx (%)"] = (
    1 - results["MAPE"]
) * 100

results = results.sort_values(
    "MAPE"
)

print("\nModel Comparison:")
print(results)

results.to_csv(
    "model_results.csv",
    index=False
)


#Actual vs Predicted Prices
best_predictions = rf_predictions

plt.figure(figsize=(8, 6))

plt.scatter(
    y_valid,
    best_predictions,
    alpha=0.6
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")

plt.title(
    "Actual vs Predicted House Prices"
)

plt.tight_layout()

plt.savefig(
    "images/actual_vs_predicted.png",
    dpi=300
)

plt.show()