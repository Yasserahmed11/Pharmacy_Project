# Pharmacy Project - Main Script

# Backend Configuration 
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend (fix TclError)

# Required Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Data Collection / Importing

DATA_PATH = "Data/PharmacyTransactionalDataset"

all_data = []

for city in os.listdir(DATA_PATH):
    city_path = os.path.join(DATA_PATH, city)
    if os.path.isdir(city_path):
        for zone in os.listdir(city_path):
            zone_path = os.path.join(city_path, zone)
            if os.path.isdir(zone_path):
                for file in os.listdir(zone_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(zone_path, file)
                        df = pd.read_csv(file_path)

                        # Add location info
                        df["City"] = city
                        df["Zone"] = zone

                        all_data.append(df)

pharmacy_df = pd.concat(all_data, ignore_index=True)

print("Data loaded successfully")
print("Dataset shape:", pharmacy_df.shape)

# Step 3: Data Cleaning & Preprocessing

pharmacy_df["addeddate"] = pd.to_datetime(pharmacy_df["addeddate"], errors="coerce")

pharmacy_df.drop_duplicates(inplace=True)

pharmacy_df.fillna({"Sales_Sheet": 0,"Sales_pack": 0}, inplace=True)

print("After cleaning shape:", pharmacy_df.shape)

# Step 4: Exploratory Data Analysis (EDA)

print("\nDescriptive Statistics:")
print(pharmacy_df.describe())

print("\nTransactions per City:")
print(pharmacy_df["City"].value_counts())

# Step 5: Feature Engineering

pharmacy_df["Total_Sales"] = (pharmacy_df["Sales_Sheet"] + pharmacy_df["Sales_pack"])

pharmacy_df["Month"] = pharmacy_df["addeddate"].dt.month
pharmacy_df["Day"] = pharmacy_df["addeddate"].dt.day

pharmacy_df["Hour"]=pd.to_datetime(pharmacy_df["time_"], errors="coerce").dt.hour

# Step 6: Statistical Analysis

avg_sales_city = pharmacy_df.groupby("City")["Total_Sales"].mean()
avg_sales_type = pharmacy_df.groupby("type")["Total_Sales"].mean()

print("\nAverage Sales per City:")
print(avg_sales_city)

print("\nAverage Sales per Type:")
print(avg_sales_type)

# Step 7: Data Visualization (Saved to Files)

plt.figure()
sns.barplot(x=avg_sales_city.index, y=avg_sales_city.values)
plt.title("Average Total Sales by City")
plt.xlabel("City")
plt.ylabel("Average Total Sales")
plt.savefig("avg_sales_by_city.png")
plt.close()

plt.figure()
sns.barplot(x=avg_sales_type.index, y=avg_sales_type.values)
plt.title("Average Total Sales by Type")
plt.xlabel("Type")
plt.ylabel("Average Total Sales")
plt.savefig("avg_sales_by_type.png")
plt.close()

print("Plots saved successfully")

# Step 8: Modeling & Prediction

model_df = pharmacy_df[
 ["Sales_Sheet", "Sales_pack", "Month", "Day", "Hour"]
]

# تأكد من عدم وجود NaN
model_df = model_df.fillna(0)

print("\nModel Data Shape:", model_df.shape)

X = model_df
y = pharmacy_df["Total_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("\n==============================")
print("Model Evaluation Result")
print("Mean Squared Error (MSE):", mse)
print("==============================")

# End of Script
