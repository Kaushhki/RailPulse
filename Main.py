import pandas as pd

df = pd.read_csv('train_delays.csv')
 
print(df.shape)
print(df.columns.tolist())
print(df.head(10))
print(df.dtypes)
print(df.isnull().sum())
print(df.shape)
print(df['route'].unique())
print(df['route'].nunique())
print(df['time_slot'].unique())
print(df['day_of_week'].unique())
print(df['delay_minutes'].describe())
print(df['is_monsoon'].value_counts())
print(df['is_weekend'].value_counts())

print(df.groupby('route')['delay_minutes'].mean())
print(df.groupby('time_slot')['delay_minutes'].mean())
print(df.groupby('day_of_week')['delay_minutes'].mean())
print(df.groupby(['route', 'time_slot'])['delay_minutes'].mean())

df_model = df.drop(columns=['date', 'is_monsoon', 'is_weekend'])

df_encoded = pd.get_dummies(df_model, columns=['route', 'time_slot', 'day_of_week'], drop_first=True)

print(df_encoded.shape)
print(df_encoded.columns.tolist())
print(df_encoded.head()) 
print(df['month'].unique())

df_model = df.drop(columns=['date', 'is_monsoon', 'is_weekend'])

df_encoded = pd.get_dummies(df_model, columns=['route', 'time_slot', 'day_of_week', 'month'], drop_first=True)

print(df_encoded.shape)
print(df_encoded.columns.tolist())

df_encoded['month'] = df['month'] 

train_df = df_encoded[df_encoded['month'].isin([10, 11, 12, 1])].drop(columns=['month'])
test_df = df_encoded[df_encoded['month'].isin([2, 3, 4])].drop(columns=['month'])

X_train = train_df.drop(columns=['delay_minutes'])
y_train = train_df['delay_minutes']

X_test = test_df.drop(columns=['delay_minutes'])
y_test = test_df['delay_minutes']

print(X_train.shape, X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print('MAE:', mae)
print('RMSE:', rmse)
print('R2:', r2)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print('RF MAE:', rf_mae)
print('RF RMSE:', rf_rmse)
print('RF R2:', rf_r2)

importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances_sorted = importances.sort_values(ascending=False)
print(importances_sorted)

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_rmse = np.sqrt(gb_mse)
gb_r2 = r2_score(y_test, gb_predictions)

print('GB MAE:', gb_mae)
print('GB RMSE:', gb_rmse)
print('GB R2:', gb_r2)

import joblib

joblib.dump(model, 'delay_regression_model.pkl')

loaded_model = joblib.load('delay_regression_model.pkl')
test_prediction = loaded_model.predict(X_test.iloc[[0]])
print(test_prediction)
 
from sklearn.preprocessing import StandardScaler

cluster_features = df_encoded.drop(columns=['month'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

print(scaled_features.shape)
print(scaled_features[:3])

from sklearn.cluster import KMeans

inertia_values = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_features)
    inertia_values.append(km.inertia_)

for k, inertia in zip(k_range, inertia_values):
    print(k, inertia)

kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels_4 = kmeans_4.fit_predict(scaled_features)

df_encoded['cluster_k4'] = cluster_labels_4

comparison_k4 = pd.crosstab(df['time_slot'], df_encoded['cluster_k4'])
print(comparison_k4)

cluster_3_rows = df[df_encoded['cluster_k4'] == 3]

print(cluster_3_rows['route'].value_counts())
print(cluster_3_rows['day_of_week'].value_counts())
print(cluster_3_rows['month'].value_counts())
print(cluster_3_rows['delay_minutes'].describe())

cluster_features_v2 = df_encoded.drop(columns=['month', 'cluster_k4'], errors='ignore')
cluster_features_v2 = cluster_features_v2[[c for c in cluster_features_v2.columns if not c.startswith('month_')]]

scaled_features_v2 = scaler.fit_transform(cluster_features_v2)

kmeans_4_v2 = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels_4_v2 = kmeans_4_v2.fit_predict(scaled_features_v2)

comparison_v2 = pd.crosstab(df['time_slot'], cluster_labels_4_v2)
print(comparison_v2)

cluster_3_v2_rows = df[cluster_labels_4_v2 == 3]

print(cluster_3_v2_rows['day_of_week'].value_counts())
print(cluster_3_v2_rows['delay_minutes'].describe())
print(df[cluster_labels_4_v2 != 3]['delay_minutes'].describe())

kmeans_8 = KMeans(n_clusters=8, random_state=42, n_init=10)
cluster_labels_8 = kmeans_8.fit_predict(scaled_features_v2)

comparison_8 = pd.crosstab([df['time_slot'], df['day_of_week']], cluster_labels_8)
print(comparison_8)

from sklearn.metrics import silhouette_score

sil_4 = silhouette_score(scaled_features_v2, cluster_labels_4_v2)
sil_8 = silhouette_score(scaled_features_v2, cluster_labels_8)

print('Silhouette K=4:', sil_4)
print('Silhouette K=8:', sil_8)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features_v2)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels_4_v2, cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clusters (K=4) Visualized via PCA')
plt.colorbar(scatter, label='Cluster')
plt.savefig('cluster_visualization.png')
plt.show()

print('Explained variance ratio:', pca.explained_variance_ratio_)
print('Total variance explained:', sum(pca.explained_variance_ratio_))