import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('heart.csv')
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=334)

# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Build the Neural Network model
model = Sequential([
    Dense(64, input_dim=x_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_scaled, y_train, epochs=50, batch_size=10, validation_data=(x_test_scaled, y_test))

# Neural Network Predictions
y_pred_nn = (model.predict(x_test_scaled) > 0.5).astype(int)

# Logistic Regression using StatsModels
x_train_const = sm.add_constant(x_train)
x_test_const = sm.add_constant(x_test)  # Add constant to the testing set

logit_model = sm.Logit(y_train, x_train_const)
result = logit_model.fit()

# Logistic Regression Predictions
y_pred_logistic = (result.predict(x_test_const) > 0.5).astype(int)

# Accuracy Metrics
def accuracy_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

precision_nn, recall_nn, f1_nn = accuracy_metrics(y_test, y_pred_nn)
precision_logistic, recall_logistic, f1_logistic = accuracy_metrics(y_test, y_pred_logistic)

print(f"Neural Network - Precision: {precision_nn}, Recall: {recall_nn}, F1 Score: {f1_nn}")
print(f"Logistic Regression - Precision: {precision_logistic}, Recall: {recall_logistic}, F1 Score: {f1_logistic}")

# Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(y_true, y_pred):
    ranks = []
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            ranks.append(1)
    return np.mean(ranks)

# Calculate MRR
mrr_nn = mean_reciprocal_rank(y_test, y_pred_nn)
mrr_logistic = mean_reciprocal_rank(y_test, y_pred_logistic)

print(f"Neural Network - MRR: {mrr_nn}")
print(f"Logistic Regression - MRR: {mrr_logistic}")

# Diversity Metrics
from scipy.spatial.distance import pdist

# Intra-List Diversity
def intra_list_diversity(recommendations):
    if len(recommendations) <= 1:
        return 0
    distances = pdist(recommendations, metric='cosine')
    return np.mean(distances)

# Inter-List Diversity
def inter_list_diversity(recommendation_lists):
    diversities = [intra_list_diversity(recommendations) for recommendations in recommendation_lists]
    return np.mean(diversities)

# Assuming recommendations_nn and recommendations_logistic are the recommendation lists
recommendations_nn = model.predict(x_test_scaled)  # Assuming these are item representations
recommendations_logistic = result.predict(x_test_const)

# Convert Pandas Series to NumPy array
recommendations_logistic = recommendations_logistic.values

# Reshape recommendations_nn and recommendations_logistic if necessary
recommendations_nn = recommendations_nn.reshape(-1, 1) if len(recommendations_nn.shape) == 1 else recommendations_nn
recommendations_logistic = recommendations_logistic.reshape(-1, 1) if len(recommendations_logistic.shape) == 1 else recommendations_logistic

# Calculate diversity metrics
intra_div_nn = intra_list_diversity(recommendations_nn)
intra_div_logistic = intra_list_diversity(recommendations_logistic)
inter_div_nn = inter_list_diversity([recommendations_nn])
inter_div_logistic = inter_list_diversity([recommendations_logistic])

print(f"Neural Network - Intra-List Diversity: {intra_div_nn}, Inter-List Diversity: {inter_div_nn}")
print(f"Logistic Regression - Intra-List Diversity: {intra_div_logistic}, Inter-List Diversity: {inter_div_logistic}")

# Assuming you have a popularity measure for your items
# Replace 'popularity' with the correct column name representing item popularity in your DataFrame
item_popularity_column = 'popularity'
if item_popularity_column in df.columns:
    item_popularity = df[item_popularity_column].values  
else:
    print("Column 'popularity' not found in DataFrame.")

# Average Popularity
def average_popularity(recommendations, item_popularity):
    return np.mean([item_popularity[item] for item in recommendations])

# Novelty Score (inverse of popularity)
def novelty_score(recommendations, item_popularity):
    return np.mean([1 / item_popularity[item] for item in recommendations])

# Convert predictions to indices of recommended items
indices_nn = np.where(y_pred_nn == 1)[0]
indices_logistic = np.where(y_pred_logistic == 1)[0]

# Check if item_popularity is defined before using it
if 'item_popularity' in locals():
    avg_popularity_nn = average_popularity(indices_nn, item_popularity)
    avg_popularity_logistic = average_popularity(indices_logistic, item_popularity)
    novelty_nn = novelty_score(indices_nn, item_popularity)
    novelty_logistic = novelty_score(indices_logistic, item_popularity)

    print(f"Neural Network - Average Popularity: {avg_popularity_nn}, Novelty Score: {novelty_nn}")
    print(f"Logistic Regression - Average Popularity: {avg_popularity_logistic}, Novelty Score: {novelty_logistic}")
else:
    print("item_popularity is not defined.")

# Visualization
metrics = ['Precision', 'Recall', 'F1 Score', 'MRR']
nn_metrics = [precision_nn, recall_nn, f1_nn, mrr_nn]
logistic_metrics = [precision_logistic, recall_logistic, f1_logistic, mrr_logistic]

diversity_metrics = ['Intra-List Diversity', 'Inter-List Diversity']
nn_diversity = [intra_div_nn, inter_div_nn]
logistic_diversity = [intra_div_logistic, inter_div_logistic]

if 'item_popularity' in locals():
    popularity_metrics = ['Average Popularity', 'Novelty Score']
    nn_popularity = [avg_popularity_nn, novelty_nn]
    logistic_popularity = [avg_popularity_logistic, novelty_logistic]

# Plotting the metrics
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, nn_metrics, width, label='Neural Network')
plt.bar(x + width/2, logistic_metrics, width, label='Logistic Regression')
plt.xticks(x, metrics)
plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.legend()

plt.subplot(1, 2, 2)
x = np.arange(len(diversity_metrics))
plt.bar(x - width/2, nn_diversity, width, label='Neural Network')
plt.bar(x + width/2, logistic_diversity, width, label='Logistic Regression')
plt.xticks(x, diversity_metrics)
plt.ylabel('Score')
plt.title('Diversity Metrics')
plt.legend()

plt.tight_layout()
plt.show()

if 'item_popularity' in locals():
    plt.figure(figsize=(8, 4))

    x = np.arange(len(popularity_metrics))
    plt.bar(x - width/2, nn_popularity, width, label='Neural Network')
    plt.bar(x + width/2, logistic_popularity, width, label='Logistic Regression')
    plt.xticks(x, popularity_metrics)
    plt.ylabel('Score')
    plt.title('Popularity and Novelty Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()
