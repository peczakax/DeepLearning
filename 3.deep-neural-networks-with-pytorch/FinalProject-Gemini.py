import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Data Loading and Preprocessing
# Load the dataset (replace with your actual dataset path)
try:
    data = pd.read_csv('LeagueofLegends.csv') # Or whatever name you've stored it as
except FileNotFoundError:
    print("Dataset file not found. Please ensure the file exists in the same directory or provide the correct path.")
    exit()

# Preprocess the data (example - adapt to your specific dataset)
# Select relevant features and target variable
# Example: Assuming 'blueWins' is the target and other columns are features
X = data.drop('blueWins', axis=1)  # Replace 'blueWins' if different
y = data['blueWins']
# Convert string columns to numerical (if needed)
for col in X.columns:
    if X[col].dtype == 'object':
       try:
           X[col] = X[col].astype(float) #  Attempt direct conversion to float
       except: 
           print(f"Column {col} is of object/string type and couldn't be directly converted to numeric. Please investigate further.")
           exit()
# Handle missing values (if any) - simple imputation for example. Consider more advanced methods if necessary.
X.fillna(X.mean(), inplace=True) 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)



# Step 2: Logistic Regression Model
input_dim = X_train.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 1),
    nn.Sigmoid()
)

# Step 3: Model Training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Initialize learning rate

epochs = 100  # Adjust as needed
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Step 4: Model Optimization and Evaluation (L2 Regularization)
# Add L2 regularization to the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01) # Example weight_decay value

# Retrain the model (repeat Step 3 with the new optimizer)


# Step 5 & 6: Evaluation, Visualization, Saving/Loading

# Make predictions
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_binary = (y_pred > 0.5).float()  # Convert probabilities to binary predictions

# Evaluation metrics (add more if needed - precision, recall, F1, etc.)
conf_matrix = confusion_matrix(y_test.numpy(), y_pred_binary.numpy())
fpr, tpr, _ = roc_curve(y_test.numpy(), y_pred.numpy())
roc_auc = auc(fpr, tpr)

print("Confusion Matrix:\n", conf_matrix)
print("ROC AUC:", roc_auc)

# Visualizations
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# Save the model
torch.save(model.state_dict(), 'lol_model.pth')

# Load the model
loaded_model = nn.Sequential(
    nn.Linear(input_dim, 1),
    nn.Sigmoid()
)

loaded_model.load_state_dict(torch.load('lol_model.pth'))


# Step 7: Hyperparameter Tuning (Learning Rate) – (Illustrative example)
learning_rates = [0.001, 0.01, 0.1]
best_lr = 0.01  # Placeholder
best_accuracy = 0.0 # Placeholder

# Step 8: Feature Importance
weights = model[0].weight.data.numpy()
# (Visualization of feature importance is dataset-specific and can be added here)
plt.show()

