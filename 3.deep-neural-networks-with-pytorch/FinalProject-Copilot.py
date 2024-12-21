import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Step 1: Data Loading and Preprocessing
# Task 1: Load the League of Legends dataset and preprocess it for training.
data = pd.read_csv('league_of_legends.csv')
X = data.drop('match_outcome', axis=1)
y = data['match_outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Step 2: Logistic Regression Model
# Task 2: Implement a logistic regression model using PyTorch.
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 3: Model Training
# Task 3: Train the logistic regression model on the dataset.
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 4: Model Optimization and Evaluation
# Task 4: Implement optimization techniques and evaluate the model's performance.
with torch.no_grad():
    model.eval()
    train_outputs = model(X_train)
    test_outputs = model(X_test)
    train_loss = criterion(train_outputs, y_train)
    test_loss = criterion(test_outputs, y_test)
    print(f'Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Step 5: Visualization and Interpretation
# Task 5: Visualize the model's performance and interpret the results.
train_pred = (train_outputs >= 0.5).float()
test_pred = (test_outputs >= 0.5).float()
train_cm = confusion_matrix(y_train, train_pred)
test_cm = confusion_matrix(y_test, test_pred)
print('Train Confusion Matrix:\n', train_cm)
print('Test Confusion Matrix:\n', test_cm)

fpr, tpr, _ = roc_curve(y_test, test_outputs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Step 6: Model Saving and Loading
# Task 6: Save and load the trained model.
torch.save(model.state_dict(), 'logistic_regression_model.pth')
loaded_model = LogisticRegressionModel(input_dim)
loaded_model.load_state_dict(torch.load('logistic_regression_model.pth'))
loaded_model.eval()

# Step 7: Hyperparameter Tuning
# Task 7: Perform hyperparameter tuning to find the best learning rate.
best_lr = 0.01
best_accuracy = 0
for lr in [0.001, 0.01, 0.1]:
    model = LogisticRegressionModel(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test)
        test_pred = (test_outputs >= 0.5).float()
        accuracy = (test_pred == y_test).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
print(f'Best Learning Rate: {best_lr}, Best Accuracy: {best_accuracy:.4f}')

# Step 8: Feature Importance
# Task 8: Evaluate feature importance to understand the impact of each feature on the prediction.
with torch.no_grad():
    feature_importance = model.linear.weight[0].numpy()
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()

