#!/usr/bin/env python
# coding: utf-8

# Implementation of training and evaluating SARIMA models for website traffic forecasting:

# 1. Using SARIMA

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# In[2]:


# Load the dataset
df = pd.read_csv(r'C:\Users\manas\OneDrive\Desktop\584 final project\webtraffic.csv')


# In[3]:


df


# In[4]:


# Split the dataset into a training set and a test set
train_set = df[:int(len(df) * 0.8)]
test_set = df[int(len(df) * 0.8):]


# In[5]:


# Train the SARIMA model
order = (1, 1, 1)
model = SARIMAX(train_set['Sessions'], order=order)
results = model.fit()


# In[6]:


# Train the SARIMA model
order = (1, 1, 1)
model = SARIMAX(train_set['Sessions'], order=order)
results = model.fit()


# In[7]:



# Make predictions on the test set
start_index = test_set.index[0]
end_index = test_set.index[-1]
predictions = results.predict(start=start_index, end=end_index)


# In[8]:


# Plotting actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(train_set.index, train_set['Sessions'], label='Train')
plt.plot(test_set.index, test_set['Sessions'], label='Test')
plt.plot(test_set.index, predictions, label='Predictions', color='red')
plt.title('SARIMA Model - Web Traffic Forecasting')
plt.xlabel('Hour')
plt.ylabel('Sessions')
plt.legend()
plt.show()


# In[9]:



# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(test_set['Sessions'], predictions)
rmse = np.sqrt(mse)


# In[10]:


# Print evaluation metrics
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)


# In[11]:



from sklearn.metrics import accuracy_score

# Create a binary column indicating high or low traffic for the entire dataset
threshold = 1000000000  # Adjust this threshold based on your problem
df['HighTraffic'] = np.where(df['Sessions'] > threshold, 1, 0)

# Split the dataset into a training set and a test set
train_set = df[:int(len(df) * 0.8)]
test_set = df[int(len(df) * 0.8):]

# Train the SARIMA model
order = (1, 1, 1)
model = SARIMAX(train_set['Sessions'], order=order)
results = model.fit()

# Make predictions on the test set
start_index = test_set.index[0]
end_index = test_set.index[-1]
predictions = results.predict(start=start_index, end=end_index)

# Convert predictions to binary classes based on the threshold
predicted_classes = np.where(predictions > threshold, 1, 0)

# Calculate accuracy
accuracy = accuracy_score(test_set['HighTraffic'], predicted_classes)

# Print accuracy
print('Accuracy:', accuracy)


# 2. Using MLPregressor

# In[12]:


# Prepare the training data
X_train = train_set.index.values.reshape(-1, 1)  # Using hour values as features
y_train = train_set['Sessions']


# In[13]:


# Prepare the test data
X_test = test_set.index.values.reshape(-1, 1)
y_test = test_set['Sessions']


# In[14]:


# Initialize the MLPRegressor
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)


# In[15]:


# Train the model
mlp.fit(X_train, y_train)

# Make predictions on the test set
predictions = mlp.predict(X_test)


# In[16]:


# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)


# In[17]:


# Plotting actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_set.index, y_test, label='Actual Traffic')
plt.plot(test_set.index, predictions, label='Predicted Traffic', color='red')
plt.title('Actual vs. Predicted Traffic using MLP')
plt.xlabel('Hour')
plt.ylabel('Sessions')
plt.legend()
plt.show()


# 3. Using MLP with resilient backpropagation (Rprop)

# In[18]:


# Initialize the MLPRegressor with Rprop
mlp = MLPRegressor(solver='rprop', hidden_layer_sizes=(100,), max_iter=1000, random_state=42)


# In[19]:


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler





# Prepare the training data
X_train = train_set.index.values.reshape(-1, 1).astype(np.float32)  # Using hour values as features
y_train = train_set['Sessions'].values.reshape(-1, 1).astype(np.float32)

# Prepare the test data
X_test = test_set.index.values.reshape(-1, 1).astype(np.float32)
y_test = test_set['Sessions'].values.reshape(-1, 1).astype(np.float32)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)
X_test = scaler_X.transform(X_test)
y_test = scaler_y.transform(y_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x

# Instantiate the model
model = MLP()

# Define the Rprop optimizer
optimizer = optim.Rprop(model.parameters(), lr=0.01)

# Define the Mean Squared Error loss
criterion = nn.MSELoss()

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions on the test set
with torch.no_grad():
    predictions = model(X_test_tensor)

# Inverse transform to get the original scale
predictions = scaler_y.inverse_transform(predictions.numpy())
y_test = scaler_y.inverse_transform(y_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Plotting actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_set.index, y_test, label='Actual Traffic')
plt.plot(test_set.index, predictions, label='Predicted Traffic', color='red')
plt.title('Actual vs. Predicted Traffic using MLP with Rprop (PyTorch)')
plt.xlabel('Hour')
plt.ylabel('Sessions')
plt.legend()
plt.show()


#  4. Using recurrent neural network (RNN)

# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out


# In[22]:


# Instantiate the model
input_size = 1  # Number of features (in this case, it's 1 for 'Hour')
hidden_size = 50
output_size = 1  # Number of output units
model = SimpleRNN(input_size, hidden_size, output_size)


# In[23]:


# Define the Mean Squared Error loss
criterion = nn.MSELoss()


# In[24]:


# Define the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[25]:


# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor.unsqueeze(2))  # Add an extra dimension for the input sequence
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# In[26]:



# Make predictions on the test set
with torch.no_grad():
    test_inputs = X_test_tensor.unsqueeze(2)  # Add an extra dimension for the input sequence
    predictions = model(test_inputs)


# In[27]:


# Inverse transform to get the original scale
predictions = scaler_y.inverse_transform(predictions.numpy())
y_test = scaler_y.inverse_transform(y_test)


# In[29]:


# Plotting actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_set.index, y_test, label='Actual Traffic')
plt.plot(test_set.index, predictions, label='Predicted Traffic', color='red')
plt.title('Actual vs. Predicted Traffic using RNN (PyTorch)')
plt.xlabel('Hour')
plt.ylabel('Sessions')
plt.legend()
plt.show()


# 5. deep learning stacked autoencoder (SAE).

# In[30]:


# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)


# In[31]:


# Define the Stacked Autoencoder (SAE) model
class StackedAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(StackedAutoencoder, self).__init__()
        encoder_layers = []
        decoder_layers = []

        # Build encoder layers
        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(input_size, hidden_size))
            encoder_layers.append(nn.ReLU())
            input_size = hidden_size

        # Build decoder layers
        for hidden_size in reversed(hidden_sizes[:-1]):
            decoder_layers.append(nn.Linear(input_size, hidden_size))
            decoder_layers.append(nn.ReLU())
            input_size = hidden_size

        # Final layer of decoder
        decoder_layers.append(nn.Linear(input_size, input_size))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# In[32]:


# Instantiate the SAE model
input_size = 1  # Number of features (in this case, it's 1 for 'Hour')
hidden_sizes = [50, 20, 10]  # You can adjust the number of hidden layers and their sizes
sae_model = StackedAutoencoder(input_size, hidden_sizes)


# In[33]:


# Define the Mean Squared Error loss
criterion = nn.MSELoss()

# Define the Adam optimizer
optimizer = optim.Adam(sae_model.parameters(), lr=0.001)


# In[34]:


# Train the SAE model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = sae_model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# In[35]:


# Encode the training and test data
with torch.no_grad():
    encoded_X_train = sae_model.encoder(X_train_tensor).numpy()
    encoded_X_test = sae_model.encoder(X_test_tensor).numpy()


# In[36]:


# Visualize the encoded features
print("Shapes before squeezing:")
print("X_test:", X_test.shape)
print("encoded_X_test:", encoded_X_test.shape)



# In[37]:


plt.scatter(X_test.squeeze(), encoded_X_test[:, 0], label='Encoded Features (Test Set)', marker='x')
plt.title('Encoded Features using Stacked Autoencoder')
plt.xlabel('Hour')
plt.ylabel('Encoded Features (First Column)')
plt.legend()
plt.show()

