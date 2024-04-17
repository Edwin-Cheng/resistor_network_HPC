#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import random
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

# Seed setup and early stopping classes remain the same...
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epochs, epoch):

        score = -val_loss


        if epoch <= int(epochs*0.7) and self.counter == self.patience-1:
          self.patience = 1000

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'checkpoint.pt') # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def Load_data(network_type, name):
  # Data loading
  filename = f'training_data_{network_type}_{name}.txt'
  data = pd.read_csv(filename, header=None)
  X = data.values[:,:-1]
  y = data.values[:,-1]

  # Convert targets to one-hot encoding
  encoder = OneHotEncoder(sparse=False)
  y = encoder.fit_transform(y.reshape(-1, 1))
  num_classes = y.shape[1]

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
  return X_train, X_test, y_train, y_test, num_classes

# DNN model
class DNN(torch.nn.Module):
    def __init__(self, hidden, num_classes):
        super(DNN, self).__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden[0], hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden[1], hidden[2]),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden[2], hidden[3]),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden[2], num_classes)
        )
        for layer in self.out:
            if type(layer) == torch.nn.Linear:
                torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

class Pipeline:
    def __init__(self, X_train, X_test, y_train, y_test, learning_rate, hidden_size, num_classes, epochs, patience):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.patience = patience


    def _feature_process(self):
        X_scaler = StandardScaler()
        self.X_scaler = X_scaler
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
        X_train = torch.from_numpy(X_train[:, np.newaxis, :]).float()
        X_test = torch.from_numpy(X_test[:, np.newaxis, :]).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        if torch.cuda.is_available():
            X_train = X_train.cuda()
            X_test = X_test.cuda()
            y_train = y_train.cuda()
            y_test = y_test.cuda()
        self.X_test = X_test
        self.y_test = y_test
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        return train_dataloader, test_dataloader

    def _build_model(self):
        #setup_seed(0)
        model = DNN(self.hidden_size, self.num_classes)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _run(self, train_dataloader, test_dataloader, model, epochs, patience):
        early_stopping = EarlyStopping(patience, verbose=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        t0 = time.time()
        for epoch in range(epochs):  #Total training steps, modify it when you need
            model.train()
            train_loss = list()
            last_loss = 0
            if epoch%100 == 0:
              print(f"--------------------------------------------Epoch: {epoch} ----------------------------------------------------------------")
            for data in train_dataloader:
                optimizer.zero_grad()
                X_, y_ = data
                output = model(X_)
                loss = criterion(output, torch.max(y_, 1)[1])
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            #print('train_loss:', np.mean(train_loss))

            with torch.no_grad():
                test_loss = list()
                model.eval()

                for i, data in enumerate(test_dataloader):
                    X_, y_ = data
                    output = model(X_)
                    loss = criterion(output, torch.max(y_, 1)[1])
                    test_loss.append(loss.item())
                    if i == len(test_dataloader)-1:
                      last_loss = loss

                #print("test_loss:", np.mean(test_loss))
                early_stopping(np.mean(test_loss), model, epochs, epoch)
            if early_stopping.early_stop:
                time_taken = int(time.time() - t0)
                minute, second = time_taken//60, time_taken%60
                print("Early stopping")
                print(f"Last Epoch is {epoch}")
                print(f"Last loss is {last_loss}")
                print(f"Time taken is {minute} minute {second} second")
                break

        model.load_state_dict(torch.load(f'checkpoint.pt'))
        return model

    def get_metrics(self, model):
        y_pred = model(self.X_test).cpu().detach().numpy()
        y_test = self.y_test.cpu().detach().numpy()
        accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        result = {
            'y_test': np.argmax(y_test, axis=1),
            'y_pred': np.argmax(y_pred, axis=1)
        }
        return accuracy, result

    def run(self):
        train_dataloader, test_dataloader = self._feature_process()
        model = self._build_model()
        model = self._run(train_dataloader, test_dataloader, model, epochs=self.epochs, patience=self.patience)
        accuracy, result = self.get_metrics(model)
        return accuracy, result, model, self.X_scaler
    

from generate_data import setup_training_data
def run_experiment(network_type = "square", name = str, exp_num = str, epochs=20000, patience=500):
    X_train, X_test, y_train, y_test, num_classes = Load_data(network_type, name)
    # (self, X_train, X_test, y_train, y_test, learning_rate, hidden_size, num_classes, epochs, patience)
    pipeline = Pipeline(X_train, X_test, y_train, y_test, 0.0015, [4,128,128], num_classes,  epochs, patience)     #structure of DNN
    accuracy, result, model, scaler = pipeline.run()
    print(f"The model accuracy is {accuracy}")
    #print(f"The model loss is {loss}")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(f'model_info_{exp_num}.txt', 'w') as f:
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'accuracy: {accuracy}\n')
        f.write(f'')