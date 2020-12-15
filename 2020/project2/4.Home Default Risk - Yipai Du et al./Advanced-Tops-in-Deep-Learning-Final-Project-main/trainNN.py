# -*- coding: utf-8 -*-
import copy
import gc
import sys
import time

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class NN(torch.nn.Module):
    """
    Pytorch Deep Neural Network, using the Sequential Module
    Loss function: Specify the order of axis to average over. .mean(1) first averages over the features, .mean(0) over the samples
    --> Error metrics for training and evaluation have to be specified properly before training the model
    """
    def __init__(self, input_shape, output_shape, dropout=0.01, patience=30):
        super(NN, self).__init__()
        # Initialize layer shapes
        self.n_in, self.n_out = input_shape, output_shape
        self.n_h1, self.n_h2, self.n_h3, self.n_h4 = 200, 200, 200, 200
        # Define the MSE loss criterion
        self.criterion = torch.nn.BCELoss(reduction='none')
        # Setup the neural network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_h1),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.n_h1, self.n_h2),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.n_h2, self.n_h3),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.n_h3, self.n_h4),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(self.n_h3, self.n_h4), torch.nn.Dropout(p=dropout),
            # torch.nn.ReLU(inplace=True), torch.nn.Linear(self.n_h3, self.n_h4),
            # torch.nn.Dropout(p=dropout), torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(self.n_h3, self.n_h4), torch.nn.Dropout(p=dropout),
            # torch.nn.ReLU(inplace=True), torch.nn.Linear(self.n_h3, self.n_h4),
            # torch.nn.Dropout(p=dropout), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.n_h4, 1),
            torch.nn.Sigmoid())
        # Setup early stopping parameters
        self.patience = patience
        self.old_loss = 1e8
        self.stop_counter = 0

    def forward(self, inputs):
        return self.network(inputs)

    def loss(self, predictions, targets, weights):
        return self.criterion(predictions,
                              targets).squeeze().dot(weights.squeeze())

    def early_stop(self, new_loss, min=True):
        if min == True:
            if new_loss < self.old_loss:
                self.best_state = copy.deepcopy(self.state_dict())
                self.stop_counter = 0
                self.old_loss = new_loss
            else:
                self.stop_counter += 1
        # Implement old_loss as o instead of 1000
        else:
            if new_loss > self.old_loss:
                self.best_state = copy.deepcopy(self.state_dict())
                self.stop_counter = 0
                self.old_loss = new_loss
            else:
                self.stop_counter += 1
        if self.stop_counter > self.patience:
            return True
        else:
            return False

    def retain_best_state(self):
        self.load_state_dict(self.best_state)
        print("best model retained")
        return

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class Unbuffered:
    def __init__(self, stream, subfix):
        self.txt_stream = open("log_" + subfix + ".txt",
                               "w")  # File where you need to keep the logs
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        # self.stream.flush()
        self.txt_stream.write(
            data)  # Write the data of stdout here to a text file as well

    def flush(self):
        self.stream.flush()


if __name__ == '__main__':
    sys.stdout = Unbuffered(sys.stdout, "nn")
    cuda = torch.device("cuda:0")
    # Reset random number seed

    # Specify the learning parameters
    epochs = 200
    learning_rate = 0.0001
    batch_size = 512
    dropout = 0.5
    # Generate data
    #####################

    X_train = np.load("data/X_train_no_ext.npy")
    y_train = np.load("data/y_train_no_ext.npy").reshape((-1, 1))
    # weights_train = y_train.sum() / float(y_train.shape[0]) * (1 - y_train) + (
    #     1 - y_train.sum() / float(y_train.shape[0])) * y_train
    weights_train = np.ones(y_train.shape, dtype='float32') / float(
        y_train.shape[0])
    X_valid = np.load("data/X_valid.npy")
    y_valid = np.load("data/y_valid.npy").reshape((-1, 1))
    # weights_valid = y_valid.sum() / float(y_valid.shape[0]) * (1 - y_valid) + (
    #     1 - y_valid.sum() / float(y_valid.shape[0])) * y_valid
    weights_valid = np.ones(y_valid.shape, dtype='float32') / float(
        y_valid.shape[0])
    # importance = np.load("data/importance.npy", allow_pickle=True)
    # ind = np.where(importance < 1e-10)
    # X_train = np.delete(X_train, ind, axis=1)
    # X_valid = np.delete(X_valid, ind, axis=1)
    feature_dim = X_train.shape[1]
    # Calculate shapes
    print("Input feature dimension %d, training number of samples %d" %
          (X_train.shape[1], X_train.shape[0]))

    X_train = torch.from_numpy(X_train).cuda()
    y_train = torch.from_numpy(y_train).cuda()
    weights_train = torch.from_numpy(weights_train).cuda()
    X_valid = torch.from_numpy(X_valid).cuda()
    y_valid = torch.from_numpy(y_valid).cuda()
    weights_valid = torch.from_numpy(weights_valid).cuda()
    torch.manual_seed(0)
    np.random.seed(0)

    ########
    # Convert the data to tensors
    train = data_utils.TensorDataset(X_train, y_train, weights_train)
    # Create an iteratable dataset for the training partitiong, using the batch size
    train_loader = data_utils.DataLoader(train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=False)
    whole_loader = data_utils.DataLoader(train,
                                         batch_size=X_train.shape[0],
                                         pin_memory=False)
    del X_train, y_train, train
    # Initialize model and move all parameters to CUDA
    model = NN(feature_dim, 1, dropout)
    model.cuda(cuda)
    print("total number of trainable parameters: ", model.get_n_params())
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 amsgrad=False)

    # Initialize iteration parameters
    start_time = time.time()
    number_of_batches = len(train_loader)  # For averaging the test loss

    for t in range(epochs):
        avg_loss = 0
        model.train()
        for batch in train_loader:
            y_pred = model(
                batch[0].cuda())  # + 0.01 * torch.randn_like(batch[0]))
            loss = model.loss(y_pred, batch[1].cuda(), batch[2].cuda())
            # y_pred = model(X_valid)
            # loss = model.loss(y_pred, y_valid)
            # Backpropagation of loss
            # Zero the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_loss /= number_of_batches
        # Enter evaluation mode
        model.eval()
        with torch.no_grad():
            yhat_valid = model(X_valid)
            eval_loss = model.loss(yhat_valid, y_valid, weights_valid).item()
            valid_auc = roc_auc_score(y_valid.detach().cpu().numpy(),
                                      yhat_valid.detach().cpu().numpy())
            del yhat_valid
            # for batch in whole_loader:
            #     yhat_valid = model(batch[0].cuda())
            #     train_auc = roc_auc_score(batch[1].detach().cpu().numpy(),
            #                               yhat_valid.detach().cpu().numpy())
            #     del yhat_valid
        # Early stopping update
        gc.collect()
        print(
            "Epoch {0}: Average Train Loss {1}, Evaluation Loss {2}, Evaluation Auc {3}"
            .format(t, avg_loss, eval_loss, valid_auc))

        # Early Stopping condition
        if model.early_stop(-valid_auc) is True:
            print("Early Stopping at epoch %d" % t)
            break
    model.retain_best_state()  # recover the best model during training
    print(
        "Training complete for epochs=%d, learning rate=%f, batchsize=%d, dropout=%f"
        % (epochs, learning_rate, batch_size, dropout))
    print("Time elapsed: ", time.time() - start_time)

    # # Move the model to cpu()
    model.cpu()
    torch.save(model, "data/NN_model.pt")
