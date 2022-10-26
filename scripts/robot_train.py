import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchid.datasets import SubsequenceDataset
from torch.utils.data import DataLoader
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Settings
    num_iter = 500
    lr = 1e-3 

    u = np.load(os.path.join("data_resampled", "train", "u_2.npy")).astype(np.float32) 
    y = np.load(os.path.join("data_resampled", "train", "y_2.npy")).astype(np.float32) 

    u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.40, random_state=42, shuffle=False)

    # Prepare train tensors
    u_train = torch.tensor(u_train.reshape(-1, 20))
    y_train = torch.tensor(y_train.reshape(-1, 6))  

    train_data = SubsequenceDataset(u_train, y_train, subseq_len=500)
    train_loader = DataLoader(train_data, batch_size=3000, shuffle=True)


    # Prepare model, loss and optimizer
    ss_model = NeuralStateSpaceModel(n_x=6, n_u=20, n_feat=25, init_small=True)
    ss_simulator = ForwardEulerSimulator(ss_model)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ss_model.parameters(), lr=lr)
    model_name = "ssmodel"
    model_filename = f"{model_name}.pt"
    # ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Training loop
    LOSS = []
    start_time = time.time()
    for itr in range(num_iter):

        train_loss = 0

        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1) # transpose to time_first

            # fit 
            batch_x0 = batch_y[0,:,:]
            batch_y_sim = ss_simulator(batch_x0, batch_u)
        
            loss = loss_fn(batch_y_sim, batch_y)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        # Reporting
        LOSS.append(train_loss)
        with torch.no_grad():
            print(f'Iter {itr} | Train Loss {train_loss:.5f}')


    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")


    plt.plot(LOSS)
    model_name = "ssmodel"
    model_filename = f"{model_name}.pt"
    torch.save(ss_model.state_dict(), os.path.join("models", model_filename))
    
    u_test = torch.tensor(u_test)
    y_test = torch.tensor(y_test)

    u_test = u_test.reshape(1, -1, 20).transpose(0,1)
    y_test = y_test.reshape(1, -1, 6).transpose(0,1)

    # Simulate the trajectory 
    traj_idx = 0
    ss_simulator = ForwardEulerSimulator(ss_model)
    y_test = y_test[:,traj_idx,:]
    u_test = u_test[:,traj_idx,:]
    x_0 = y_test[0, :].clone().detach() # initial state
    y_pred = ss_simulator(x_0, u_test)

    mpe_traj = loss_fn(y_pred, y_test).detach().numpy()
    y_test = y_test.detach().numpy()
    y_pred = y_pred.detach().numpy()

    fig = plt.figure()
    plt.suptitle("TEST MPE " + str(mpe_traj))

    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(y_test[:, i], label='True')
        plt.plot(y_pred[:, i], label='Fit')
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper right')
    plt.show()

    # fig.savefig('nominal_predictions.png', dpi=200, bbox_inches='tight')