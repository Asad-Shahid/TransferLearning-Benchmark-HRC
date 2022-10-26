import os
import numpy as np
import torch
import torch.nn as nn
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
import matplotlib.pyplot as plt

if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    model_name = "ssmodel_exp"
    model_filename = f"{model_name}.pt"

    # Prepare model, loss and optimizer
    ss_model = NeuralStateSpaceModel(n_x=6, n_u=20, n_feat=25, init_small=True)
    ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))
    loss_fn = nn.MSELoss()

    # Load the data
    u_test = np.load(os.path.join("data_resampled", "test", "u_175.npy")).astype(np.float32) 
    y_test = np.load(os.path.join("data_resampled", "test", "y_175.npy")).astype(np.float32)
    

    u_test = torch.tensor(u_test)
    y_test = torch.tensor(y_test)

    u_test = u_test.reshape(1, -1, 20).transpose(0,1)
    y_test = y_test.reshape(1, -1, 6).transpose(0,1)

    # Simulate the trajectory 
    ss_simulator = ForwardEulerSimulator(ss_model)
    y_test = y_test[:,0,:]
    u_test = u_test[:,0,:]
    x_0 = y_test[0, :].clone().detach() # initial state
    y_pred = ss_simulator(x_0, u_test)

    # Extarct one test trajectory in particular
    mpe_traj = loss_fn(y_pred, y_test).detach().numpy()
    y_test = y_test.detach().numpy()
    y_pred = y_pred.detach().numpy()

    fig = plt.figure(figsize=(6, 5))
    plt.margins(0,0)
    # plt.suptitle("TEST MPE " + str(mpe_traj))
    plt.subplot(321)
    plt.plot(y_test[:, 0], label='Measured')
    plt.plot(y_pred[:, 0], label='Predicted')
    plt.ylabel(r"$\mathit{F_x}$ [N]")
    plt.tick_params(labelbottom=False)
    plt.subplot(323)
    plt.plot(y_test[:, 1], label='Measured')
    plt.plot(y_pred[:, 1], label='Predicted')
    plt.ylabel(r"$\mathit{F_y}$ [N]")
    plt.tick_params(labelbottom=False)
    plt.subplot(325)
    plt.plot(y_test[:, 2], label='Measured')
    plt.plot(y_pred[:, 2], label='Predicted')
    plt.ylabel(r"$\mathit{F_z}$ [N]")
    plt.xlabel("time [samples]")
    plt.subplot(322)
    plt.plot(y_test[:, 3], label='Measured')
    plt.plot(y_pred[:, 3], label='Predicted')
    # fig.text(0.49, 0.77, r"$\mathit{\tau_x}$ [Nm]", va='center', rotation='vertical')
    plt.ylabel(r"$\mathit{\tau_x}$ [Nm]")
    plt.tick_params(labelbottom=False)
    plt.subplot(324)
    plt.plot(y_test[:, 4], label='Measured')
    plt.plot(y_pred[:, 4], label='Predicted')
    plt.ylabel(r"$\mathit{\tau_y}$ [Nm]")
    plt.tick_params(labelbottom=False)
    plt.subplot(326)
    plt.plot(y_test[:, 5], label='Measured')
    plt.plot(y_pred[:, 5], label='Predicted')
    plt.ylabel(r"$\mathit{\tau_z}$ [Nm]")
    plt.xlabel("time [samples]")
    plt.tight_layout()


    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper right')
    plt.show()

    # fig.savefig('predictions.png', dpi=200, bbox_inches='tight')