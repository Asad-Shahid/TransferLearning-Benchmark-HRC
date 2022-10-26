import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load the data
    u = np.load(os.path.join("data_resampled", "train", "u_2.npy")).astype(np.float32) 
    y = np.load(os.path.join("data_resampled", "train", "y_2.npy")).astype(np.float32)

    # Plot set-point and measured positions
    fig = plt.figure()
    y_label = ["x [m]", "y [m]", "z [m]"]
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(u[:, i], label='Set-point')
        plt.plot(u[:, i+7], label='Measured')
        plt.xlabel("time [samples]")
        plt.ylabel(y_label[i])
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper right')

    # Plot measured velocity of end-effector
    fig = plt.figure()
    y_label = ["$\mathit{v_x}$ [m/s]", "$\mathit{v_y}$ [m/s]", "$\mathit{v_z}$ [m/s]", "$\mathit{\omega_x}$ [rad/s]",
                "$\mathit{\omega_y}$ [rad/s]", "$\mathit{\omega_z}$ [rad/s]"] 
    for i in range(6):
        plt.subplot(str(321+i))
        plt.plot(u[:, i+14])
        plt.xlabel("time [samples]")
        plt.ylabel(y_label[i])
    plt.tight_layout()

    # Plot measured external wrench at the end-effector
    fig = plt.figure()
    y_label = ["$\mathit{F_x}$ [N]", "$\mathit{F_y}$ [N]", "$\mathit{F_z}$ [N]", "$\mathit{T_x}$ [Nm]",
                "$\mathit{T_y}$ [Nm]", "$\mathit{T_z}$ [Nm]"] 
    for i in range(6):
        plt.subplot(str(321+i))
        plt.plot(y[:, i])
        plt.xlabel("time [samples]")
        plt.ylabel(y_label[i])
    plt.tight_layout()
    plt.show()