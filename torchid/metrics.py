import numpy as np
from diffutil.products import jvp, unflatten_like


def r_squared(y_true, y_pred, time_axis=0):
    """ Computes the R-square index.

    The R-squared index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    r_squared_val : np.array
        Array of r_squared value.
    """

    SSE = np.sum((y_pred - y_true)**2, axis=time_axis)
    y_mean = np.mean(y_true, axis=time_axis, keepdims=True)
    SST = np.sum((y_true - y_mean)**2, axis=time_axis)

    return 1.0 - SSE/SST


def error_rmse(y_true, y_pred, time_axis=0):
    """ Computes the Root Mean Square Error (RMSE).

    The RMSE index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    RMSE : np.array
        Array of r_squared value.

    """

    SSE = np.mean((y_pred - y_true)**2, axis=time_axis)
    RMSE = np.sqrt(SSE)
    return RMSE


def error_nrmse(y_true, y_pred, time_axis=0):
    """ Computes the Normalized Root Mean Square Error (NRMSE).

    The NRMSE index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    NRMSE : np.array
        Array of r_squared value.

    """

    SSE = np.mean((y_pred - y_true)**2, axis=time_axis)
    RMSE = np.sqrt(SSE)
    NRMSE = RMSE/np.std(y_true, axis=time_axis)
    return NRMSE


def error_mean(y_true, y_pred, time_axis=0):
    """ Computes the error mean value.

    The RMSE index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    e_mean : np.array
        Array of error means.
    """

    e_mean = np.mean(y_true - y_pred, axis=time_axis)
    return e_mean


def error_mae(y_true, y_pred, time_axis=0):
    """ Computes the error Mean Absolute Value (MAE)

    The RMSE index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    e_mean : np.array
        Array of error mean absolute values.
    """

    e_mean = np.mean(np.abs(y_true - y_pred), axis=time_axis)
    return e_mean

def fit_index(y_true, y_pred, time_axis=0):
    """ Computes the per-channel fit index.

    The fit index is commonly used in System Identification. See the definitionin the System Identification Toolbox
    or in the paper 'Nonlinear System Identification: A User-Oriented Road Map',
    https://arxiv.org/abs/1902.00683, page 31.
    The fit index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    fit_val : np.array
        Array of r_squared value.

    """

    err_norm = np.linalg.norm(y_true - y_pred, axis=time_axis, ord=2)  # || y - y_pred ||
    y_mean = np.mean(y_true, axis=time_axis)
    err_mean_norm = np.linalg.norm(y_true - y_mean, ord=2)  # || y - y_mean ||
    fit_val = 100*(1 - err_norm/err_mean_norm)

    return fit_val

def r_squared_theta(theta, states, actions, model):
        """ Computes R^2 index between measured output and jvp output
            for a given parameter vector theta
    
        Arguments:
        theta (torch.tensor): output of adaptation of the nominal model
        states (torch.tensor): true states tensor [time, traj, n_feat]
        actions (torch.tensor): actions tensor [time, traj, n_feat]
        model (StateSpaceWrapper): NN model

        """
        theta_lin_f = unflatten_like(theta, tensor_lst=list(model.parameters()))
        states_pred = model(actions)
        states_pred = states_pred.view(states.shape[0], -1, states.shape[2])
        states_lin = states_pred + jvp(states_pred, model.parameters(), theta_lin_f)[0]
        states = states.detach().numpy()[:,0,:]
        states_lin = states_lin.detach().numpy()[:,0,:]
        R_sq_theta = r_squared(states, states_lin, time_axis=0)
        
        return R_sq_theta


if __name__ == '__main__':
    N = 20
    ny = 2
    SNR = 10
    y_true = SNR*np.random.randn(N, 2)
    y_pred = np.copy(y_true) + np.random.randn(N, 2)
    err_rmse_val = error_rmse(y_pred, y_true)
    r_squared_val = r_squared(y_true, y_pred)
    fit_val = fit_index(y_true, y_pred)

    print(f"RMSE: {err_rmse_val}")
    print(f"R-squared: {r_squared_val}")
    print(f"fit index: {fit_val}")
