#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
from scipy.optimize import curve_fit


def split(arr, nrows, ncols):
    """
    Split a matrix into sub-matrices.

    Parameters:
    - arr: Input matrix to be split.
    - nrows: Number of rows for each sub-matrix.
    - ncols: Number of columns for each sub-matrix.

    Returns:
    - result: List of sub-matrices.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisible by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisible by {}".format(w, ncols)

    # Reshape the input matrix into sub-matrices
    result = (arr.reshape(h // nrows, nrows, -1, ncols)
                  .swapaxes(1, 2)
                  .reshape(-1, nrows, ncols))

    return result


def psd_(imarray, direction):
    """
    This function computes the power spectral density (psd) for East-West and North-South directions and returns
    the frequency and psd.

    Parameters:
    - imarray: Image array file.
    - direction: 'N' for North-South and 'E' for East-West direction.

    Returns:
    - psd: Power spectral density.
    - freqs: Frequency.
    """
    psd0 = np.zeros(int(len(imarray) / 2) + 1)

    if direction == 'N':
        # Compute psd for North-South direction
        for i in range(0, len(imarray)):
            freqs, psd = signal.welch(imarray[:, i], nperseg=len(imarray))
            psd0 = psd + psd0
    elif direction == 'E':
        # Compute psd for East-West direction
        for j in range(0, len(imarray)):
            freqs, psd = signal.welch(imarray[j, :], nperseg=len(imarray))
            psd0 = psd + psd0

    # Normalize psd by the length of the input array
    return psd0 / len(imarray), freqs


def segment_plot(X, Y, resolution, npixel, arg, S):
    """
    This function fits the powerlaw to the data, finds breaks, and plots the fitted lines alongside the breakpoint.

    Parameters:
    - X: Frequencies from the power spectral density function.
    - Y: PSD from the power spectral density function.
    - resolution: Image resolution.
    - npixel: Length of the square image.
    - arg: Title for the plot.
    - S: If S==1, it saves the plot.

    Returns:
    - B0: Slope of the first power law.
    - B1: Slope of the second power law.
    - brk: Frequency at the breakpoint.
    """

    # Extracting the break point.
    my_pwlf = pwlf.PiecewiseLinFit(np.log10(X[1:]), np.log10(Y[1:]))
    breaks = my_pwlf.fit(2)[1]
    
    # The frequency at breakpoint
    f_x = np.power(10, breaks)

    # Find the slope (powerlaw) and intercept of the lines fitted on the PSD plot.
    # Broken powerlaw
    if len(X[X < f_x]) > 1 and len(X[X > f_x]) > 1:                                     
        # Frequencies less than the frequency at the breakpoint
        y0 = Y[0:len(X[X < f_x]) + 1]
        x0 = X[0:len(X[X < f_x]) + 1]
        p0 = np.polyfit(np.log(x0[1:]), np.log(y0[1:]), 1)  # Fit a line to psds
        z0 = np.polyval(p0, np.log(x0[1:]))
        B0 = -p0[0]  # First Power Law

        # Frequencies greater than the frequency at the breakpoint
        y = Y[len(X[X < f_x]) - 1:]
        x = X[len(X[X < f_x]) - 1:]
        p = np.polyfit(np.log(x), np.log(y), 1)  # Fit a line to psds
        z = np.polyval(p, np.log(x))
        B1 = -p[0]  # Second Power Law
        
        # Converting the breakpoint from frequency into meters
        brk = 1 / f_x
        
        # psd and powerlaw fit log-log Plot
        plt.figure(figsize=(12, 6))
        plt.loglog(X[1:], Y[1:], 'k')  # psd log-log plot
        plt.loglog(x0[1:], np.exp(z0), '--b')  # log-log plot of the first powerlaw fit 
        plt.loglog(x, np.exp(z), '--b')  # log-log plot of the second powerlaw fit 
        plt.title(arg[0], fontsize=15)
        plt.xlabel('k', fontsize=15)
        plt.ylabel('Power Spectral Density', fontsize=15)

        # Show the breakpoint in the plot with a vertical line
        if B1 > 0 and B1 > B0:
            plt.axvline(f_x, ls='-.', c='r')           
            plt.text(.85 * f_x, np.mean(np.exp(z)), "{:.1f} m".format(brk), fontsize=20, rotation=90)
            plt.text(x0[2], np.median((y0)), "\u03B2 = {:.2f}".format(B0), fontsize=20)
            plt.text(np.mean(x), np.mean((y)), "\u03B2 = {:.2f}".format(B1), fontsize=20)
            
        # If it does not follow a powerlaw, print "None" for the breakpoint and betas    
        else:
            brk = 0
            B0 = None
            B1 = None
            
    # If only one powerlaw fits the data        
    else:
        y = Y[1:]
        x = X[1:]
        p = np.polyfit(np.log(x), np.log(y), 1)  # Fit a line to psd
        z = np.polyval(p, np.log(x))
        
        # Plot the psd log-log with the powerlaw fit
        plt.figure(figsize=(12, 6))
        plt.loglog(X[1:], Y[1:], 'k')  # psd log-log plot
        plt.loglog(x, np.exp(z), '--b')  # log-log plot of the powerlaw fit
        plt.title(arg[0], fontsize=15)
        plt.xlabel('k', fontsize=15)
        plt.ylabel('Power Spectral Density', fontsize=15)
        B0 = None
        B1 = None
        brk = 0
    
    # Save the plot    
    if S == 1:
        plt.savefig('{}_{}.png'.format(arg[0], arg[1]), dpi=300)  
    
    return B0, B1, brk


def round_up_to_odd(f):
    """
    Round a floating-point number up to the nearest odd integer.

    Parameters:
    - f: Input floating-point number.

    Returns:
    - Nearest odd integer.
    """
    return np.ceil(f) // 2 * 2 + 1


def train_test(X, Y, spl, seed):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - X: Input features.
    - Y: Target values.
    - spl: Proportion of the dataset to include in the training split (e.g., 0.8 for 80%).
    - seed: Seed for reproducibility.

    Returns:
    - x_train: Training set for input features.
    - x_test: Testing set for input features.
    - y_train: Training set for target values.
    - y_test: Testing set for target values.
    """
    N = len(X)
    sample = int(spl * N)
    np.random.seed(seed)
    idx = np.random.permutation(N)
    
    train_idx, test_idx = idx[:sample], idx[sample:]
    
    x_train, x_test = X[train_idx, :, :, :], X[test_idx, :, :, :]
    y_train, y_test = Y[train_idx, :, :], Y[test_idx, :, :]
    
    return x_train, x_test, y_train, y_test


def det_coeff(y_true, y_pred):
    """
    Calculate the coefficient of determination (R-squared) using TensorFlow backend functions.

    Parameters:
    - y_true: True target values.
    - y_pred: Predicted values.

    Returns:
    - R-squared value.
    """
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    
    # Avoid division by zero with a small epsilon value
    epsilon = tf.keras.backend.epsilon()
    r_squared = 1 - SS_res / (SS_tot + epsilon)
    
    return r_squared

def feature_importance_(X, y, loaded_model):
    """
    Evaluate feature importance based on Mean Squared Error (MSE) and R-squared.

    Parameters:
    - X: Input features.
    - y: Target values.
    - loaded_model: Trained machine learning model.

    Returns:
    - MSE_R_2: Array containing MSE and R-squared values.
    """
    np.random.seed(42)
    permuted_train_test = copy.deepcopy(X)
    MSE_R_2 = np.empty((permuted_train_test.shape[3], 2))

    # Iterate over the Variables
    for variable in range(permuted_train_test.shape[3]):
        permuted_train_test = copy.deepcopy(X)

        # Iterate over the Images
        for img_idx in range(len(permuted_train_test)):
            # Permute the Feature
            np.apply_along_axis(
                np.random.shuffle,
                axis=-1,
                arr=permuted_train_test[img_idx, :, :, variable]
            )

        MSE_R_2[variable] = loaded_model.evaluate(
            permuted_train_test, y, batch_size=len(permuted_train_test), verbose=0
        )

    return MSE_R_2

def plot_feature_importance(MSE_R_2, label, arg):
    """
    Plot the feature importance based on Mean Squared Error (MSE) and R-squared.

    Parameters:
    - MSE_R_2: Array containing MSE and R-squared values.
    - label: Label for the plot.
    - arg: Argument for saving the plot.

    Returns:
    None
    """
    X = np.arange(MSE_R_2.shape[0])
    fig = plt.figure(figsize=(20, 10))
    plt.bar(X + 0.00, MSE_R_2[:, 0], color='b', width=0.25, label='MSE')
    plt.bar(X + 0.25, MSE_R_2[:, 1], color='g', width=0.25, label='R-squared')
    objects = label
    plt.xticks(X, objects, rotation=15, size=18)
    plt.legend(prop={'size': 24})
    plt.savefig('{}.png'.format(arg), dpi=300)

