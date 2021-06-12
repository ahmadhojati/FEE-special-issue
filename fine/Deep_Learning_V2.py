#!/usr/bin/env python
# coding: utf-8

# # Deep Learning

# In[1]:


# ## This code is an attempt to estimate snow depth using topographical features and vegetation structure.
# I'm trying to develop the results for Grand Mesa, Colorado, USA.The features I use are ground elevation, slope, aspect,
# canopy percent cover, canopy height and foliage height diversity at 0.5, 1 and 2 m voxel size. The data set is in 1m resolution and I have 952 images.
# Image dimension is 250*250*9, which the first band of the image is snow depth and other 8 bands are elevation, aspect, 
# slope, canopy percent cover, canopy height, and FHD0.5, FHD1.0 and FHD 2.0 respectively. I prepared the dataset to be in 9 bands which you can have 
# find in my data folder.  Airborne lidar data products are from the NASA Airborne Snow Observatory (ASO) and Quantum 
# collected on two campaigns during 2016-2017 and 2019-2020. 
# 


# In[2]:


# # To run this code you need to load some modules and install several packages if they are not installed on your system:

# module load cuda10.0/toolkit/10.0.130
# module load python36
# module load gdal/gcc8/3.0.4
# pip install --user osgeo
# pip3 install --global-option=build_ext --global-option="-I/cm/shared/apps/gdal/gcc8/3.0.4/include" GDAL==3.0.4 --user
# pip install --user sklearn
# pip install --user tensorflow==1.15
# pip install --user keras==2.2.4
# pip install --user opencv-python
# pip install --user pandas
# pip install --user progressbar2
# pip install --user h5py==2.10.0
# pip install --user seaborn

# Note: If the system says the requisits are already satisfied but python does not detect the libraries, 
#     add "--ignore-installed" to the installation line.


# ## Load packages

# In[3]:


from os.path import abspath
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d
import tensorflow as tf
import os
import copy
import h5py
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers, models
import IPython
import numpy as np
import scipy as sp
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
# import feature_reconstruction as fs
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from scipy import signal, optimize
import math
import pwlf


# # Data Preparation

# ## Read the image file

# In[4]:


filepath = r"data/SnwDpth_Grnd_Asp_Slp_CPC_CHM_FHD512.tif"
# filepath = r"../Data/test.tif"
dataset = gdal.Open(filepath, gdal.GA_ReadOnly) 


# ### Show number of bands and size of the image

# In[5]:


dataset.RasterCount, dataset.RasterXSize, dataset.RasterYSize


# In[6]:


num_layers = dataset.RasterCount
Npix = dataset.RasterXSize
Nlines = dataset.RasterYSize


# ## Convert gdal file to array

# In[7]:


rasterArray = dataset.ReadAsArray()


# ### swap the dimension

# In[8]:


swapped = np.moveaxis(rasterArray, 0, 2)
swapped.shape


# ### Filter negative snow depth

# In[9]:


swapped[swapped[:,:,0] < 0,0] = 0         # Snow depth
swapped[swapped[:,:,1] < 0 ,1] = 0         # Elevation
swapped[swapped[:,:,4] < 0 ,4] = 0         # CCP
swapped[swapped[:,:,5] < 0 ,5] = 0         # CHM
swapped[swapped[:,:,6] < 0 ,6] = 0         # FHD 0.5m
swapped[swapped[:,:,7] < 0 ,7] = 0         # FHD 1m
swapped[swapped[:,:,8] < 0 ,8] = 0         # FHD 2m


# ## Split the image into multiple images

# ### Split Function

# In[10]:


def split(arr, nrows, ncols):
    """Split a matrix into sub-matrices."""

    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


# ### Split

# In[11]:


win_len = 250                       # Columns
win_width = 250                     # Rows 
num_imgs = int((Npix/win_len)*(Nlines/win_width))

data = np.empty((num_imgs, win_len, win_width,num_layers))
for i in range(num_layers):
    data[:,:,:,i] = split(swapped[:,:,i],win_len,win_width)


# In[12]:


print('The shape of data is:',data.shape)


# In[ ]:


resolution = 1


# ### Power Spectral Density function

# In[ ]:


### This function computes the power spectral density (psd) for East-West and North-South directions and returns  
### the frequency and psd.
### The inputs are:
### "imarray": image array file.
### "direction": 'N' for North-South and 'E' for East-West direction.

def psd_(imarray,direction):
    psd0 = np.zeros(int(len(imarray)/2)+1)
    if direction == 'N':
        for i in range(0,len(imarray)):
            freqs, psd = signal.welch(imarray[:,i],nperseg=len(imarray))
            psd0 = psd + psd0
    elif direction == 'E':
        for j in range(0,len(imarray)):
            freqs, psd = signal.welch(imarray[j,:],nperseg=len(imarray))
            psd0 = psd + psd0
    return psd0/len(imarray), freqs    


# ### Breaks function

# In[ ]:


### This function fits the powerlaw to the data, finds breaks, and plots the fitted lines along side the breakpoint.
### The inputs are:
### "X": frequencies from the power spectral density function.
### "Y": psd from the power spectral density function.
### "resolution" : image resolution.
### "npixel" : lenght of the square image.
### "arg" : title for the plot.
### "S": If S==1 it saves the plot

def segment_plot(X, Y, resolution, npixel, arg, S):
    # Extracting the break point. 
    my_pwlf = pwlf.PiecewiseLinFit(np.log10(X[1:]), np.log10(Y[1:]))
    breaks = my_pwlf.fit(2)[1]
    
    # The frequency at breakpoint
    f_x = np.power(10,breaks)

    # Find the slope (powerlaw) and intercept of the lines fitted on the psd plot.
    # Broken powerlaw
    if len(X[X<f_x]) > 1 and len(X[X>f_x] > 1):                                     
        # Frequencies less than the frequency at the breakpoint
        y0 = Y[0:len(X[X<f_x])+1]
        x0 = X[0:len(X[X<f_x])+1]
        p0 = np.polyfit(np.log(x0[1:]), np.log(y0[1:]), 1) # Fit a line to psds
        z0 = np.polyval(p0,np.log(x0[1:]))
        B0 = -p0[0]                                        # First Power Law
#         print('\u03B2_0:', B0)

        # Frequencies greater than the frequency at the breakpoint
        y = Y[len(X[X<f_x])-1:]
        x = X[len(X[X<f_x])-1:]
        p = np.polyfit(np.log(x), np.log(y), 1)            # Fit a line to psds
        z = np.polyval(p,np.log(x))
        B1 = -p[0]                                         # Second Power Law
#         print('\u03B2_1:', B1)
        
        # Converting the breakpoint from frequency into meters
#         brk = 1/f_x * resolution *np.max(X)                
        brk = 1/f_x
        
        # psd and powerlaw fit log-log Plot
        plt.figure(figsize=(12, 6))
        plt.loglog(X[1:], Y[1:],'k')                       # psd log-log plot
        plt.loglog(x0[1:],np.exp(z0),'--b')                # log-log plot of the first powerlaw fit 
        plt.loglog(x,np.exp(z),'--b')                      # log-log plot of the second powerlaw fit 
        plt.title(arg[0],fontsize=15)
        plt.xlabel('k',fontsize=15)
        plt.ylabel('Power Spectral Density',fontsize=15)

        # Show the breakpoint in the plot with a vertical line
        if B1>0 and B1>B0:
            plt.axvline(f_x,ls = '-.',c = 'r')           
            plt.text(.85*f_x,np.mean(np.exp(z)), "{:.1f} m".format(brk) , fontsize=20, rotation = 90)
            plt.text(x0[2], np.median((y0)), "\u03B2 = {:.2f}".format(B0) , fontsize=20)
            plt.text(np.mean(x), np.mean((y)), "\u03B2 = {:.2f}".format(B1) , fontsize=20)
            
        # If it does not follow a powerlaw, print "None" for the breakpoint and betas    
        else:
            brk = 0
            B0 = None
            B1 = None    
#         print('Break in meters:',brk)
#         print('----------------------')
        
    # If only one powerlaw fits the data        
    else:
        y = Y[1:]
        x = X[1:]
        p = np.polyfit(np.log(x), np.log(y), 1)               # Fit a line to psd
        z = np.polyval(p,np.log(x))
        
        # Plot the psd log-log with the powerlaw fit
        plt.figure(figsize=(12, 6))
        plt.loglog(X[1:], Y[1:],'k')                         # psd log-log plot
        plt.loglog(x,np.exp(z),'--b')                        # log-log plot of the powerlaw fit
        plt.title(arg[0],fontsize=15)
        plt.xlabel('k',fontsize=15)
        plt.ylabel('Power Spectral Density',fontsize=15)
        B0 = None
        B1 = None
        brk = 0
    
    # Save the plot    
    if S==1:
        plt.savefig('{}_{}.png'.format(arg[0],arg[1]),dpi=300)  
        
    return B0, B1, brk


# ### Round to odd function

# In[ ]:


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1



# ### Check if the crop is correct

# In[13]:


title = ['Cropped Snow Depth','Original Snow Depth']
plt.figure(figsize=(10,10))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.grid(False)
    plt.title(title[i], fontsize=16)
    if i==0:
        plt.imshow(data[0,:,:,0])
    else:
        plt.imshow(swapped[0:win_len,0:win_len,0])  
plt.savefig('Cropped_vs_Original_Snowdepth.png',dpi=300)
# plt.show()


# ## Plot all the layers

# In[15]:


title = ['Snow Depth','Digital Elevation Model','Aspect', 'Slope', 'Canopy Percent Cover','Canopy Height Model','FHD_05', 'FHD_1', 'FHD_2']
# data_masked = np.ma.masked_where(data == 0, data)
img_id = 1
plt.figure(figsize=(20,15))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.grid(False)
    plt.title(title[i], fontsize=16)
    plt.imshow(data[img_id,:,:,i])
plt.savefig('Layers.png',dpi=300)


# In[16]:


del swapped 
del rasterArray
del dataset 


# # Modeling

# ## Train, Test and Validation

# ### Train and test split function

# In[17]:


def train_test(X,Y,spl, seed):
    N = len(X)
    sample = int(spl*N)
    np.random.seed(seed)
    idx = np.random.permutation(N)  
    train_idx, test_idx = idx[:sample], idx[sample:]
    x_train, x_test, y_train, y_test = X[train_idx,:,:,:], X[test_idx,:,:,:],Y[train_idx,:,:], Y[test_idx,:,:]
    
    return x_train, x_test, y_train, y_test


# ### Train and Test

# In[18]:


spl = 0.7
x_train, x_test, y_train, y_test = train_test(data[:,:,:,1:],data[:,:,:,0], spl, 123)


# In[19]:


del data


# ### Standardize train and test

# In[20]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, num_layers-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(-1, num_layers-1)).reshape(x_test.shape)


# ### Train and Validation

# In[21]:


x_train, x_val, y_train, y_val = train_test(x_train,y_train, 0.8, 123)


# ### Outputs for Train, Test and Validation

# In[22]:


y_train = y_train.reshape(y_train.shape[0], win_len, win_len, 1)
y_test = y_test.reshape(y_test.shape[0], win_len, win_len, 1)
y_val = y_val.reshape(y_val.shape[0], win_len, win_len, 1)


# ## Network

# ### Defining $R^2$ Function for accuracy assessment

# In[23]:


def det_coeff(y_true, y_pred):
    SS_res =  tf.keras.backend.sum(tf.keras.backend.square( y_true-y_pred ))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square( y_true - tf.keras.backend.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + tf.keras.backend.epsilon()) )


# ### Network Architecture

# In[24]:


optimizer = 'adam'

layer_0 = tf.keras.layers.Input(x_train.shape[1:])
    
# First Convolution layer with padding
model_0 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')(layer_0)
    
# Second Convolution layer with padding
model_0 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')(model_0)

#model_0 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu')(model_0)

    
# Batch Normalization
model_0 = tf.keras.layers.BatchNormalization(axis=-1)(model_0)
    
# %20 Dropout
model_0 = tf.keras.layers.SpatialDropout2D(0.2)(model_0)
    
# Third Convolution layer with padding
model_0 = tf.keras.layers.Conv2D(filters = 64, kernel_size=3, padding = 'same',activation = 'relu')(model_0)

#model_0 = tf.keras.layers.Conv2D(filters = 128, kernel_size = 5, padding = 'same', activation = 'relu')(model_0)


# %20 Dropout
model_0 = tf.keras.layers.SpatialDropout2D(0.2)(model_0)
    
# Forth Convolution layer with padding
model_0 = tf.keras.layers.Conv2D(filters = 100, kernel_size = 3, padding = 'same',activation = 'relu')(model_0)

# %20 Dropout
model_0 = tf.keras.layers.SpatialDropout2D(0.2)(model_0) 
    
# Shortcut layer (Fifth Convolution layer with padding)
shortcut_x = tf.keras.layers.Conv2D(filters = 100, kernel_size = 3, padding = 'same')(layer_0)

# Add shortcut to the convolution result
model_0 = tf.keras.layers.Add()([shortcut_x, model_0])
    
# Activation on new convolution layer
model_1 = tf.keras.layers.Activation('relu')(model_0)

# Sixth Convolution layer with padding
model_2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 1, padding = 'same')(model_1)

# Create model from inputs and outputs
model = tf.keras.models.Model(inputs = layer_0, outputs = model_2)

#COMPILE THE MODEL
if(optimizer=='adam'):
    optim = tf.keras.optimizers.Adam(lr = 0.001)
else:
    optim = tf.keras.optimizers.SGD(lr = 0.0001)
    
# Compile the model
model.compile(loss = tf.keras.losses.mean_squared_error,optimizer = optim, metrics = [det_coeff])

# Show the model structure
print(model.summary())


# ## Training 

# ### Train and Save the model parameters

# In[26]:


epochs = 200

# Callback to Keras to save best model weights
best_weights="models/best_model_Grand_Mesa.h5"
model_save = tf.keras.callbacks.ModelCheckpoint(best_weights
                                                , monitor = 'val_loss'
                                                , verbose = 1
                                                , save_best_only = True
                                                , save_weights_only = True
                                                , mode='min')

# Fit the model
hist = model.fit(x =  x_train, y = y_train, epochs = epochs, validation_data = (x_val, y_val),
                 verbose = 1, callbacks = [model_save])


# ### Plot the train and validation performance

# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(hist.history['loss'],label='train')
plt.plot(hist.history['val_loss'],label='validation')
plt.legend()
plt.title('Mean Squared Error', fontsize=12)
plt.ylabel('MSE',fontsize=12)
plt.xlabel('Epochs', fontsize=12);  
plt.savefig('Train_MSE_vs_Validation_MSE.png',dpi=300)


# ### Save the model

# In[27]:


model_json = model.to_json()
with open("models/best_model_Grand_Mesa.json", "w") as json_file:
    json_file.write(model_json)


# ### Get the weights for the first convolution layer

# In[28]:


#Conv1 
conv1_kernels = model.layers[1].get_weights()[0]
print('conv1_kernels.shape:{}'.format(conv1_kernels.shape))


# ### Plot the Conv Weights of  the First Layer

# In[29]:


fig, axes = plt.subplots(2, 10, figsize=(15,3), subplot_kw={'xticks': [], 'yticks': []})
axes = axes.flatten()
for i in range(len(axes)):
    axes[i].imshow(conv1_kernels[:,:,0,i], cmap='gray')
    axes[i].set_title('Map'+str(i+1), fontsize=8)
    
#save the figure    
plt.savefig('Conv_weights.png',dpi=300)


# ## Predict

# ### Load json and create model

# In[30]:


json_file = open("models/best_model_Grand_Mesa.json",'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/best_model_Grand_Mesa.h5")


# ### Predict for the Test data and print the $R^2$ 

# In[31]:


loaded_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.mean_squared_error, metrics = [det_coeff])

print('MSE and R-squared for Test are:')
loaded_model.evaluate(x_test, y_test, batch_size=len(x_test),verbose=2)


# ### Train and Validation $R^2$

# In[32]:


# Train
print('MSE and R-squared for Trian are:')
loaded_model.evaluate(x_train, y_train, batch_size=len(x_train),verbose=2)

# Validation
print('MSE and R-squared for Validation are:')
loaded_model.evaluate(x_val, y_val, batch_size=len(x_val),verbose=2)


# ### Prediction values

# In[33]:


y_prd = loaded_model.predict(x_test).flatten()


# ### Plot the Predicted snow depth vs Original snow depth

# In[ ]:


title = ['Original Snow Depth','Predicted Snow Depth','Difference']
img_id = 159

# Reshape the y_prd to 31*31 images
y_prd_reshape =  y_prd.reshape(y_test.shape)

print('R-square for image {} is:'.format(img_id))
print(metrics.r2_score(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))
print('-------------------------------')

print('Mean Absolute Error for image {} is:'.format(img_id))
print(metrics.mean_absolute_error(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))

vmin = y_test[img_id,:,:,0].min()
vmax = y_test[img_id,:,:,0].max()

fig, ax = plt.subplots(1, 3,figsize=(15,15))
for i in range(3):
    if i == 0:
        im = ax[i].imshow(y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    elif i==1:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    else:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0]-y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.4, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('Original_vs_Predicted_Snowdepth1.png',dpi=300)    



title = ['Original Snow Depth','Predicted Snow Depth','Difference']
img_id = 23

# Reshape the y_prd to 31*31 images
y_prd_reshape =  y_prd.reshape(y_test.shape)

print('R-square for image {} is:'.format(img_id))
print(metrics.r2_score(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))
print('-------------------------------')

print('Mean Absolute Error for image {} is:'.format(img_id))
print(metrics.mean_absolute_error(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))

vmin = y_test[img_id,:,:,0].min()
vmax = y_test[img_id,:,:,0].max()

fig, ax = plt.subplots(1, 3,figsize=(15,15))
for i in range(3):
    if i == 0:
        im = ax[i].imshow(y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    elif i==1:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    else:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0]-y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.4, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('Original_vs_Predicted_Snowdepth2.png',dpi=300)    



title = ['Original Snow Depth','Predicted Snow Depth','Difference']
img_id = 67

# Reshape the y_prd to 31*31 images
y_prd_reshape =  y_prd.reshape(y_test.shape)

print('R-square for image {} is:'.format(img_id))
print(metrics.r2_score(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))
print('-------------------------------')

print('Mean Absolute Error for image {} is:'.format(img_id))
print(metrics.mean_absolute_error(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))

vmin = y_test[img_id,:,:,0].min()
vmax = y_test[img_id,:,:,0].max()

fig, ax = plt.subplots(1, 3,figsize=(15,15))
for i in range(3):
    if i == 0:
        im = ax[i].imshow(y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    elif i==1:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    else:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0]-y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.4, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('Original_vs_Predicted_Snowdepth3.png',dpi=300)    



title = ['Original Snow Depth','Predicted Snow Depth','Difference']
img_id = 98

# Reshape the y_prd to 31*31 images
y_prd_reshape =  y_prd.reshape(y_test.shape)

print('R-square for image {} is:'.format(img_id))
print(metrics.r2_score(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))
print('-------------------------------')

print('Mean Absolute Error for image {} is:'.format(img_id))
print(metrics.mean_absolute_error(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))
vmin = y_test[img_id,:,:,0].min()
vmax = y_test[img_id,:,:,0].max()

fig, ax = plt.subplots(1, 3,figsize=(15,15))
for i in range(3):
    if i == 0:
        im = ax[i].imshow(y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    elif i==1:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    else:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0]-y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.4, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('Original_vs_Predicted_Snowdepth4.png',dpi=300) 


title = ['Original Snow Depth','Predicted Snow Depth','Difference']
img_id = 209

# Reshape the y_prd to 31*31 images
y_prd_reshape =  y_prd.reshape(y_test.shape)

print('R-square for image {} is:'.format(img_id))
print(metrics.r2_score(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))
print('-------------------------------')

print('Mean Absolute Error for image {} is:'.format(img_id))
print(metrics.mean_absolute_error(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))

vmin = y_test[img_id,:,:,0].min()
vmax = y_test[img_id,:,:,0].max()

fig, ax = plt.subplots(1, 3,figsize=(15,15))
for i in range(3):
    if i == 0:
        im = ax[i].imshow(y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    elif i==1:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    else:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0]-y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.4, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('Original_vs_Predicted_Snowdepth5.png',dpi=300) 



title = ['Original Snow Depth','Predicted Snow Depth','Difference']
img_id = 270

# Reshape the y_prd to 250*250 images
y_prd_reshape =  y_prd.reshape(y_test.shape)

print('R-square for image {} is:'.format(img_id))
print(metrics.r2_score(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))
print('-------------------------------')

print('Mean Absolute Error for image {} is:'.format(img_id))
print(metrics.mean_absolute_error(y_test[img_id].flatten(), y_prd_reshape[img_id].flatten()))

vmin = y_test[img_id,:,:,0].min()
vmax = y_test[img_id,:,:,0].max()

fig, ax = plt.subplots(1, 3,figsize=(15,15))
for i in range(3):
    if i == 0:
        im = ax[i].imshow(y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    elif i==1:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
    else:
        im = ax[i].imshow(y_prd_reshape[img_id,:,:,0]-y_test[img_id,:,:,0],vmin = vmin, vmax = vmax)
        ax[i].set_title(title[i], fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.4, 0.02, 0.2])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('Original_vs_Predicted_Snowdepth6.png',dpi=300) 



# ### Feature Importance

def feature_importance_(X,y,loaded_model):
    np.random.seed(42)
    permuted_train_test = copy.deepcopy(X)
    MSE_R_2 = np.empty((permuted_train_test.shape[3],2))

    # --------------------------- #
    # Iterate over the Variables  #
    # --------------------------- # 
    for variable in range(permuted_train_test.shape[3]):
        permuted_train_test = copy.deepcopy(X)

        # ----------------------- #
        # Iterate over the Images #
        # ----------------------- #
        for img_idx in range(len(permuted_train_test)):
            # ----------------------- #
            # Permute the Feature     #
            # ----------------------- #
            np.apply_along_axis(
                np.random.shuffle
                ,axis=-1
                ,arr=permuted_train_test[img_idx,:,:,variable])

        MSE_R_2[variable] = loaded_model.evaluate(permuted_train_test, y,
                                                     batch_size=len(permuted_train_test),verbose=0)
                                                   
    return MSE_R_2


# ### Feature importance plot function

def plot_feature_importance(MSE_R_2,label, arg):
    
    X = np.arange(MSE_R_2.shape[0])
    fig = plt.figure(figsize=(20,10))
    plt.bar(X + 0.00, MSE_R_2[:,0], color = 'b', width = 0.25, label = 'MSE')
    plt.bar(X + 0.25, MSE_R_2[:,1], color = 'g', width = 0.25, label = 'R-squared')
    objects = label
    plt.xticks(X, objects,rotation=15, size=18)
    plt.legend(prop={'size':24})
    plt.savefig('{}.png'.format(arg),dpi=300) 


MSE_R_2_test =  feature_importance_(x_test,y_test,loaded_model)
MSE_R_2_train =  feature_importance_(x_train,y_train,loaded_model)
MSE_R_2_val =  feature_importance_(x_val,y_val,loaded_model)


label = ('Digital Elevation Model','Aspect', 'Slope', 'Canopy Percent Cover','Canopy Height Model', 
             'FHD_05', 'FHD_1', 'FHD_2') 



plot_feature_importance(MSE_R_2_test,label, 'Feature_importance_test')
plot_feature_importance(MSE_R_2_train,label, 'Feature_importance_train')
plot_feature_importance(MSE_R_2_val,label, 'Feature_importance_validation')

