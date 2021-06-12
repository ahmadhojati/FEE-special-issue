# FEE-special-issue

This is python code related to our paper in Forest Ecology and Environment journal.

The fine folder contains the main code to predict fine resolution snow depth by features that define forest structural diversity.
Features come from fine reolution (1m) airborne lidar data.


It contains "model" folder which is a destination for saving the deep learning model parameters and weights.
"output" folder is a folder to save the results of the model.
"data folder contains our main dataset having 9 bands, where the first layer is snow depth and other bands are ground elevation, slope, aspect,
canopy percent cover, canopy height and foliage height diversity at 0.5, 1 and 2 m voxel sizes.
The main code is called "Deep_Learning_V2.py".
This code predicts snow depth using topography and vegetation structure by fine resolution airborne lidar data.

"Scaled" folder uses the same dataset and codes as above and the concept of scale breaks to resample the input features to a coarser resolution and attempts 
to predict snow depth using coarser topographical and vegetation structural information.

To run the codes you need to load some modules and install several packages if they are not installed on your system:

##### module load cuda10.0/toolkit/10.0.130
### module load python36
### module load gdal/gcc8/3.0.4
### pip install --user osgeo
### pip3 install --global-option=build_ext --global-option="-I/cm/shared/apps/gdal/gcc8/3.0.4/include" GDAL==3.0.4 --user
### pip install --user sklearn
### pip install --user tensorflow==1.15
### pip install --user keras==2.2.4
### pip install --user opencv-python
### pip install --user pandas
### pip install --user progressbar2
### pip install --user h5py==2.10.0
### pip install --user seaborn


Note: If the system says the requisits are already satisfied but python does not detect the libraries, 
add "--ignore-installed" to the installation line.


Note: as the dataset is about 2GB, we could not upload it here. To access the data please contact "ahmadhojatimalek@u.boisestate.edu".
