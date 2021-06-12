The main code is called "Deep_Learning_V2.py".
This code predicts snow depth using topography and vegetation structure by fine resolution airborne lidar data.

To run this code you need to load some modules and install several packages if they are not installed on your system:

module load cuda10.0/toolkit/10.0.130
module load python36
module load gdal/gcc8/3.0.4
pip install --user osgeo
pip3 install --global-option=build_ext --global-option="-I/cm/shared/apps/gdal/gcc8/3.0.4/include" GDAL==3.0.4 --user
pip install --user sklearn
pip install --user tensorflow==1.15
pip install --user keras==2.2.4
pip install --user opencv-python
pip install --user pandas
pip install --user progressbar2
pip install --user h5py==2.10.0
pip install --user seaborn


Note: If the system says the requisits are already satisfied but python does not detect the libraries, 
add "--ignore-installed" to the installation line.
