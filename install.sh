sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
#python3 -m pip install virtualenv
virtualenv lpr_venv --python=python3.7
source lpr_venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install opencv-contrib-python
python3 -m pip install cython
python3 -m pip install imutils
python3 -m pip install colorful
python3 -m pip install iridi
python3 -m pip install roipoly
python3 -m pip install shapely
python3 -m pip install tensorflow==1.14.0
python3 -m pip install tensorflow-gpu==1.14.0
python3 -m pip install keras==2.3
python3 -m pip install h5py==2.10.0
python3 -m pip install pyfiglet
python3 -m pip install iridi
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$PATH"
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda/'
