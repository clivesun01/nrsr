# clone dso codes
git clone https://github.com/dso-org/deep-symbolic-optimization.git
cp -r ./codes/* ./deep-symbolic-optimization/dso/dso/
cd ./deep-symbolic-optimization/

# Set up key packaging-related tools:
pip install --upgrade pip
pip install "setuptools<58.0.0"  # Required for installing deap==1.3.0

# Install dso library:
#export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS" # Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found
pip install -e ./dso # Install DSO package and core dependencies

# Install other dependencies:
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia
# ^ Or use the pip alternative torch installation command from https://pytorch.org/get-started/locally/
# Choose a different version of CUDA or CPU-only, as needed.
# pip install -r requirements.txt
pip install scikit-image
