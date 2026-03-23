set -e  

echo "Installing miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh

export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc

echo "Creating conda environment..."
conda create -n hyret python=3.8 -y

source $HOME/miniconda/etc/profile.d/conda.sh
conda activate hyret

echo "Installing dependencies..."
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda install einops matplotlib timm -c conda-forge -y
conda install fvcore iopath -c conda-forge -y
pip install opencv-python

echo "Downloading resnet50 pre-trained weights..."
mkdir -p pretrain
curl -L https://download.pytorch.org/models/resnet50-0676ba61.pth -o pretrain/resnet50.pth
echo "Pretrain weights downloaded."

echo "Done (conda activate hyret)"