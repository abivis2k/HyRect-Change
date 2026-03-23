set -e  

echo "Installing miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh

export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc
conda config --set always_yes True
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "Creating conda environment..."
conda create -n hyret python=3.8 -y

git clone https://github.com/abivis2k/HyRect-Change.git
cd HyRect-Change/
git checkout jlcorrei

source $HOME/miniconda/etc/profile.d/conda.sh
conda activate hyret

cd datasets/
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/WHU-CD256.zip
unzip WHU-CD256.zip
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/LEVIR-CD256.zip
unzip LEVIR-CD256.zip
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/CDD-CD256.zip
unzip CDD-CD256.zip
rm *.zip
cd ../

cd pretrain/
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/resnet50.pth
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth -P pretrain/
cd ../

sudo apt-get install -y tmux
tmux new -s train

echo "Installing dependencies..."
# conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge -y
# conda install einops matplotlib timm -c conda-forge -y
# conda install fvcore iopath -c conda-forge -y
# pip install opencv-python
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 
pip install einops==0.3.2 fvcore==0.1.5.post20221221 matplotlib==3.7.5 opencv-python setuptools==75.1.0 timm==1.0.24 wheel==0.44.0

echo "Done (conda activate hyret)"