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

source $HOME/miniconda/etc/profile.d/conda.sh
conda activate hyret

echo "Coding GitHub repo..."
git clone https://github.com/abivis2k/HyRect-Change.git
cd HyRect-Change/
git checkout jlcorrei

echo "Downloading datasets..."
cd datasets/
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/WHU-CD256.zip
unzip WHU-CD256.zip
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/LEVIR-CD256.zip
unzip LEVIR-CD256.zip
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/CDD-CD256.zip
unzip CDD-CD256.zip
rm *.zip
cd ../

echo "Downloading pretrain weights..."
cd pretrain/
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/resnet50.pth
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/mask2former-swin-base-cityscapes-semantic.pth
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/mask2former-swin-tiny-cityscapes-semantic.pth
wget https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar
cd ../

echo "Copying over checkpoint..."
cp -r /mnt/ncsudrive/j/jlcorrei/checkpoints/hyret_whu_swinb/ checkpoints/
cp -r /mnt/ncsudrive/j/jlcorrei/checkpoints/hyret_whu_swinb_cityscapes/ checkpoints/

echo "Installing dependencies..."
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 
python -m pip install einops==0.3.2 fvcore==0.1.5.post20221221 matplotlib==3.7.5 opencv-python setuptools==75.1.0 timm==1.0.24 wheel==0.44.0
python -m pip install scipy
python -m pip install transformers

echo "DONE"
