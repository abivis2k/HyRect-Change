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
conda create -n hyret --yes python=3.10

source $HOME/miniconda/etc/profile.d/conda.sh
conda activate hyret

echo "Coding GitHub repo..."
git clone https://github.com/abivis2k/HyRect-Change.git
cd HyRect-Change/
git checkout main
mv .env.example .env

echo "Downloading datasets..."
cd datasets/
wget -nc https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/WHU-CD256.zip
unzip -qn WHU-CD256.zip
wget -nc https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/LEVIR-CD256.zip
unzip -qn LEVIR-CD256.zip
wget -nc https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/CDD-CD256.zip
unzip -qn CDD-CD256.zip
rm *.zip
cd ../

echo "Downloading pretrain weights..."
cd pretrain/
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/resnet50.pth
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/mask2former-swin-base-cityscapes-semantic.pth
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/mask2former-swin-tiny-cityscapes-semantic.pth
wget https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar
wget https://huggingface.co/datasets/42meow/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/resnet101.pth
cd ../

echo "Copying over author's checkpoints..."
cd checkpoints/
wget https://huggingface.co/mustansarfiaz/HyRet/resolve/main/hyret_cdd_ckpt.pt
wget https://huggingface.co/mustansarfiaz/HyRet/resolve/main/hyret_levir_ckpt.pt
wget https://huggingface.co/mustansarfiaz/HyRet/resolve/main/hyret_whu_ckpt.pt

echo "Copying over student checkpoints..."
# Please uncomment the wget commands for the dataset you'd like to evaluate:
#
# LEVIR
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet101_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swint_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swint_citysem_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swinb_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swinb_citysem_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_mamba1k_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet50_flip30_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet50_flip30_crop_ckpt.pt
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet50_flip30_crop_ce_dice_ckpt.pt

# CDD
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet101_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swint_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swint_citysem_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swinb_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swinb_citysem_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_mamba1k_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet50_flip30_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet50_flip30_crop_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet50_flip30_crop_ce_dice_ckpt.pt

# WHU
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet101_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swint_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swint_citysem_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swinb_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swinb_citysem_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_mamba1k_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet50_flip30_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet50_flip30_crop_ckpt.pt
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet50_flip30_crop_ce_dice_ckpt.pt


echo "Installing dependencies..."
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 
python -m pip install einops==0.3.2 fvcore==0.1.5.post20221221 matplotlib==3.7.5 opencv-python setuptools==75.1.0 timm==1.0.24 wheel==0.44.0 dotenv
python -m pip install scipy
python -m pip install transformers

echo "DONE"
