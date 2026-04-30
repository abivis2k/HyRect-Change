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

echo "Copying over author's checkpoints..."
cd checkpoints/
mkdir hyret_levir
cd hyret_levir
wget https://huggingface.co/mustansarfiaz/HyRet/resolve/main/hyret_levir_ckpt.pt
cd ../
mkdir hyret_cdd
cd hyret_cdd
wget https://huggingface.co/mustansarfiaz/HyRet/resolve/main/hyret_cdd_ckpt.pt
cd ../
mkdir hyret_whu
cd hyret_whu
wget https://huggingface.co/mustansarfiaz/HyRet/resolve/main/hyret_whu_ckpt.pt
cd ../

echo "Copying over student checkpoints..."
# Please uncomment the commands for the dataset you'd like to evaluate:

# LEVIR
mkdir hyret_levir_resnet101
cd hyret_levir_resnet101
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet101_ckpt.pt
cd ../
mkdir hyret_levir_swint
cd hyret_levir_swint
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swint_ckpt.pt
cd ../
mkdir hyret_levir_swint_citysem
cd hyret_levir_swint_citysem
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swint_citysem_ckpt.pt
cd ../
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swinb_ckpt.pt
mkdir hyret_levir_swinb_citysem
cd hyret_levir_swinb_citysem
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_swinb_citysem_ckpt.pt
cd ../
mkdir hyret_levir_mamba1k
cd hyret_levir_mamba1k
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_mamba1k_ckpt.pt
cd ../
mkdir hyret_levir_resnet50_flip30
cd hyret_levir_resnet50_flip30
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet50_flip30_ckpt.pt
cd ../
mkdir hyret_levir_resnet50_flip30_crop
cd hyret_levir_resnet50_flip30_crop
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet50_flip30_crop_ckpt.pt
cd ../
mkdir hyret_levir_resnet50_flip30_crop_ce_dice
cd hyret_levir_resnet50_flip30_crop_ce_dice
wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_levir_resnet50_flip30_crop_ce_dice_ckpt.pt
cd ../

# CDD
# mkdir hyret_cdd_resnet101
# cd hyret_cdd_resnet101
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet101_ckpt.pt
# cd ../
# mkdir hyret_cdd_swint
# cd hyret_cdd_swint
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swint_ckpt.pt
# cd ../
# mkdir hyret_cdd_swint_citysem
# cd hyret_cdd_swint_citysem
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swint_citysem_ckpt.pt
# cd ../
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swinb_ckpt.pt
# mkdir hyret_cdd_swinb_citysem
# cd hyret_cdd_swinb_citysem
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_swinb_citysem_ckpt.pt
# cd ../
# mkdir hyret_cdd_mamba1k
# cd hyret_cdd_mamba1k
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_mamba1k_ckpt.pt
# cd ../
# mkdir hyret_cdd_resnet50_flip30
# cd hyret_cdd_resnet50_flip30
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet50_flip30_ckpt.pt
# cd ../
# mkdir hyret_cdd_resnet50_flip30_crop
# cd hyret_cdd_resnet50_flip30_crop
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet50_flip30_crop_ckpt.pt
# cd ../
# mkdir hyret_cdd_resnet50_flip30_crop_ce_dice
# cd hyret_cdd_resnet50_flip30_crop_ce_dice
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_cdd_resnet50_flip30_crop_ce_dice_ckpt.pt
# cd ../

# WHU
# mkdir hyret_whu_resnet101
# cd hyret_whu_resnet101
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet101_ckpt.pt
# cd ../
# mkdir hyret_whu_swint
# cd hyret_whu_swint
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swint_ckpt.pt
# cd ../
# mkdir hyret_whu_swint_citysem
# cd hyret_whu_swint_citysem
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swint_citysem_ckpt.pt
# cd ../
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swinb_ckpt.pt
# mkdir hyret_whu_swinb_citysem
# cd hyret_whu_swinb_citysem
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_swinb_citysem_ckpt.pt
# cd ../
# mkdir hyret_whu_mamba1k
# cd hyret_whu_mamba1k
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_mamba1k_ckpt.pt
# cd ../
# mkdir hyret_whu_resnet50_flip30
# cd hyret_whu_resnet50_flip30
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet50_flip30_ckpt.pt
# cd ../
# mkdir hyret_whu_resnet50_flip30_crop
# cd hyret_whu_resnet50_flip30_crop
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet50_flip30_crop_ckpt.pt
# cd ../
# mkdir hyret_whu_resnet50_flip30_crop_ce_dice
# cd hyret_whu_resnet50_flip30_crop_ce_dice
# wget https://huggingface.co/datasets/Csc722Sp26Group1/CSC722_SP26_GROUP1_CHANGE_DETECTION/resolve/main/hyret_whu_resnet50_flip30_crop_ce_dice_ckpt.pt
# cd ../
cd ../

echo "Installing dependencies..."
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 
python -m pip install einops==0.3.2 fvcore==0.1.5.post20221221 matplotlib==3.7.5 opencv-python setuptools==75.1.0 timm==1.0.24 wheel==0.44.0
python -m pip install scipy
python -m pip install transformers

echo "DONE"
