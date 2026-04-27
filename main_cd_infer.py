from argparse import ArgumentParser
import torch
from models.trainer import *
import os

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='hyret_cdd', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoints', type=str)
    parser.add_argument('--vis_root', default='./vis', type=str)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='CDD', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)
    parser.add_argument('--hflip', default=0.5, type=float, help='horizontal flip probability (0.0 - 1.0)')
    parser.add_argument('--vflip', default=0.5, type=float, help='vertical flip probability (0.0 - 1.0)')
    parser.add_argument('--flip', default=0.5, type=float, help='complete flip probability (0.0 - 1.0)')
    parser.add_argument('--crop', action='store_true', default=False)
    parser.add_argument('--crop_prob', default=0.5, type=float)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--pretrain', default='./pretrain/resnet50.pth', type=str)
    parser.add_argument('--multi_scale_train', default=False, type=bool)
    parser.add_argument('--multi_scale_infer', default=False, type=bool)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.5, 0.5, 0.5, 0.8, 1.0])
    parser.add_argument('--net_G', default='ChangeBind', type=str, help='ChangeBind model')
    parser.add_argument('--loss', default='ce', type=str)
    # ADDED
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str, help='resnet50 | swin_base | convnext')

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str, help='linear | step')
    parser.add_argument('--lr_decay_iters', default=[100], type=int)
    
    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    #train(args)

    test(args)
