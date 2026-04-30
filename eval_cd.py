from argparse import ArgumentParser
import torch
from models.evaluator import *

print(torch.cuda.is_available())


"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='our_levir', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--checkpoints_root', default='./checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)
    parser.add_argument('--hflip', default=0.5, type=float, help='horizontal flip probability (0.0 - 1.0)')
    parser.add_argument('--vflip', default=0.5, type=float, help='vertical flip probability (0.0 - 1.0)')
    parser.add_argument('--flip', default=0.5, type=float, help='complete flip probability (0.0 - 1.0)')
    parser.add_argument('--crop', action='store_true', default=False)
    parser.add_argument('--crop_prob', default=0.5, type=float)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--net_G', default='ChangeBind', type=str, help='ScratchFormer')
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str, help='resnet50 | swin_base | convnext')

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoints_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()

