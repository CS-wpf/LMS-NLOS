import os
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net
from train import _train
from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)
    # print(model)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='LMS-NLOS', choices=['MS-NLOS', 'LMS-NLOS'], type=str)
    parser.add_argument('--data_dir', type=str, default='the path of the dataset')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--con_am', type=float, default='parameters for content loss')
    parser.add_argument('--mf_M1', type=float, default='parameters for auxiliary loss')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])
    parser.add_argument('--token_mlp', type=str, default='pefns', help='pefn/pefns token mlp')
    parser.add_argument('--token_attn', type=str, default='SCIA', help='space channel interaction attention')

    # Test
    parser.add_argument('--test_model', type=str, default='the path of the model')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    # GPU
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use')

    args = parser.parse_args()
    args.model_save_dir = os.path.join('LMS/', args.model_name, 'weights/')
    args.result_dir = os.path.join('LMS/', args.model_name, 'results/')
    print(args)
    main(args)
