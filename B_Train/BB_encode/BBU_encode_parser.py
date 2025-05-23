import argparse
import torch


def init_parser():
    parser = argparse.ArgumentParser(description="Regress")
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before logging test')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--nf', type=int, default=10, metavar='N',  # 6
                        help='features')
    parser.add_argument('--n_layers', type=int, default=3, metavar='N',
                        help='number of layers')
    parser.add_argument('--embedding_dim', type=int, default=30, metavar='N',
                        help='')

    parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                        help='maximum power to take into one-hot features')
    parser.add_argument('--charge_scale', type=int, default=20, metavar='N',
                        help='charge scale')
    parser.add_argument('--node_attr', type=int, default=1, metavar='N',
                        help='node_attr or not')
    parser.add_argument('--weight_decay', type=float, default=1e-8, metavar='N',
                        help='weight decay')
    parser.add_argument('--attention', type=bool, default=True, metavar='N',
                        help='use attention')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    return args
