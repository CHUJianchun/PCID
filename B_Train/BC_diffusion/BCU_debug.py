# Rdkit import should be first, do not move it
from torch.utils.data import DataLoader, random_split

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from os.path import join
import os
import torch
import time
import pickle
from A_Preprocess.AU_diffusion_dataset import DiffusionDataset
from U_Chem.dataset_info import *
# from U_Chem import dataset
from U_Chem.models import get_optim, get_ionic_liquid_model
from U_Sub_Structure.EDM_model import En_diffusion
from U_Sub_Structure.EDM_model.utils import assert_correctly_masked
from U_Sub_Structure.EDM_model import utils as flow_utils
from U_Chem.utils import prepare_context, compute_mean_mad
from B_Train.BB_diffusion.train_test import train_epoch, test, analyze_and_save

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_0')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=4,  # 9 for QM, 4 for GEOM-drugs
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=64,  # 256 for QM
                    help='number of features')
parser.add_argument('--tanh', type=eval, default=False,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=400)
parser.add_argument('--wandb_usr', type=str)

parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument(
    '--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--stop_test', type=int, default=10,
                    help='do not test all data in test set')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument('--resume', type=str, default='debug_0',
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',  # 本来是sum
                    help='"sum" or "mean"')
args = parser.parse_args()

# dataset_info = get_dataset_info(args.dataset, args.remove_h)

# atom_encoder = dataset_info['atom_encoder']
# atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = "jc-chu"

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)

# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('ZDataU_Param/*.txt')

# Retrieve QM9 dataloaders
# dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
dataset = DiffusionDataset()

dataloaders = dict()
dataloaders['train'], dataloaders['valid'], dataloaders['test'] = random_split(
    dataset,
    lengths=[int(len(dataset) * 0.7), int(len(dataset) * 0.2), len(dataset) - int(len(dataset) * 0.7) - int(
        len(dataset) * 0.2)],
    generator=torch.Generator().manual_seed(1215))
for key in dataloaders.keys():
    dataloaders[key] = DataLoader(dataloaders[key], batch_size=args.batch_size, shuffle=True,
                                  drop_last=True)
charge_scale = 53
data_dummy = next(iter(dataloaders['train']))

property_norms = compute_mean_mad(dataloaders)
anion_context_dummy, cation_context_dummy = prepare_context(data_dummy)
context_node_nf = anion_context_dummy.size(2)

args.context_node_nf = context_node_nf

# Create EGNN flow
model, anion_nodes_dist, cation_nodes_dist, prop_dist = get_ionic_liquid_model(
    args, device, anion_data_info, cation_data_info, dataloaders['train'])

model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    flow_state_dict = torch.load(join('ZDataC_SavedModel/', args.resume, 'generative_model.npy'))
    model.load_state_dict(flow_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp
    analyze_and_save(args=args, epoch=0, model_sample=model_ema, device=device, prop_dist=prop_dist,
                     n_samples=args.n_stability_samples)
    nll_val = test(args=args, loader=dataloaders['valid'], epoch=0, eval_model=model_ema_dp,
                   anion_nodes_dist=anion_nodes_dist, cation_nodes_dist=cation_nodes_dist,
                   partition='Val', device=device, dtype=dtype)
    nll_test = test(args=args, loader=dataloaders['test'], epoch=0, eval_model=model_ema_dp,
                    anion_nodes_dist=anion_nodes_dist, cation_nodes_dist=cation_nodes_dist,
                    partition='Test', device=device, dtype=dtype)


if __name__ == "__main__":
    main()
