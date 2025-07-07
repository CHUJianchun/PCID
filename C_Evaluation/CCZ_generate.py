from torch.utils.data import DataLoader, random_split

from B_Train.BB_encode.BBA_train_encoding import get_encoding_model

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import B_Train.BC_diffusion.utils as utils
import argparse
import wandb
from os.path import join
import os
import torch
import time
import pickle
from A_Preprocess.AU_diffusion_dataset import *
from U_Chem.dataset_info import *
# from U_Chem import dataset
from U_Chem.models import get_optim, get_ionic_liquid_model, get_ion_model
from U_Sub_Structure.EDM_model import En_diffusion
from U_Sub_Structure.EDM_model.utils import assert_correctly_masked
from U_Sub_Structure.EDM_model import utils as flow_utils
from U_Chem.utils import prepare_context, compute_mean_mad
from B_Train.BC_diffusion.train_test_separate import train_epoch, test, analyze_and_save
from B_Train.BC_diffusion.BCAA_train_separate import *


@torch.no_grad()
def main():
    labels = torch.zeros(10).to(device, dtype)
    model_save_dict = "ZDataC_SavedEncodeModel/encoder.model"
    dataset = PropertyDiffusionDataset()
    model = get_encoding_model(args)
    model.load_state_dict(torch.load(model_save_dict))
    anion_embedding_list = torch.zeros((int(dataset.labels[:, 0].max()) + 1,
                                        int(args.embedding_dim / 3)))
    cation_embedding_list = torch.zeros((int(dataset.labels[:, 1].max()) + 1,
                                         args.embedding_dim - int(args.embedding_dim / 3)))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    model.eval()
    embedding = model.get_property_embedding(properties=labels)

    if args.resume is not None:
        flow_state_dict = torch.load('ZDataC_SavedModel/%s/generative_model.npy' % args.exp_name)
        # optim_state_dict = torch.load(optim, 'ZDataC_SavedModel/%s/optim.npy' % args.exp_name)
        model.load_state_dict(flow_state_dict)
        # optim.load_state_dict(optim_state_dict)

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

    best_nll_val = 1e9
    best_nll_test = 1e9

    analyze_and_save(args=args, epoch=0, model_sample=model_ema, device=device, prop_dist=prop_dist,
                     n_samples=args.n_stability_samples, context=embedding)


if __name__ == "__main__":
    main()
