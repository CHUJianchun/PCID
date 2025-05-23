from torch import optim
from torch.utils.data import DataLoader, random_split
from U_Sub_Structure.EGNN_model.util_for_regress import *
from U_Sub_Structure.MLP import *
from A_Preprocess.AU_regress_dataset import RegressDatasetSeparate
from B_Train.BA_regress.BAU_regress_parser import init_parser
from U_Sub_Structure.EGNN_model.EGNN_for_regress import EGNN_Regress_Seperate


def get_regress_model(args):
    return EGNN_Regress_Seperate(anion_in_node_nf=24, anion_in_edge_nf=0, anion_hidden_nf=args.nf,
                                 cation_in_node_nf=24, cation_in_edge_nf=0, cation_hidden_nf=args.nf,
                                 device=args.device, n_layers=args.n_layers,
                                 coords_weight=0.2, act_fn=nn.LeakyReLU(),
                                 attention=args.attention, node_attr=args.node_attr)


def load_oped_args(property_, args_):
    """
    batch size :: num features :: layers
    Solubility
    [[10, 3, 3, 0.05671009661841996]]  # the best
    Viscosity
    [[10, 20, 6, 0.0002123395674567485]]  # the best
    Heat_capacity
    [[10, 10, 3, 0.00636298474855721]]  # the best
    Density
    [[10, 20, 10, 0.013017299545422327]]  # the best
    """
    if property_ == "Solubility":
        args_.batch_size = 10
        args_.nf = 3
        args_.n_layers = 3
    elif property_ == "Viscosity":
        args_.batch_size = 10
        args_.nf = 20
        args_.n_layers = 6
    elif property_ == "Heat_capacity":
        args_.batch_size = 10
        args_.nf = 10
        args_.n_layers = 3
    elif property_ == "Density":
        args_.batch_size = 10
        args_.nf = 20
        args_.n_layers = 10
    else:
        raise ValueError("property input invalid")
    return args_


def train_model_for(property_, start_epoch=-1):
    args = init_parser()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    print(args)
    args = load_oped_args(property_, args)
    model_save_dict = "ZDataC_SavedRegressModel/Regress_separate_" + property_ + ".model"
    dataset = RegressDatasetSeparate(property_)
    # data_mean, data_mad = mean_mad(dataset)
    data_min, data_max = min_max(dataset)
    t_min, t_max, p_min, p_max = min_max_t_p()
    model = get_regress_model(args)
    if start_epoch != -1:
        model.load_state_dict(torch.load(model_save_dict))
    optimizer = optim.Adam(model.parameters(), lr=args.lr * args.batch_size, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8, last_epoch=start_epoch)
    loss_l1 = nn.L1Loss()
    dataloader = {}
    dataloader['train'], dataloader['valid'], dataloader['test'] = random_split(
        dataset,
        lengths=[int(len(dataset) * 0.7), int(len(dataset) * 0.2), len(dataset) - int(len(dataset) * 0.7) - int(
            len(dataset) * 0.2)],
        generator=torch.Generator().manual_seed(1215))
    for key in dataloader.keys():
        dataloader[key] = DataLoader(dataloader[key], batch_size=args.batch_size, shuffle=True,
                                     drop_last=True)

    def train(epoch_, loader, partition="train"):

        res_ = {'loss': 0, 'counter': 0, 'loss_arr': []}
        for i, data in enumerate(loader):
            if partition == 'train':
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            batch_size, anion_n_nodes, _ = data['anion_atom_position_list'].size()
            batch_size, cation_n_nodes, _ = data['cation_atom_position_list'].size()
            anion_atom_positions = data['anion_atom_position_list'].view(batch_size * anion_n_nodes, -1).to(device,
                                                                                                            dtype)
            anion_atom_mask = data['anion_node_mask_list'].view(batch_size * anion_n_nodes, -1).to(device, dtype)
            anion_edge_mask = data['anion_edge_mask_list'].view(-1, 1).to(device, dtype)
            anion_one_hot = data['anion_one_hot_list'].to(device, dtype)
            anion_charges = data['anion_charge_list'].to(device, dtype)
            anion_nodes = preprocess_input(anion_one_hot, anion_charges, args.charge_power, args.charge_scale, device)
            anion_nodes = anion_nodes.view(batch_size * anion_n_nodes, -1)
            anion_edges = get_adj_matrix(anion_n_nodes, batch_size, device)

            cation_atom_positions = data['cation_atom_position_list'].view(batch_size * cation_n_nodes, -1).to(device,
                                                                                                               dtype)
            cation_atom_mask = data['cation_node_mask_list'].view(batch_size * cation_n_nodes, -1).to(device, dtype)
            cation_edge_mask = data['cation_edge_mask_list'].view(-1, 1).to(device, dtype)
            cation_one_hot = data['cation_one_hot_list'].to(device, dtype)
            cation_charges = data['cation_charge_list'].to(device, dtype)
            cation_nodes = preprocess_input(cation_one_hot, cation_charges, args.charge_power, args.charge_scale,
                                            device)
            cation_nodes = cation_nodes.view(batch_size * cation_n_nodes, -1)
            cation_edges = get_adj_matrix(cation_n_nodes, batch_size, device)

            temperature = (data['temperature_list'].view(-1, 1).to(device, dtype) - t_min) / (t_max - t_min)
            pressure = (data['pressure_list'].view(-1, 1).to(device, dtype) - p_min) / (p_max - p_min)
            label = data['value_list'].to(device, dtype)

            prediction = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                               anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                               anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                               cation_h0=cation_nodes, cation_x=cation_atom_positions, cation_edges=cation_edges,
                               cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                               cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                               t=temperature, p=pressure)
            prefix = ""
            if partition == 'train':
                # loss = loss_l1(prediction, (label - data_mean) / data_mad)
                if property_ == "Solubility":
                    loss = loss_l1(prediction, label)
                else:
                    loss = loss_l1(prediction, (label - data_min) / (data_max - data_min))
                loss.backward()
                optimizer.step()

            else:
                # loss = loss_l1(prediction, (label - data_mean) / data_mad)
                if property_ == "Solubility":
                    loss = loss_l1(prediction, label)
                else:
                    loss = loss_l1(prediction, (label - data_min) / (data_max - data_min))
                prefix = ">> %s \t" % partition
            res_['loss'] += loss.item() * batch_size
            res_['counter'] += batch_size
            res_['loss_arr'].append(loss.item())
            if i % args.log_interval == 0:
                print(
                    prefix + "\033[0;31;40mEpoch\033[0m %d \t \033[0;32;40mIteration\033[0m %d \t \033[0;33;40mAE "
                             "loss\033[0m %.4f" % (epoch_, i, res_['loss'] / res_['counter']))
        return res_['loss'] / res_['counter']

    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    for epoch in range(0, args.epochs):
        train(epoch, dataloader['train'], partition='train')
        lr_scheduler.step()
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, dataloader['valid'], partition='valid')
            test_loss = train(epoch, dataloader['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
                torch.save(model.state_dict(), model_save_dict)
                print("model saved")
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (
                res['best_val'], res['best_test'], res['best_epoch']))


if __name__ == "__main__":
    train_model_for(property_="Solubility")
    train_model_for(property_="Viscosity")
    train_model_for(property_="Heat_capacity")
    train_model_for(property_="Density")
    pass
