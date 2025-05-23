from torch import optim
from torch.utils.data import DataLoader, random_split

from B_Train.BA_regress.BAA_regress_separate_ion_V2 import get_regress_model
from U_Sub_Structure.EGNN_model.util_for_regress import *
from U_Sub_Structure.MLP import *
from A_Preprocess.AU_regress_dataset import RegressDataset
from B_Train.BA_regress.BAU_regress_parser import init_parser
from U_Sub_Structure.EGNN_model.EGNN_for_regress import EGNN_Regress


def optim_hyper_param_for(property_):
    args = init_parser()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    print(args)
    args.device = device
    print(property_)
    batch_size_range = (10, 100, 200)
    nf_range = (3, 10, 20)
    layer_range = (3, 6, 10)
    dataset = RegressDatasetSeparate(property_)
    data_mean, data_mad = mean_mad(dataset)
    data_min, data_max = min_max(dataset)
    t_min, t_max, p_min, p_max = min_max_t_p()
    loss_record = []
    args.attention = True
    for bs in batch_size_range:
        for nf in nf_range:
            for lar in layer_range:
                # print([bs, nf, lar])
                args.n_layers = lar
                args.batch_size = bs
                args.nf = nf
                model = get_regress_model(args)
                optimizer = optim.Adam(model.parameters(), lr=args.lr * bs, weight_decay=args.weight_decay)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
                loss_l1 = nn.L1Loss()
                dataloader = {}
                dataloader['train'], dataloader['valid'], dataloader['test'] = random_split(
                    dataset,
                    lengths=[int(len(dataset) * 0.7),
                             int(len(dataset) * 0.2),
                             len(dataset) - int(len(dataset) * 0.7) - int(len(dataset) * 0.2)],
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
                        anion_atom_positions = data['anion_atom_position_list'].view(batch_size * anion_n_nodes, -1).to(
                            device,
                            dtype)
                        anion_atom_mask = data['anion_node_mask_list'].view(batch_size * anion_n_nodes, -1).to(device,
                                                                                                               dtype)
                        anion_edge_mask = data['anion_edge_mask_list'].view(-1, 1).to(device, dtype)
                        anion_one_hot = data['anion_one_hot_list'].to(device, dtype)
                        anion_charges = data['anion_charge_list'].to(device, dtype)
                        anion_nodes = preprocess_input(anion_one_hot, anion_charges, args.charge_power,
                                                       args.charge_scale, device)
                        anion_nodes = anion_nodes.view(batch_size * anion_n_nodes, -1)
                        anion_edges = get_adj_matrix(anion_n_nodes, batch_size, device)

                        cation_atom_positions = data['cation_atom_position_list'].view(batch_size * cation_n_nodes,
                                                                                       -1).to(device,
                                                                                              dtype)
                        cation_atom_mask = data['cation_node_mask_list'].view(batch_size * cation_n_nodes, -1).to(
                            device, dtype)
                        cation_edge_mask = data['cation_edge_mask_list'].view(-1, 1).to(device, dtype)
                        cation_one_hot = data['cation_one_hot_list'].to(device, dtype)
                        cation_charges = data['cation_charge_list'].to(device, dtype)
                        cation_nodes = preprocess_input(cation_one_hot, cation_charges, args.charge_power,
                                                        args.charge_scale,
                                                        device)
                        cation_nodes = cation_nodes.view(batch_size * cation_n_nodes, -1)
                        cation_edges = get_adj_matrix(cation_n_nodes, batch_size, device)

                        temperature = (data['temperature_list'].view(-1, 1).to(device, dtype) - t_min) / (t_max - t_min)
                        pressure = (data['pressure_list'].view(-1, 1).to(device, dtype) - p_min) / (p_max - p_min)
                        label = data['value_list'].to(device, dtype)

                        prediction = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                           anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                           anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                           cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                           cation_edges=cation_edges,
                                           cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                           cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                           t=temperature, p=pressure)
                        prefix = ""
                        if partition == 'train':
                            if property_ == "Solubility":
                                loss = loss_l1(prediction, label)
                            else:
                                loss = loss_l1(prediction, (label - data_min) / (data_max - data_min))
                            loss.backward()
                            optimizer.step()

                        else:
                            if property_ == "Solubility":
                                loss = loss_l1(prediction, label)
                            else:
                                loss = loss_l1(prediction, (label - data_min) / (data_max - data_min))
                            # prefix = ">> %s \t" % partition
                        res_['loss'] += loss.item() * batch_size
                        res_['counter'] += batch_size
                        res_['loss_arr'].append(loss.item())
                    return res_['loss'] / res_['counter']
                res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
                for epoch in range(0, 6):
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
                loss_record.append([bs, nf, lar, res['best_val']])
                print([[bs, nf, lar, res['best_val']]])
    return loss_record


if __name__ == "__main__":
    optim_hyper_param_for(property_="Solubility")
    optim_hyper_param_for(property_="Viscosity")
    optim_hyper_param_for(property_="Heat_capacity")
    optim_hyper_param_for(property_="Density")
    # 别忘了学习率乘以了batch size
