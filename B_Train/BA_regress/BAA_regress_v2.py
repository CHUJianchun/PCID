from torch import optim
from torch.utils.data import DataLoader, random_split
from U_Sub_Structure.EGNN_model.util_for_regress import *
from U_Sub_Structure.MLP import *
from A_Preprocess.AU_regress_dataset import RegressDataset
from B_Train.BA_regress.BAU_regress_parser import init_parser
from U_Sub_Structure.EGNN_model.EGNN_for_regress import EGNN_Regress
args = init_parser()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)


def train_model_for(property_, start_epoch=-1):
    model_save_dict = "ZDataC_SavedModel/Regress_" + property_ + ".model"
    dataset = RegressDataset(property_)
    data_mean, data_mad = mean_mad(dataset)
    t_min, t_max, p_min, p_max = min_max_t_p()
    model = EGNN_Regress(in_node_nf=30, in_edge_nf=0, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                         coords_weight=0.1,
                         attention=args.attention, node_attr=args.node_attr)
    if start_epoch != -1:
        model.load_state_dict(torch.load(model_save_dict))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=start_epoch)
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

            batch_size, n_nodes, _ = data['atom_position_list'].size()
            atom_positions = data['atom_position_list'].view(batch_size * n_nodes, -1).to(device, dtype)
            atom_mask = data['node_mask_list'].view(batch_size * n_nodes, -1).to(device, dtype)
            edge_mask = data['edge_mask_list'].view(-1, 1).to(device, dtype)
            one_hot = data['one_hot_list'].to(device, dtype)
            charges = data['charge_list'].to(device, dtype)
            nodes = preprocess_input(one_hot, charges, args.charge_power, args.charge_scale, device)
            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = get_adj_matrix(n_nodes, batch_size, device)
            temperature = (data['temperature_list'].view(-1, 1).to(device, dtype) - t_min) / (t_max - t_min)
            pressure = (data['pressure_list'].view(-1, 1).to(device, dtype) - p_min) / (p_max - p_min)
            label = data['value_list'].to(device, dtype)
            prediction = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask,
                               edge_mask=edge_mask,
                               n_nodes=n_nodes, t=temperature, p=pressure)
            prefix = ""
            if partition == 'train':
                loss = loss_l1(prediction, (label - data_mean) / data_mad)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            else:
                loss = loss_l1(prediction, (label - data_mean) / data_mad)
                prefix = ">> %s \t" % partition
            res_['loss'] += loss.item() * batch_size
            res_['counter'] += batch_size
            res_['loss_arr'].append(loss.item())
            if i % args.log_interval == 0:
                print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (
                    epoch_, i, res_['loss'] / res_['counter']))
        return res_['loss'] / res_['counter']
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    for epoch in range(0, args.epochs):
        train(epoch, dataloader['train'], partition='train')
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
    # train_model_for(property_="Viscosity")
    # train_model_for(property_="Heat_capacity")
    # train_model_for(property_="Density")
