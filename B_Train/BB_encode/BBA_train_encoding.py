import torch.nn.functional
from torch import optim
from torch.utils.data import DataLoader, random_split
from U_Sub_Structure.EGNN_model.util_for_regress import *
from U_Sub_Structure.MLP import *
from A_Preprocess.AU_diffusion_dataset import *
from B_Train.BB_encode.BBU_encode_parser import init_parser
from U_Sub_Structure.EGNN_model.EGNN_for_regress import EGNN_Encoder


def get_encoding_model(args):
    return EGNN_Encoder(anion_in_node_nf=24, anion_in_edge_nf=0, anion_hidden_nf=args.nf,
                        cation_in_node_nf=24, cation_in_edge_nf=0, cation_hidden_nf=args.nf,
                        embedding_dim=args.embedding_dim, device=args.device, n_layers=args.n_layers,
                        coords_weight=0.2, act_fn=nn.LeakyReLU(),
                        attention=args.attention, node_attr=args.node_attr)


def train_model(start_epoch=-1):
    args = init_parser()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    print(args)
    model_save_dict = "ZDataC_SavedEncodeModel/encoder.model"
    dataset = PropertyDiffusionDataset()
    model = get_encoding_model(args)
    if start_epoch != -1:
        model.load_state_dict(torch.load(model_save_dict))
    optimizer = optim.Adam(model.parameters(), lr=args.lr * args.batch_size / 100, weight_decay=args.weight_decay)
    print(f"Learning rate = {args.lr * args.batch_size / 100}")
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.4)
    dataloader = {}
    dataloader['train'], dataloader['valid'], dataloader['test'] = random_split(
        dataset,
        lengths=[int(len(dataset) * 0.7), int(len(dataset) * 0.2), len(dataset) - int(len(dataset) * 0.7) - int(
            len(dataset) * 0.2)],
        generator=torch.Generator().manual_seed(1215))
    for key in dataloader.keys():
        dataloader[key] = DataLoader(dataloader[key], batch_size=args.batch_size, shuffle=True,
                                     drop_last=True)

    def encode_loss(graph_embedding, property_embedding):
        loss = torch.nn.functional.l1_loss(graph_embedding, property_embedding)
        # diff_loss = torch.nn.functional.l1_loss(graph_embedding - torch.mean(property_embedding, dim=0),
        #                                         torch.ones_like(graph_embedding) - 0.25)
        diff_loss = torch.mean(
            torch.abs(torch.var(graph_embedding, dim=0) - 0.3) + torch.abs(torch.var(property_embedding, dim=0) - 0.3))
        return loss + diff_loss

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
            labels = data['labels'].to(device, dtype)
            graph_embedding, property_embedding = model(anion_h0=anion_nodes, anion_x=anion_atom_positions,
                                                        anion_edges=anion_edges,
                                                        anion_node_mask=anion_atom_mask,
                                                        anion_edge_mask=anion_edge_mask,
                                                        anion_n_nodes=anion_n_nodes, anion_edge_attr=None,

                                                        cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                                        cation_edges=cation_edges,
                                                        cation_node_mask=cation_atom_mask,
                                                        cation_edge_mask=cation_edge_mask,
                                                        cation_n_nodes=cation_n_nodes, cation_edge_attr=None,

                                                        properties=labels[:, 2:])

            prefix = ""
            loss = encode_loss(graph_embedding, property_embedding)
            if partition == 'train':
                # loss = loss_l1(prediction, (label - data_mean) / data_mad)
                loss.backward()
                optimizer.step()
            else:
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
    train_model(start_epoch=0)
