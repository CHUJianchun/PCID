from A_Preprocess.AU_regress_dataset import AllILDataset
from B_Train.BA_regress.BAA_regress_separate_ion_V2 import *
from A_Preprocess.AA_read_data_seperate_ion import *
from U_Sub_Structure.EGNN_model.EGNN_for_regress import EGNN_Regress_Seperate
from B_Train.BA_regress.BAA_regress_separate_ion_V2 import get_regress_model, load_oped_args


def unsparse_data():
    args = init_parser()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    args.device = device
    """
        0: M=226.03,  # 离子液体相对分子质量，自变量
        1: dynamic_viscosity_335=0.022,  # 离子液体在335.64K下的粘度 Pa s
        2: dynamic_viscosity_353=0.0139,  # 离子液体在353.15K下的粘度 Pa s
        3: density_335=1174.36,  # 离子液体的密度 kg/m3
        4: density_353=1161.93,  # 离子液体的密度 kg/m3
        5: heat_capacity_335=384.5,  # 离子液体在335.64K下的比热容 J/K/mol
        6: heat_capacity_353=393.0,  # 离子液体353.15K下的比热容 J/K/mol
        7: solubility_335=0.2,  # 离子液体在335.64K, 3MPa下的mole溶解度 ！！！高！！！
        8: solubility_353=0.0062,  # 离子液体在353.15K, 101kPa下的mole溶解度
        9: solubility_339=0.025,  # 离子液体在339.14K, 101kPa下的mole溶解度
    """
    t_min, t_max, p_min, p_max = min_max_t_p()
    t_335 = (335.64 - t_min) / (t_max - t_min).to(device, dtype).unsqueeze(dim=0).unsqueeze(dim=1)
    t_353 = (353.15 - t_min) / (t_max - t_min).to(device, dtype).unsqueeze(dim=0).unsqueeze(dim=1)
    t_339 = (339.14 - t_min) / (t_max - t_min).to(device, dtype).unsqueeze(dim=0).unsqueeze(dim=1)
    p_101 = (101 - p_min) / (p_max - p_min).to(device, dtype).unsqueeze(dim=0).unsqueeze(dim=1)
    p_1k = (1000 - p_min) / (p_max - p_min).to(device, dtype).unsqueeze(dim=0).unsqueeze(dim=1)
    dataset = AllILDataset()
    all_il_property_list = torch.zeros((len(dataset), 10))
    dataset = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    def unsparse_data_for(property_):
        args = init_parser()
        args.device = device
        model_save_dict = "ZDataC_SavedRegressModel/Regress_separate_" + property_ + ".model"
        args = load_oped_args(property_, args)
        model = get_regress_model(args)
        model.load_state_dict(torch.load(model_save_dict))
        # data_mean, data_mad = mean_mad(property_=property_)
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataset)):
                batch_size = 1
                _, anion_n_nodes, __ = data['anion_atom_position_list'].size()
                _, cation_n_nodes, __ = data['cation_atom_position_list'].size()
                anion_atom_positions = data['anion_atom_position_list'].view(batch_size * anion_n_nodes, -1).to(device,
                                                                                                                dtype)
                anion_atom_mask = data['anion_node_mask_list'].view(batch_size * anion_n_nodes, -1).to(device, dtype)
                anion_edge_mask = data['anion_edge_mask_list'].view(-1, 1).to(device, dtype)
                anion_one_hot = data['anion_one_hot_list'].to(device, dtype)
                anion_charges = data['anion_charge_list'].to(device, dtype)
                anion_nodes = preprocess_input(anion_one_hot, anion_charges, args.charge_power, args.charge_scale,
                                               device)
                anion_nodes = anion_nodes.view(batch_size * anion_n_nodes, -1)
                anion_edges = get_adj_matrix(anion_n_nodes, batch_size, device)

                cation_atom_positions = data['cation_atom_position_list'].view(batch_size * cation_n_nodes, -1).to(
                    device,
                    dtype)
                cation_atom_mask = data['cation_node_mask_list'].view(batch_size * cation_n_nodes, -1).to(device, dtype)
                cation_edge_mask = data['cation_edge_mask_list'].view(-1, 1).to(device, dtype)
                cation_one_hot = data['cation_one_hot_list'].to(device, dtype)
                cation_charges = data['cation_charge_list'].to(device, dtype)
                cation_nodes = preprocess_input(cation_one_hot, cation_charges, args.charge_power, args.charge_scale,
                                                device)
                cation_nodes = cation_nodes.view(batch_size * cation_n_nodes, -1)
                cation_edges = get_adj_matrix(cation_n_nodes, batch_size, device)
                """
                    0: M=226.03,  # 离子液体相对分子质量，自变量
                    1: dynamic_viscosity_335=0.022,  # 离子液体在335.64K下的粘度 Pa s
                    2: dynamic_viscosity_353=0.0139,  # 离子液体在353.15K下的粘度 Pa s
                """
                all_il_property_list[i, 0] = data['anion_mw_list'] + data['cation_mw_list']
                if property_ == "Viscosity":
                    vis_335 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                    anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                    anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                    cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                    cation_edges=cation_edges,
                                    cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                    cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                    t=t_335, p=p_101)
                    vis_353 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                    anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                    anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                    cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                    cation_edges=cation_edges,
                                    cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                    cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                    t=t_353, p=p_101)
                    all_il_property_list[i, 1] = vis_335.detach().cpu()
                    all_il_property_list[i, 2] = vis_353.detach().cpu()
                    """
                        3: density_335=1174.36,  # 离子液体的密度 kg/m3
                        4: density_353=1161.93,  # 离子液体的密度 kg/m3
                    """
                elif property_ == "Density":
                    den_335 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                    anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                    anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                    cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                    cation_edges=cation_edges,
                                    cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                    cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                    t=t_335, p=p_101)
                    den_353 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                    anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                    anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                    cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                    cation_edges=cation_edges,
                                    cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                    cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                    t=t_353, p=p_101)
                    all_il_property_list[i, 3] = den_335.detach().cpu()
                    all_il_property_list[i, 4] = den_353.detach().cpu()
                    """
                        5: heat_capacity_335=384.5,  # 离子液体在335.64K下的比热容 J/K/mol
                        6: heat_capacity_353=393.0,  # 离子液体353.15K下的比热容 J/K/mol
                    """
                elif property_ == "Heat_capacity":
                    hc_335 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                   anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                   anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                   cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                   cation_edges=cation_edges,
                                   cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                   cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                   t=t_335, p=p_101)
                    hc_353 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                   anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                   anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                   cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                   cation_edges=cation_edges,
                                   cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                   cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                   t=t_353, p=p_101)
                    all_il_property_list[i, 5] = hc_335.detach().cpu()
                    all_il_property_list[i, 6] = hc_353.detach().cpu()
                    """
                        7: solubility_335=0.2,  # 离子液体在335.64K, 3MPa下的mole溶解度 ！！！高！！！
                        8: solubility_353=0.0062,  # 离子液体在353.15K, 101kPa下的mole溶解度
                        9: solubility_339=0.025,  # 离子液体在339.14K, 101kPa下的mole溶解度
                    """
                elif property_ == "Solubility":
                    sol_335 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                    anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                    anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                    cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                    cation_edges=cation_edges,
                                    cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                    cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                    t=t_335, p=p_1k)
                    hc_353 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                   anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                   anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                   cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                   cation_edges=cation_edges,
                                   cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                   cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                   t=t_353, p=p_101)
                    hc_339 = model(anion_h0=anion_nodes, anion_x=anion_atom_positions, anion_edges=anion_edges,
                                   anion_node_mask=anion_atom_mask, anion_edge_mask=anion_edge_mask,
                                   anion_n_nodes=anion_n_nodes, anion_edge_attr=None,
                                   cation_h0=cation_nodes, cation_x=cation_atom_positions,
                                   cation_edges=cation_edges,
                                   cation_node_mask=cation_atom_mask, cation_edge_mask=cation_edge_mask,
                                   cation_n_nodes=cation_n_nodes, cation_edge_attr=None,
                                   t=t_339, p=p_101)
                    all_il_property_list[i, 7] = sol_335.detach().cpu()
                    all_il_property_list[i, 8] = hc_353.detach().cpu()
                    all_il_property_list[i, 9] = hc_339.detach().cpu()
                else:
                    raise ValueError("Property out of targets")

    unsparse_data_for(property_="Solubility")
    unsparse_data_for(property_="Viscosity")
    unsparse_data_for(property_="Heat_capacity")
    unsparse_data_for(property_="Density")
    torch.save(all_il_property_list, "ZDataB_ProcessedData/all_il_property_list.tensor")


if __name__ == "__main__":
    unsparse_data()
