from tqdm import tqdm
from B_Train.BB_encode.BBA_train_encoding import *


@torch.no_grad()
def save_encoding():
    args = init_parser()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    print(args)
    model_save_dict = "ZDataC_SavedEncodeModel/encoder.model"
    dataset = PropertyDiffusionDataset()
    model = get_encoding_model(args)
    model.load_state_dict(torch.load(model_save_dict))
    all_il_embedding_list = torch.zeros((len(dataset), args.embedding_dim))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    model.eval()
    for i, data in enumerate(tqdm(dataloader)):
        labels = data['labels'].to(device, dtype)
        all_il_embedding_list[i] = model.get_property_embedding(properties=labels[:, 2:])
    torch.save(all_il_embedding_list, "ZDataB_ProcessedData/all_il_embedding_list.tensor")


@torch.no_grad()
def save_encoding_separate():
    args = init_parser()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    print(args)
    model_save_dict = "ZDataC_SavedEncodeModel/encoder.model"
    dataset = PropertyDiffusionDataset()
    model = get_encoding_model(args)
    model.load_state_dict(torch.load(model_save_dict))
    # 先 anion 后 cation
    anion_embedding_list = torch.zeros((int(dataset.labels[:, 0].max()) + 1,
                                        int(args.embedding_dim / 3)))
    cation_embedding_list = torch.zeros((int(dataset.labels[:, 1].max()) + 1,
                                         args.embedding_dim - int(args.embedding_dim / 3)))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    model.eval()
    for i, data in enumerate(tqdm(dataloader)):
        labels = data['labels'].to(device, dtype)
        embedding = model.get_property_embedding(properties=labels[:, 2:])
        anion_embedding_list[int(labels[:, 0])] = embedding[:, :int(args.embedding_dim / 3)]
        cation_embedding_list[int(labels[:, 1])] = embedding[:, int(args.embedding_dim / 3):]
    torch.save(anion_embedding_list, "ZDataB_ProcessedData/Embedding_List/anion_embedding_list.tensor")
    torch.save(cation_embedding_list, "ZDataB_ProcessedData/Embedding_List/cation_embedding_list.tensor")
    # 321 anions, 851 cations, 273171 ILs
    return anion_embedding_list, cation_embedding_list


if __name__ == "__main__":
    # save_encoding()
    ael, cel = save_encoding_separate()
