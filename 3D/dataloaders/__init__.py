from dataloaders.datasets import  nodule_dataset
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'nodule':
        data_path = args.data_path
        label_path = args.label_path
        train_data_list = args.train_data_list
        val_data_list = args.val_data_list

        num_class = 2
        train_set = nodule_dataset.PulmonuryNodule(num_classes=2, nodule_list_csv=train_data_list, data_path=data_path, \
                                                   label_path=label_path, is_train=True, is_HEM=False)
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=1)

        val_set = nodule_dataset.PulmonuryNodule(num_classes=2, nodule_list_csv=val_data_list, data_path=data_path, \
                                                   label_path=label_path, is_train=True, is_HEM=False)
        val_loader = DataLoader(val_set,
                                batch_size=args.batch_size,
                                shuffle=False, num_workers=1)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

