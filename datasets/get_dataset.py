from torch.utils.data import DataLoader
from datasets.dataset import DB_ModelNet40, DB_7Scene, DB_ICL_NUIM

def get_dataset(opts, db_nm, partition, batch_size, shuffle, drop_last, is_normal=False, cls_idx=-1, n_cores=1):
    loader, db = None, None
    if db_nm == "modelnet40":
        db = DB_ModelNet40(opts, partition, is_normal, cls_idx=cls_idx)
        loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=n_cores)
    if db_nm == "scene7":
        db = DB_7Scene(opts, partition, is_normal, cls_idx=cls_idx)
        loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=n_cores)
    if db_nm == "icl_nuim":
        db = DB_ICL_NUIM(opts, partition, is_normal, cls_idx=cls_idx)
        loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=n_cores)
    return loader, db
