from datasets.base_dataset import BaseDataset
from utils.commons import load_data

class DB_ModelNet40(BaseDataset):
    def __init__(self, opts, partition, is_normal=False, cls_idx=-1):
        super(DB_ModelNet40, self).__init__(opts, partition, is_normal, cls_idx)
        self.pcs, self.lbs = self.load_data(opts, partition) # (B, N, 3), (B, )

    def load_data(self, opts, partition):
        db_path = opts.db.path
        db = load_data(db_path)[partition]
        pcs = db["normal_pcs"][:, :, :3]
        lbs = db["lbs"]
        if self.unseen:
            if self.partition == 'test':
                pcs = pcs[lbs >= 20]
                lbs = lbs[lbs >= 20]
            elif self.partition == 'train':
                pcs = pcs[lbs < 20]
                lbs = lbs[lbs < 20]
        if self.cls_idx != -1:
            pcs = pcs[lbs == self.cls_idx]
            lbs = lbs[lbs == self.cls_idx]
        return pcs, lbs

class DB_7Scene(BaseDataset):
    def __init__(self, opts, partition, is_normal=False, cls_idx=-1):
        super(DB_7Scene, self).__init__(opts, partition, is_normal, cls_idx)
        self.pcs, self.lbs = self.load_data(opts, partition) # (B, N, 3), (B, )

    def load_data(self, opts, partition):
        db_path = opts.db.path
        db = load_data(db_path)[partition]
        db["normal_pcs"] = db["normal_pcs"].transpose(0, 2, 1)
        pcs = db["normal_pcs"][:, :, :3]
        lbs = db["lbs"]
        if self.cls_idx != -1:
            pcs = pcs[lbs == self.cls_idx]
            lbs = lbs[lbs == self.cls_idx]
        return pcs, lbs

class DB_ICL_NUIM(BaseDataset):
    def __init__(self, opts, partition, is_normal=False, cls_idx=-1):
        super(DB_ICL_NUIM, self).__init__(opts, partition, is_normal, cls_idx)
        self.pcs, self.lbs = self.load_data(opts, partition) # (B, N, 3), (B, )

    def load_data(self, opts, partition):
        db_path = opts.db.path
        db = load_data(db_path)[partition]
        db["normal_pcs"] = db["normal_pcs"].transpose(0, 2, 1)
        pcs = db["normal_pcs"][:, :, :3]
        lbs = db["lbs"]
        if self.cls_idx != -1:
            pcs = pcs[lbs == self.cls_idx]
            lbs = lbs[lbs == self.cls_idx]
        return pcs, lbs