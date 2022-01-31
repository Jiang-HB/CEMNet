from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import numpy as np, pdb, pickle

def load_data(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

class BaseDataset(Dataset):
    def __init__(self, opts, partition, is_normal=False, cls_idx=-1):
        self.opts = opts
        self.cls_idx = cls_idx
        self.n_points = opts.db.n_points
        self.n_sub_points = opts.db.n_sub_points
        self.partition = partition
        self.gaussian_noise = opts.db.gaussian_noise
        self.unseen = opts.db.unseen
        self.factor = opts.db.factor
        self.is_normal = is_normal
        self.pcs = None

    def load_data(self, opts, partition):
        db_path = opts.db.path
        db = load_data(db_path)[partition]
        if opts.infos.db_nm == "scene7":
            db["normal_pcs"] = db["normal_pcs"].transpose(0, 2, 1)
        pcs = db["normal_pcs"][:, :, :3]
        lbs = db["lbs"]
        if self.cls_idx != -1:
            pcs = pcs[lbs == self.cls_idx]
            lbs = lbs[lbs == self.cls_idx]
        return pcs, lbs

    def jitter_pointcloud(self, pc, sigma=0.01, clip=0.05):
        pc += np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
        return pc

    def farthest_subsample_points(self, pc1, pc2, n_sub_points=768):
        pc1, pc2 = pc1.T, pc2.T
        nbrs1 = NearestNeighbors(n_neighbors=n_sub_points, algorithm='auto',
                                 metric=lambda x, y: minkowski(x, y)).fit(pc1)
        random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
        idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((n_sub_points,))
        nbrs2 = NearestNeighbors(n_neighbors=n_sub_points, algorithm='auto',
                                 metric=lambda x, y: minkowski(x, y)).fit(pc2)
        random_p2 = random_p1
        idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((n_sub_points,))
        return pc1[idx1, :].T, pc2[idx2, :].T

    def __getitem__(self, item):
        pc = self.pcs[item][:self.opts.db.n_points] # (N, 3)
        if self.partition != 'train':
            np.random.seed(item)

        angle_x = np.random.uniform(0., np.pi / self.factor)
        angle_y = np.random.uniform(0., np.pi / self.factor)
        angle_z = np.random.uniform(0., np.pi / self.factor)
        t_lb = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])

        pc1 = pc.T # (3, N)
        r_lb = np.array([angle_z, angle_y, angle_x])
        pc2 = Rotation.from_euler('zyx', r_lb).apply(pc1.T).T + np.expand_dims(t_lb, axis=1)

        pc1 = np.random.permutation(pc1.T).T
        pc2 = np.random.permutation(pc2.T).T

        if self.gaussian_noise:
            pc1 = self.jitter_pointcloud(pc1)
            pc2 = self.jitter_pointcloud(pc2)

        if self.n_points != self.n_sub_points:
            pc1, pc2 = self.farthest_subsample_points(pc1, pc2, n_sub_points = self.n_sub_points)

        return pc1.astype('float32'), pc2.astype('float32'), r_lb.astype('float32'), t_lb.astype('float32')

    def __len__(self):
        return len(self.pcs)