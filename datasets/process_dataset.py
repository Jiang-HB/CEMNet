import h5py, sys, os, glob, numpy as np, pdb
sys.path.insert(0, "..")
from datasets.utils.gen_normal import gen_normal
from datasets.utils.commons import save_data
import datasets.se_math.mesh as mesh, datasets.se_math.transforms as transforms, torchvision, torch

# ============ ModelNet40 ==============
def process_modelnet40():
    db_dir = "/test/datasets/registration/modelnet40"
    res = {"train": {"normal_pcs": [], "lbs": []},
           "test": {"normal_pcs": [], "lbs": []}}
    for partition in ["train", "test"]:
        for db_path in glob.glob(os.path.join(db_dir, "ply_data_%s*.h5" % partition)):
            f = h5py.File(db_path)
            normal_pcs_batch = np.concatenate([f['data'][:], f['normal'][:]], axis=2) # (B, N, 6)
            lbs_batch = f['label'][:].astype('int64') # (B, 1)
            f.close()
            res[partition]["normal_pcs"].append(normal_pcs_batch)
            res[partition]["lbs"].append(lbs_batch)
            print("--- %s, finished. ---" % (db_path))
        res[partition]["normal_pcs"] = np.concatenate(res[partition]["normal_pcs"], 0)
        res[partition]["lbs"] = np.concatenate(res[partition]["lbs"], 0).reshape(-1)

    save_data(os.path.join(db_dir, "modelnet40_normal_n2048.pth"), res)

# ============ 7Scene ==============
def get_cls(cls_path):
    cls = [line.rstrip('\n') for line in open(cls_path)]
    cls.sort()
    cls_to_lb = {cls[i]: i for i in range(len(cls))}
    return cls, cls_to_lb

def process_7scene():
    db_dir = "/test/datasets/registration/7scene"
    res = {"train": {"normal_pcs": [], "lbs": []},
           "test": {"normal_pcs": [], "lbs": []}}
    transform = torchvision.transforms.Compose([transforms.Mesh2Points(),
                                                transforms.OnUnitCube(),
                                                transforms.Resampler(2048)])
    for partition in ["train", "test"]:
        pc_dir = os.path.join(db_dir, "7scene")
        cls_path = os.path.join(db_dir, "categories/7scene_%s.txt" % (partition))
        cls_nms, nms_to_lbs = get_cls(cls_path)
        for cls_nm in cls_nms:
            pc_paths = os.path.join(pc_dir, cls_nm, "*.ply")
            lb = nms_to_lbs[cls_nm]
            for pc_path in glob.glob(pc_paths):
                pc = mesh.plyread(pc_path)
                pc = transform(pc).numpy().transpose([1, 0]) # [3, 2048]
                normal_pc = gen_normal(pc[None, :])
                res[partition]["normal_pcs"].append(normal_pc)
                res[partition]["lbs"].append(lb)
                print("--- %s, finished. ---" % (pc_path))
        res[partition]["normal_pcs"] = np.concatenate(res[partition]["normal_pcs"], 0)
        res[partition]["lbs"] = np.asarray(res[partition]["lbs"])
    save_data(os.path.join(db_dir, "7scene_normal_n2048.pth"), res)

# ============ ICL-NUIM ==============
def process_icl_nuim():
    db_dir = "/test/datasets/registration/icl_nuim"
    res = {"train": {"normal_pcs": [], "lbs": []},
           "test": {"normal_pcs": [], "lbs": []}}
    transform = torchvision.transforms.Compose([transforms.OnUnitCube(),
                                                transforms.Resampler(2048)])
    for partition in ["train", "test"]:
        db_path = os.path.join(db_dir, "icl_nuim_%s.h5" % partition)
        if partition == "train":
            f = h5py.File(db_path, "r")
            pcs = f['points'][...]
            for idx, pc in enumerate(pcs):
                pc = transform(torch.FloatTensor(pc)).numpy().transpose([1, 0])
                normal_pc = gen_normal(pc[None, :])
                res[partition]["normal_pcs"].append(normal_pc)
                res[partition]["lbs"].append(idx)
        elif partition == "test":
            f = h5py.File(db_path, "r")
            pcs = f['source'][...]
            # tgt_pcs = f['target'][...]
            # transforms = f['transform'][...]
            for idx in range(len(pcs)):
                pc = pcs[idx]
                pc = transform(torch.FloatTensor(pc)).numpy().transpose([1, 0])
                normal_pc = gen_normal(pc[None, :])
                res[partition]["normal_pcs"].append(normal_pc)
                res[partition]["lbs"].append(idx)
        res[partition]["normal_pcs"] = np.concatenate(res[partition]["normal_pcs"], 0)
        res[partition]["lbs"] = np.asarray(res[partition]["lbs"])

    save_data(os.path.join(db_dir, "ic_nuim_normal_n2048.pth"), res)

if __name__ == '__main__':
    process_modelnet40()
    # process_7scene()
    # process_icl_nuim()