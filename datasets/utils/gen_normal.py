import open3d as o3d, numpy as np, pdb

def gen_normal(pcs):
    """
    :param pcs: shape (B, 3, N), np.array
    :return: shape (B, 6, N)
    """
    normal_pcs = np.zeros([len(pcs), 6, pcs.shape[2]])
    for idx, pc in enumerate(pcs):
        _pc = o3d.geometry.PointCloud()
        _pc.points = o3d.utility.Vector3dVector(pc.transpose([1, 0]))
        o3d.geometry.estimate_normals(_pc, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normal_pc = np.concatenate([np.asarray(_pc.points), np.asarray(_pc.normals)], 1).transpose([1, 0])
        normal_pcs[idx] = normal_pc
    return normal_pcs

if __name__ == '__main__':
    pcs = np.zeros([5, 3, 1024])
    gen_normal(pcs)