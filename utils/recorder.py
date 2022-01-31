import numpy as np, os
from utils.commons import save_data, line_plot
from collections import defaultdict

class Recorder:
    def __init__(self, opts):
        self.results_dir = opts.results_dir
        self.results = defaultdict(list)
        self.opts = opts

    def add_res(self, res):
        for key, res in res.items():
            if isinstance(res, list):
                self.results[key].extend(res)
            else:
                self.results[key].append(res)

    def add_reslist(self, reslist):
        for key, res in reslist.items():
            self.results[key].append(res)

    def line_plot(self, nms):
        for nm in nms:
            res = self.results[nm]
            title = "%s_seed%d" % (nm, self.opts.seed)
            line_plot(np.arange(len(res)), res, title, os.path.join(self.opts.results_dir, "%s.pdf" % title))

    def save(self, file_nm=None):
        if file_nm is not None:
            save_data(os.path.join(self.opts.results_dir, "%s_%d.pth" % (file_nm, self.opts.seed)), self.results)
        else:
            save_data(os.path.join(self.opts.results_dir, "res_seed%d.pth" % (self.opts.seed)), self.results)
