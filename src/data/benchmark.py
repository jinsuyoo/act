import os

from src.data import srdata


class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(args, name=name, train=train, benchmark=True)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')
