import time
from os import path as osp
from multiprocessing import Process, Queue

import imageio


class Saver:
    def __init__(self):
        self.n_processes = 8

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None:
                        break
                    imageio.imwrite(filename, tensor)

        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]

        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(
        self, save_dir, filename, save_list, scale, rgb_range):
        filename = osp.join(save_dir, '{}_x{}_'.format(filename, scale))

        postfix = ('ACT', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].mul(255 / rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
