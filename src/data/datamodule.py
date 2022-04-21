from importlib import import_module

from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader


class SRDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.data_train = args.data_train
        self.data_test = args.data_test

        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        m = import_module('src.data.' + self.data_train.lower())
        dataset = getattr(m, self.data_train)(self.args, name=self.data_train)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=not self.cpu,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.data_test in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
            val_dataset = import_module('src.data.benchmark')
            val_dataset = getattr(val_dataset, 'Benchmark')(
                self.args, train=False, name=self.data_test
            )

        else:
            raise NotImplementedError(
                f'Incorrect test dataset [{self.data_test}] is given'
            )

        return DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.cpu,
            num_workers=self.num_workers,
        )


    def test_dataloader(self):
        if self.data_test in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
            test_dataset = import_module('src.data.benchmark')
            test_dataset = getattr(test_dataset, 'Benchmark')(
                self.args, train=False, name=self.data_test
            )

        else:
            raise NotImplementedError(
                f'Incorrect test dataset [{self.data_test}] is given'
            )

        return DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.cpu,
            num_workers=self.num_workers,
        )
