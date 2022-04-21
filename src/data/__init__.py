from src.data.datamodule import SRDataModule


def create_datamodule(args):
    return SRDataModule(args)