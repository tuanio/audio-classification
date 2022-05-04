import torch
from utils import labels2int
import pytorch_lightning as pl
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torch.nn.utils.rnn import pad_sequence


class SpeechCommandDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "./",
        batch_size: int = 64,
        n_fft: int = 200,
        pin_memory=False,
    ):
        super().__init__()
        self.root = root

        self.batch_size = batch_size
        self.transform = T.MelSpectrogram(n_fft)

        self.pin_memory = pin_memory

    def prepare_data(self):
        SPEECHCOMMANDS(self.root, download=True)

    def setup(self, stage):
        self.train_set = SPEECHCOMMANDS(self.root, subset="training")
        self.test_set = SPEECHCOMMANDS(self.root, subset="testing")
        self.val_set = SPEECHCOMMANDS(self.root, subset="validation")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.__collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=self.__collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=self.__collate_fn,
            pin_memory=self.pin_memory,
        )

    def __collate_fn(self, batch):
        mel_specs = [self.transform(i[0]).squeeze().permute(1, 0) for i in batch]
        labels = torch.LongTensor([labels2int.get(i[2]) for i in batch])

        mel_specs = pad_sequence(mel_specs, batch_first=True)
        # thêm 3 channel
        mel_specs = torch.stack([mel_specs, mel_specs, mel_specs], dim=1)

        return mel_specs, labels
