from utils import labels2int
import pytorch_lightning as pl
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torch.nn.utils.rnn import pad_sequence


class SpeechCommandDataModule(pl.LightningDataModule):
    def __init__(
        self, root: str = "speechcommands/", batch_size: int = 64, n_fft: int = 200
    ):
        self.train_set = SPEECHCOMMANDS(root, download=True, subset="training")
        self.test_set = SPEECHCOMMANDS(root, download=True, subset="testing")
        self.val_set = SPEECHCOMMANDS(root, download=True, subset="validation")

        self.batch_size = batch_size
        self.transform = T.MelSpectrogram(n_fft)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, collate_fn=self.__collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, collate_fn=self.__collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, collate_fn=self.__collate_fn
        )

    def __collate_fn(self, batch):
        mel_specs = [self.transform(i[0]).squeeze().permute(1, 0) for i in batch]
        labels = torch.LongTensor([labels2int.get(i[2]) for i in batch])

        mel_specs = pad_sequence(mel_specs, batch_first=True)
        # thêm channel
        mel_specs = mel_specs.unsqueeze(1)

        return mel_specs, labels
