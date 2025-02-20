import torch
import copy 
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from codefiles.datasets.ch_sims import CH_Sims

""" Not tested yet, should not work as is."""

class CH_SIMS_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 batch_size: int = 64, 
                 num_workers: int = 4,
                 num_modalities: int = 3,
                 missing: dict = {"missing_train": [], "missing_valid": []}):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.missing = missing

    def prepare_data(self):
        # sum of zero_fill_rates must be less than or equal to 1, assert
        assert sum(self.missing["missing_train"]) <= 1, "sum of zero_fill_rates must be less than or equal to 1"
        assert sum(self.missing["missing_valid"]) <= 1, "sum of zero_fill_rates must be less than or equal to 1"
        self.train_dataset = CH_Sims(split="train", zero_fill_rates=self.missing["missing_train"], split="train")
        self.val_dataset = CH_Sims(split="val", zero_fill_rates=self.missing["missing_valid"], split="val")
        self.test_dataset = CH_Sims(split="test", zero_fill_rates=self.missing["missing_valid"], split="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=False)
    
    def collate_fn(self, batch):
        # Collate function to handle variable-length sequences
        videos = [item['video'] for item in batch]
        audios = [item['audio'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])

        # Pad videos
        videos_padded = self.pad_video_sequences(videos)
        # Pad audios
        audios = [audio.transpose(0, 1) for audio in audios]
        audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True).transpose(1, 2).squeeze()
        # Stack input IDs and attention masks
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
        final_text = torch.cat([input_ids_padded, attention_masks_padded], 1)

        #return {
        #    'video': videos_padded,               # Tensor: (Batch, Max Frames, Channels, Height, Width)
        #    'audio': audios_padded,               # Tensor: (Batch, Max Audio Length)
        #    'input_ids': input_ids_padded,        # Tensor: (Batch, Max Seq Length)
        #    'attention_mask': attention_masks_padded,  # Tensor: (Batch, Max Seq Length)
        #    'text': final_text,
        #    'label': labels                       # Tensor: (Batch)
        #}

        data = [final_text, videos_padded, audios_padded]
        data_missing = copy.deepcopy(data)
    
        return [data, labels, data_missing]
    
    def pad_video_sequences(self, videos):
        batch_size = len(videos)
        channels, frames, height, width = videos[0].shape

        # Find max number of frames
        max_frames = max([video.shape[1] for video in videos])

        # Initialize tensor with zeros
        padded_videos = torch.zeros((batch_size, channels, max_frames, height, width))

        for i, video in enumerate(videos):
            frames = video.shape[1]
            padded_videos[i, :, :frames, :, :] = video

        return padded_videos

