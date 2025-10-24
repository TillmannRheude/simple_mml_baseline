import os 
import torch
import torchcodec

import pandas as pd

from transformers import BertTokenizer
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask


class CH_Sims(Dataset):
    def __init__(
        self,
        dataset_path: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/CHSIMS", 
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0, 0.0],
        seed: int = 42
    ) -> None: 
        super().__init__()

        self.num_modalities = 3
        self.variant = variant

        self.datadir = dataset_path
        self.chsims_dataset = pd.read_csv(os.path.join(dataset_path, "label_cv_splits.csv"))
        self.chsims_dataset = self.chsims_dataset[self.chsims_dataset[f"cv_split_{split_nr}"] == split].reset_index(drop=True)

        self.folder_names = self.chsims_dataset.loc[:, "file_name"]
        self.file_names = self.chsims_dataset.loc[:, "id"]
        self.texts = self.chsims_dataset.loc[:, "text"]
        self.targets = self.chsims_dataset.loc[:, "unimodal_0"]

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.chsims_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return len(self.chsims_dataset)

    def __getitem__(self, idx):
        video_path = os.path.join(self.datadir, self.folder_names[idx], str(self.file_names[idx]).zfill(4) + ".mp4")

        # Video
        video_decoder = torchcodec.decoders.VideoDecoder(video_path)
        video_frames = video_decoder[:]
        video = video_frames.data  # (T, C, H, W)
        # Video processing
        video = torch.nn.functional.interpolate(
            video.float(), size=(224, 224), mode='bilinear', align_corners=False
        )

        # Audio
        audio_decoder = torchcodec.decoders.AudioDecoder(video_path, sample_rate=44100, num_channels=1)
        audio_samples = audio_decoder.get_all_samples()
        audio = audio_samples.data

        # Text 
        text = self.texts[idx]
        # Text processing 
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Missing data
        if not "unimodal" in self.variant:
            video = apply_missing_mask(video, self.zero_fill_masks[idx, 0])
            audio = apply_missing_mask(audio, self.zero_fill_masks[idx, 1])
            input_ids = apply_missing_mask(input_ids.float(), self.zero_fill_masks[idx, 2])
            attention_mask = apply_missing_mask(attention_mask.float(), self.zero_fill_masks[idx, 2])

        # Get label
        label = self.targets[idx]
        label = torch.tensor(label).unsqueeze(0)

        return {
            'video': video,                
            'audio': audio,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label.long()
        }


def collate_fn(batch):
    # Find max video length (T) and max audio length (num_samples)
    audio_lens = [item['audio'].shape[-1] for item in batch]
    max_video_len = 32  # 500 / 16 = 32
    max_audio_len = max(audio_lens)

    # Pad videos
    padded_videos = []
    for item in batch:
        v = item['video']  # (T, C, H, W)
        # only take every x-th frame
        v = v[::16, ...]
        pad_len = max_video_len - v.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, v.shape[1], v.shape[2], v.shape[3]), dtype=v.dtype, device=v.device)
            v = torch.cat([v, pad], dim=0)
        padded_videos.append(v)
    padded_videos = torch.stack(padded_videos, dim=0)  # (B, T, C, H, W)

    # Pad audios
    padded_audios = []
    for item in batch:
        a = item['audio']  # (num_channels, num_samples)
        pad_len = max_audio_len - a.shape[-1]
        if pad_len > 0:
            pad = torch.zeros((a.shape[0], pad_len), dtype=a.dtype, device=a.device)
            a = torch.cat([a, pad], dim=-1)
        padded_audios.append(a)
    padded_audios = torch.stack(padded_audios, dim=0)  # (B, num_channels, num_samples)

    # Stack input_ids and attention_mask (already padded to max_length=128)
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)  # (B, L)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)  # (B, L)

    # Stack labels
    labels = torch.stack([item['label'] for item in batch], dim=0)

    # Stack text
    text = torch.stack([input_ids, attention_mask], dim=1)

    return {
        'video': padded_videos,
        'audio': padded_audios,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'text': text,
        'label': labels,
    }