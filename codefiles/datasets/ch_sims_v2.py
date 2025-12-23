import os 
import torch
import torchcodec
import torchaudio

import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask


class CH_Sims_v2(Dataset):  
    def __init__(
        self,
        dataset_path: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/CHSIMS2", 
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0, 0.0],
        seed: int = 42,
        orig_sr: int = 44100,
        target_sr: int = 16000,
        cache_audio_16k: bool = False,
    ) -> None: 
        super().__init__()

        self.num_modalities = 3
        self.variant = variant

        self.datadir = os.path.join(dataset_path, "ch-simsv2s/Preprocessed_224")
        self.chsims_dataset = pd.read_csv(os.path.join(dataset_path, "meta_cv_splits.csv"))
        self.chsims_dataset = self.chsims_dataset[self.chsims_dataset[f"cv_split_{split_nr}"] == split].reset_index(drop=True)

        self.file_paths = []
        for _, row in self.chsims_dataset.iterrows():
            path = os.path.join(self.datadir, row["video_id"], f"{row['clip_id']}.pt")
            self.file_paths.append(path)

        self.folder_names = self.chsims_dataset.loc[:, "video_id"]
        self.file_names = self.chsims_dataset.loc[:, "clip_id"]
        self.targets = self.chsims_dataset.loc[:, "label"].tolist()

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.chsims_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )
        
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.cache_audio_16k = cache_audio_16k
        self.resampler = torchaudio.transforms.Resample(self.orig_sr, self.target_sr)


    def __len__(self):
        return len(self.chsims_dataset)
    
    def _to_audio_16k(self, audio: torch.Tensor) -> torch.Tensor:
        # audio expected: (C, N) or (N,)
        if audio.dim() == 2:
            audio = audio.mean(dim=0)  # (N,)

        # normalize integer PCM to [-1, 1] if needed
        if audio.dtype in (torch.int16, torch.int32, torch.int64, torch.uint8):
            audio = audio.to(torch.float32) / 32768.0
        else:
            audio = audio.to(torch.float32)

        audio_16k = self.resampler(audio)  # (T,)
        return audio_16k

    def __getitem__(self, idx):
        data_path = self.file_paths[idx]
        with open(data_path, 'rb') as f:
            data = torch.load(f)

        video = data['video'].float()
        audio = data['audio']
        audio_16k = data.get('audio_16k', None)
        if audio_16k is None:
            audio_16k = self._to_audio_16k(audio)
            if self.cache_audio_16k:
                data['audio_16k'] = audio_16k
                with open(data_path, 'wb') as f:
                    torch.save(data, f)
        else:
            audio_16k = audio_16k.to(torch.float32)

        input_ids = data['input_ids'].long()
        attention_mask = data['attention_mask'].long()

        # Missing data
        if not "unimodal" in self.variant:
            video = apply_missing_mask(video, self.zero_fill_masks[idx, 0])
            audio = apply_missing_mask(audio_16k, self.zero_fill_masks[idx, 1])
            input_ids = apply_missing_mask(input_ids.float(), self.zero_fill_masks[idx, 2]).long()
            attention_mask = apply_missing_mask(attention_mask.float(), self.zero_fill_masks[idx, 2]).long()

        label = self.targets[idx]
        label = torch.tensor(label).squeeze()

        text = torch.stack([input_ids, attention_mask], dim=0)

        return {
            'video': video,                
            'audio': audio,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'text': text
        }

def collate_fn_v2_preprocessed(batch):
    padded_videos = torch.stack([item['video'] for item in batch], dim=0)

    audio_lens = [item['audio'].shape[-1] for item in batch]
    max_audio_len = max(audio_lens)
    padded_audio_16k = []
    for item in batch:
        a = item['audio']  # (T,)
        pad_len = max_audio_len - a.shape[-1]
        if pad_len > 0:
            a = torch.cat([a, torch.zeros(pad_len, dtype=a.dtype, device=a.device)], dim=-1)
        padded_audio_16k.append(a)
    padded_audios = torch.stack(padded_audio_16k, dim=0)  # (B, T)

    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    labels = torch.stack([item['label'] for item in batch], dim=0)
    text = torch.stack([input_ids, attention_mask], dim=1)

    return {
        'video': padded_videos,
        'audio': padded_audios,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'text': text,
        'label': labels,
    }




# older / experimental versions 
class CH_Sims_v2_raw(Dataset):
    def __init__(
        self,
        dataset_path: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/CHSIMS2", 
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0, 0.0],
        seed: int = 42
    ) -> None: 
        super().__init__()

        self.num_modalities = 3
        self.variant = variant

        self.datadir = os.path.join(dataset_path, "ch-simsv2s/Raw")
        self.chsims_dataset = pd.read_csv(os.path.join(dataset_path, "meta_cv_splits.csv"))
        self.chsims_dataset = self.chsims_dataset[self.chsims_dataset[f"cv_split_{split_nr}"] == split].reset_index(drop=True)

        self.folder_names = self.chsims_dataset.loc[:, "video_id"]
        self.file_names = self.chsims_dataset.loc[:, "clip_id"]
        self.texts = self.chsims_dataset.loc[:, "text"]
        self.targets = self.chsims_dataset.loc[:, "label"]

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
        label = torch.tensor(label).squeeze()

        return {
            'video': video,                
            'audio': audio,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label 
        }

def collate_fn_v2(batch):
    raw_data = True 

    if not raw_data:
        # Find max video length (T) and max audio length (num_samples)
        audio_lens = [item['audio'].shape[-1] for item in batch]
        max_audio_len = max(audio_lens)
    
        # Pad videos
        padded_videos = torch.stack([item['video'] for item in batch], dim=0)
        # Pad audios
        padded_audios = torch.stack([item['audio'] for item in batch], dim=0)

    if raw_data:
        # Find max video length (T) and max audio length (num_samples)
        # video_lens = [item['video'].shape[0] for item in batch]
        audio_lens = [item['audio'].shape[-1] for item in batch]
        take_every = 32
        max_video_len = 33  # 32  # 1067 / 16
        max_audio_len = max(audio_lens)

        # Pad videos
        padded_videos = []
        for item in batch:
            v = item['video']  # (T, C, H, W)
            # only take every 32th frame
            v = v[::take_every, ...]
            pad_len = max_video_len - v.shape[0]
            if pad_len > 0:
                pad = torch.zeros((pad_len, v.shape[1], v.shape[2], v.shape[3]), dtype=v.dtype, device=v.device)
                v = torch.cat([v, pad], dim=0)
            v = v[:32, ...]
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