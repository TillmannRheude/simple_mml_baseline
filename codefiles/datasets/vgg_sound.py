import os 
import torch 
import torchvision
import torchaudio
import pandas as pd 
import numpy as np
import torch.nn as nn 
from torch.utils.data import Dataset
from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask
from torchcodec.decoders import VideoDecoder, AudioDecoder


class VGGSound(Dataset):
    def __init__(
        self, 
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0],
        seed: int = 42,
    ) -> None: 
        super().__init__()

        self.num_modalities = 2
        self.variant = variant

        self.vgg_dir = "/sc-projects/sc-proj-ukb-cvd/projects/data/datasets--Loie--VGGSound/blobs/"
        self.vgg_dir_videos = self.vgg_dir + "scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/"
        self.vgg_dir_videos_preprocessed = "/sc-projects/sc-proj-ukb-cvd/projects/data/datasets--Loie--VGGSound/preprocessed"
        self.vgg_dir_csv = self.vgg_dir + "vggsound_fullfilename_cv_splits.csv"

        self.vgg_csv = pd.read_csv(self.vgg_dir_csv)
        self.vgg_csv = self.vgg_csv[self.vgg_csv["full_file_name"] != "ljjUj5fQZgs_000450.mp4"]  # drop corrupted files 
        self.vgg_csv = self.vgg_csv[self.vgg_csv["class_name"] != "extending ladders"]  # drop "extending ladders" class 

        all_unique_classes = sorted(self.vgg_csv["class_name"].dropna().unique())
        class_to_int = {class_name: idx for idx, class_name in enumerate(all_unique_classes)}

        self.vgg_dataset = self.vgg_csv[self.vgg_csv[f"cv_split_{split_nr}"] == split]  # 
        self.vgg_dataset = self.vgg_dataset[self.vgg_dataset['full_file_name'].apply(lambda x: isinstance(x, str))]
        self.vgg_dataset = self.vgg_dataset.reset_index(drop=True)

        self.file_paths = self.vgg_dataset.loc[:, "full_file_name"]
        self.targets = torch.tensor([class_to_int[class_name] for class_name in self.vgg_dataset["class_name"]])

        self.max_video_len = 32
        self.take_every = 16
        self.video_transform = torchvision.transforms.Resize((224, 224), antialias=False)
        self.sample_rate = 16000
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=159,  # 512 (win_length) - 353 (overlap)
        )

        self.image_mean = torch.tensor([106.5172,  99.6191,  93.6093])
        self.image_std = torch.tensor([72.6139, 70.8019, 71.3603])
        self.audio_mean = torch.tensor([-26.2448])
        self.audio_std = torch.tensor([22.7864])

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.vgg_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return len(self.vgg_dataset)

    def __getitem__(self, idx):
        # load video and audio 
        videofilename = self.file_paths[idx]
        video_path = os.path.join(self.vgg_dir_videos, videofilename)

        try: 
            image = torch.load(os.path.join(self.vgg_dir_videos_preprocessed, videofilename.replace(".mp4", "_image.pt")))
            audio_spectrogram = torch.load(os.path.join(self.vgg_dir_videos_preprocessed, videofilename.replace(".mp4", "_audio.pt")))
        except FileNotFoundError:
            # Decoders
            decoder_video = VideoDecoder(video_path)
            decoder_audio = AudioDecoder(video_path, sample_rate=self.sample_rate)
            
            # Video
            video = decoder_video[::self.take_every]
            video = video[:self.max_video_len]
            video = self.video_transform(video.float())
            pad_len = self.max_video_len - video.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len, video.shape[1], video.shape[2], video.shape[3])
                pad = torch.zeros(pad_shape, dtype=video.dtype, device=video.device)
                video = torch.cat([video, pad], dim=0)
            
            # Image from video
            current_video_len = len(decoder_video)
            image = decoder_video[current_video_len//2]
            image = self.video_transform(image.float())
            image = image.squeeze(0)

            # Audio
            audio_data = decoder_audio.get_all_samples().data  # (num_samples, num_channels)
            audio = audio_data.mean(dim=0, keepdim=True)
            audiolen = 10 * self.sample_rate
            if audio.shape[1] > audiolen:
                audio = audio[:, :audiolen]
            elif audio.shape[1] < audiolen:
                padding_needed = audiolen - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding_needed))

            # Spectrogram from audio 
            audio_spectrogram = self.spectrogram_transform(audio)
            target_len = 1004
            if audio_spectrogram.shape[2] > target_len:
                audio_spectrogram = audio_spectrogram[:, :, :target_len]
            elif audio_spectrogram.shape[2] < target_len:
                pad_len = target_len - audio_spectrogram.shape[2]
                audio_spectrogram = torch.nn.functional.pad(audio_spectrogram, (0, pad_len))
            audio_spectrogram = torchaudio.transforms.AmplitudeToDB()(audio_spectrogram)

            torch.save(image, os.path.join(self.vgg_dir_videos_preprocessed, videofilename.replace(".mp4", "_image.pt")))
            torch.save(audio_spectrogram, os.path.join(self.vgg_dir_videos_preprocessed, videofilename.replace(".mp4", "_audio.pt")))

        image = (image - self.image_mean.view(3, 1, 1)) / self.image_std.view(3, 1, 1)
        audio_spectrogram = (audio_spectrogram - self.audio_mean.view(1, 1)) / self.audio_std.view(1, 1)

        # Missing data 
        if not "unimodal" in self.variant:
            image = apply_missing_mask(image, self.zero_fill_masks[idx, 0])
            audio_spectrogram = apply_missing_mask(audio_spectrogram, self.zero_fill_masks[idx, 1])

        target = self.targets[idx]  #torch.tensor(self.vgg_dataset["class_num"][idx])

        item = {
            "vision": image, 
            "audio": audio_spectrogram, 
            "target": target.long(),
            "target_name": self.vgg_dataset["class_name"][idx]
        }

        # Add missing mask if needed, to be applied in collate_fn
        if not "unimodal" in self.variant:
            item["missing_mask"] = self.zero_fill_masks[idx]

        return item


def collate_fn(batch):
    # Filter out samples that failed to load
    batch = [item for item in batch if item is not None]
    if not batch:
        return {
            "vision": torch.tensor([]),
            "audio": torch.tensor([]),
            "target": torch.tensor([])
        }
    
    max_video_len = 32  # batch[0]["vision"].shape[0]
    audiolen = 10 * 16000  #batch[0]["audio"].shape[1]#

    processed_videos = []
    processed_audios = []
    targets = []

    for item in batch:
        video, audio = item['vision'], item['audio']

        # Video processing
        pad_len = max_video_len - video.shape[0]
        if pad_len > 0:
            pad_shape = (pad_len, video.shape[1], video.shape[2], video.shape[3])
            pad = torch.zeros(pad_shape, dtype=video.dtype, device=video.device)
            video = torch.cat([video, pad], dim=0)

        # Audio processing
        if audio.shape[1] > audiolen:
            audio = audio[:, :audiolen]
        elif audio.shape[1] < audiolen:
            padding_needed = audiolen - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding_needed))

        # Apply missing data masks if they exist
        if "missing_mask" in item:
            video = apply_missing_mask(video, item["missing_mask"][0])
            audio = apply_missing_mask(audio, item["missing_mask"][1])

        processed_videos.append(video)
        processed_audios.append(audio)
        targets.append(item['target'])

    padded_videos = torch.stack(processed_videos, dim=0)
    padded_audios = torch.stack(processed_audios, dim=0)
    targets = torch.stack(targets, dim=0)

    return {
        "vision": padded_videos,
        "audio": padded_audios,
        "target": targets.long()
    }