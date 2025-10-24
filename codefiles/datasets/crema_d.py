import os 
import torch 
import torchvision
import torchaudio
import pandas as pd 
import numpy as np
import torch.nn as nn 
import subprocess

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder, AudioDecoder

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask


class CREMAD(Dataset):
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

        self.cremad_dataset_full = pd.read_csv(f"/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/splits.csv")
        self.cremad_dataset = self.cremad_dataset_full[self.cremad_dataset_full[f"split_{split_nr}"] == split]

        #self.cremad_dataset_val = self.cremad_dataset_full[self.cremad_dataset_full[f"split_{split_nr}"] == "val"]
        #self.cremad_dataset_test = self.cremad_dataset_full[self.cremad_dataset_full[f"split_{split_nr}"] == "test"]
        #if split == "val":
        #    self.cremad_dataset = self.cremad_dataset_test
        #if split == "train":
        #    self.cremad_dataset = pd.concat([self.cremad_dataset, self.cremad_dataset_val])
        
        self.cremad_dataset.reset_index(inplace=True)

        self.path_videos = f"/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/VideoFlash"
        self.path_audios = f"/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/AudioWAV"  # AudioMP3

        self.target_dict_24 = { 
            "ANG_LO": 0,
            "ANG_MD": 1,
            "ANG_HI": 2,
            "ANG_XX": 3,
            "DIS_LO": 4,
            "DIS_MD": 5,
            "DIS_HI": 6,
            "DIS_XX": 7,
            "FEA_LO": 8,
            "FEA_MD": 9,
            "FEA_HI": 10,
            "FEA_XX": 11,
            "HAP_LO": 12,
            "HAP_MD": 13,
            "HAP_HI": 14,
            "HAP_XX": 15,
            "NEU_LO": 16,
            "NEU_MD": 17,
            "NEU_HI": 18,
            "NEU_XX": 19,
            "SAD_LO": 20,
            "SAD_MD": 21,
            "SAD_HI": 22,
            "SAD_XX": 23,
        }

        self.target_dict_6 = {
            "ANG": 0,
            "DIS": 1,
            "FEA": 2,
            "HAP": 3,
            "NEU": 4,
            "SAD": 5,
        }

        self.max_video_len = 146
        self.max_audio_len = 100000
        self.take_every = 4
        self.final_max_video_len = 32
        self.video_transform = torchvision.transforms.Resize((224, 224), antialias=False)

        self.vision_mean = torch.tensor([57.580039978027344, 79.20903778076172, 35.25766372680664])
        self.vision_std = torch.tensor([57.199344635009766, 68.4931411743164, 35.61355972290039])
        self.audio_mean = torch.tensor([-4.301025910535827e-06])
        self.audio_std = torch.tensor([0.016652824357151985])

        # tensor([ 96.1255, 131.8849,  57.6400])
        # tensor([41.2798, 28.4811, 24.7251])
        #self.vision_mean = torch.tensor([96.1255, 131.8849, 57.6400])
        #self.vision_std = torch.tensor([41.2798, 28.4811, 24.7251])

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.cremad_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=159,  # 512 (win_length) - 353 (overlap)
        )

    def __len__(self):
        return len(self.cremad_dataset)

    def __getitem__(self, idx):
        # load video and audio 
        videofilename = self.cremad_dataset.iloc[idx]["video"]
        audiofilename = self.cremad_dataset.iloc[idx]["audio"]
        audio_path = os.path.join(self.path_audios, audiofilename + ".wav")
        video_path = os.path.join(self.path_videos, videofilename + ".flv")

        # Video
        try:
            decoder_video = VideoDecoder(video_path.replace(".flv", ".mp4"), device="cpu")
        except Exception as e:
            print(f"Exception for video {video_path}: {e}")
            subprocess.run(['ffmpeg', '-i', video_path, video_path.replace(".flv", ".mp4"), '-y'], check=True, capture_output=True, text=True)
            decoder_video = VideoDecoder(video_path.replace(".flv", ".mp4"), device="cpu")
        current_video_len = len(decoder_video)
        image = decoder_video[current_video_len//2]
        image = self.video_transform(image.float())
        image = (image - self.vision_mean.view(1, 3, 1, 1)) / self.vision_std.view(1, 3, 1, 1)
        image = image.squeeze(0)

        video = decoder_video[:self.max_video_len]
        if video.shape[0] < self.max_video_len:
            video = torch.cat([video, torch.zeros(self.max_video_len - video.shape[0], video.shape[1], video.shape[2], video.shape[3])])
        video = video[::self.take_every]
        video = video[:self.final_max_video_len]
        video = self.video_transform(video.float())
        video = (video - self.vision_mean.view(1, 3, 1, 1)) / self.vision_std.view(1, 3, 1, 1)
        
        # Audio
        decoder_audio = AudioDecoder(audio_path, sample_rate=16000)
        audio_data = decoder_audio.get_all_samples().data  # (num_samples, num_channels)
        audio = audio_data.mean(dim=0, keepdim=True)  # Stereo -> Mono
        audio = audio[:, :self.max_audio_len]
        if audio.shape[1] < self.max_audio_len:
            pad_len = self.max_audio_len - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        
        audio_spectrogram = self.spectrogram_transform(audio)

        target_len = 299
        if audio_spectrogram.shape[2] > target_len:
            audio_spectrogram = audio_spectrogram[:, :, :target_len]
        elif audio_spectrogram.shape[2] < target_len:
            pad_len = target_len - audio_spectrogram.shape[2]
            audio_spectrogram = torch.nn.functional.pad(audio_spectrogram, (0, pad_len))
        
        audio_spectrogram = torchaudio.transforms.AmplitudeToDB()(audio_spectrogram)
        audio = (audio - self.audio_mean) / self.audio_std

        # Missing Data
        if not "unimodal" in self.variant:
            image = apply_missing_mask(image, self.zero_fill_masks[idx, 0])
            audio_spectrogram = apply_missing_mask(audio_spectrogram, self.zero_fill_masks[idx, 1])

        # Target
        target_filename_full = videofilename.split(".")[0]
        target_filename = "_".join(target_filename_full.split("_")[-2:])
        target_24 = torch.tensor(self.target_dict_24[target_filename])
        target_6 = torch.tensor(self.target_dict_6[target_filename.split("_")[0]])

        item = {
            "vision": image, 
            "audio": audio_spectrogram, 
            "target": target_6.long(),
        }

        # Add missing mask if needed, to be applied in collate_fn
        #if not "unimodal" in self.variant:
        #    item["missing_mask"] = self.zero_fill_masks[idx]

        return item