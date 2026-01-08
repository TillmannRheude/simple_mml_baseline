import os 
import torch 
import torchvision
import torchaudio
import pandas as pd 
import numpy as np
import subprocess

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder, AudioDecoder
from typing import List
from torchvision import transforms
import torch.nn.functional as F

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask


class CREMAD(Dataset):
    def __init__(
        self,
        split: str = "train",
        split_nr: int = 1,
        variant: str = "unimodal_1",
        zero_fill_rates: List[float] = [0.0],
        seed: int = 42,
        preproc_root: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/preprocessed_aug",
        csv_splits: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/splits.csv",  # _aug
    ) -> None:
        super().__init__()

        # AUG / OGM: 
        # test: https://github.com/GeWu-Lab/OGM-GE_CVPR2022/blob/main/data/CREMAD/test.csv
        # train: https://github.com/GeWu-Lab/OGM-GE_CVPR2022/blob/main/data/CREMAD/train.csv

        self.num_modalities = 2
        self.variant = variant
        self.preproc_root = preproc_root
        self.split = split

        self.cremad_dataset_full = pd.read_csv(csv_splits)

        split_col_name = split_nr if not "aug" in csv_splits else "aug"
        self.cremad_dataset = self.cremad_dataset_full[
            self.cremad_dataset_full[f"split_{split_col_name}"] == split  # aug
        ].reset_index(drop=True)

        self.sample_paths = []
        for _, row in self.cremad_dataset.iterrows():
            video_id = row["video"]
            sample_path = os.path.join(self.preproc_root, f"{video_id}.pt")
            self.sample_paths.append(sample_path)

        # augmentation
        self.vision_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.vision_transform_test = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Missing-data masks are still per-split
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.cremad_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed,
        )

    def __len__(self):
        return len(self.cremad_dataset)

    def __getitem__(self, idx):
        sample_path = self.preproc_root + "/" + self.cremad_dataset.iloc[idx]["video"] + ".pt"
        data = torch.load(sample_path, map_location="cpu")
        
        target_6 = data["target_6"]

        images = data["image"]
        
        if self.split == "train":
            # Random selection for training
            select_index = np.random.choice(len(images), size=1, replace=False)
        else:
            # Deterministic selection for val/test (e.g., middle frame)
            select_index = np.array([len(images) // 2])

        select_index.sort()
        final_images = torch.zeros((1, 3, 224, 224))
        for i in range(1):
            img = images[select_index[i]]
            if self.split == "train":
                img = self.vision_transform_train(img) 
            else:
                img = self.vision_transform_test(img)
            final_images[i] = img
        final_images = final_images.squeeze() 

        audio_spectrogram = data["audio_spectrogram"][None, ...]  # (1, 257, 299)
        
        if "unimodal" not in self.variant:
            final_images = apply_missing_mask(final_images, self.zero_fill_masks[idx, 0])
            audio_spectrogram = apply_missing_mask(audio_spectrogram, self.zero_fill_masks[idx, 1])

        return {
            "vision": final_images,
            "audio": audio_spectrogram,
            "target": target_6,
        }


# older / experimental versions 
class CREMAD_Embeddings(Dataset):
    def __init__(
        self,
        split: str = "train",
        split_nr: int = 1,
        variant: str = "unimodal_1",
        zero_fill_rates: List[float] = [0.0],
        seed: int = 42,
        csv_splits: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/splits.csv",
        preproc_root_embeddings: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/embeddings",
    ) -> None:
        super().__init__()

        self.num_modalities = 2
        self.variant = variant
        self.preproc_root_audio = preproc_root_embeddings + "_audio"
        self.preproc_root_video = preproc_root_embeddings + "_video"
        self.split = split

        self.cremad_dataset_full = pd.read_csv(csv_splits)

        split_col_name = split_nr if not "aug" in csv_splits else "aug"
        self.cremad_dataset = self.cremad_dataset_full[
            self.cremad_dataset_full[f"split_{split_col_name}"] == split  # aug
        ].reset_index(drop=True)

        self.target_dict_6 = {
            "ANG": 0,
            "DIS": 1,
            "FEA": 2,
            "HAP": 3,
            "NEU": 4,
            "SAD": 5,
        }

        # Missing-data masks are still per-split
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.cremad_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed,
        )

    def __len__(self):
        return len(self.cremad_dataset)

    def __getitem__(self, idx):
        filename = self.cremad_dataset.iloc[idx]["video"]

        video_path = self.preproc_root_video + "/" + self.cremad_dataset.iloc[idx]["video"] + ".pt"
        video_emb = torch.load(video_path, map_location="cpu")

        audio_path = self.preproc_root_audio + "/" + self.cremad_dataset.iloc[idx]["audio"] + "_audio.pt"
        audio_emb = torch.load(audio_path, map_location="cpu")
        
        target_6 = self.target_dict_6[filename.split("_")[2]]
        
        if "unimodal" not in self.variant:
            video_emb = apply_missing_mask(video_emb, self.zero_fill_masks[idx, 0])
            audio_emb = apply_missing_mask(audio_emb, self.zero_fill_masks[idx, 1])

        return {
            "vision": video_emb.float(),
            "audio": audio_emb.float(),
            "target": target_6,
        }

class CREMAD_raw(Dataset):
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
        self.cremad_dataset.reset_index(drop=True, inplace=True)

        self.path_videos = f"/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/VideoFlash"
        self.path_audios = f"/sc-projects/sc-proj-ukb-cvd/projects/data/crema-d-mirror/AudioWAV" 

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
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=159,  # 512 (win_length) - 353 (overlap)
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.vision_mean = torch.tensor([57.580039978027344, 79.20903778076172, 35.25766372680664])
        self.vision_std = torch.tensor([57.199344635009766, 68.4931411743164, 35.61355972290039])
        self.audio_mean = torch.tensor([-4.301025910535827e-06])
        self.audio_std = torch.tensor([0.016652824357151985])
        self.audio_spectrogram_mean = torch.tensor([-46.40577535913572])
        self.audio_spectrogram_std = torch.tensor([26.627834912847756])

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.cremad_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
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
        # image = (image - self.vision_mean[:, None, None]) / self.vision_std[:, None, None]
        image = image / 255.0
        image = image.squeeze(0)

        video = decoder_video[:self.max_video_len]
        if video.shape[0] < self.max_video_len:
            video = torch.cat([video, torch.zeros(self.max_video_len - video.shape[0], video.shape[1], video.shape[2], video.shape[3])])
        video = video[::self.take_every]
        video = video[:self.final_max_video_len]
        video = self.video_transform(video.float())
        
        # Audio
        decoder_audio = AudioDecoder(audio_path, sample_rate=16000)
        audio_data = decoder_audio.get_all_samples().data  # (num_channels, num_samples)
        audio = audio_data.mean(dim=0, keepdim=True)  # Stereo -> Mono
        audio = audio[:, :self.max_audio_len]
        if audio.shape[1] < self.max_audio_len:
            pad_len = self.max_audio_len - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        
        target_len = 299
        audio_spectrogram = self.spectrogram_transform(audio)
        if audio_spectrogram.shape[2] > target_len:
            audio_spectrogram = audio_spectrogram[:, :, :target_len]
        elif audio_spectrogram.shape[2] < target_len:
            pad_len = target_len - audio_spectrogram.shape[2]
            audio_spectrogram = torch.nn.functional.pad(audio_spectrogram, (0, pad_len))
        audio_spectrogram = self.amplitude_to_db(audio_spectrogram)
        audio_spectrogram = (audio_spectrogram - self.audio_spectrogram_mean) / self.audio_spectrogram_std
        audio = (audio - self.audio_mean) / self.audio_std

        # Missing Data
        if not "unimodal" in self.variant:
            image = apply_missing_mask(image, self.zero_fill_masks[idx, 0])
            audio_spectrogram = apply_missing_mask(audio_spectrogram, self.zero_fill_masks[idx, 1])

        # Target
        target_filename_full = videofilename.split(".")[0]
        target_filename = "_".join(target_filename_full.split("_")[-2:])
        target_24 = torch.tensor(self.target_dict_24[target_filename])
        target_6 = torch.tensor(self.target_dict_6[target_filename.split("_")[0]]).long()

        return {
            "vision": video,  # image, 
            "audio": audio_spectrogram, 
            "target": target_6,
        }