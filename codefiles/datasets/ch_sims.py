import torch

import pandas as pd
from transformers import BertTokenizer
from torchvision import transforms
import torchaudio
from torchaudio.transforms import Resample
from torchvision.io import read_video
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask


""" Not tested yet, should not work as is."""


class CH_Sims(Dataset):
    def __init__(self,
                 dataset_path: str = "/sc-projects/sc-proj-ukb-cvd/projects/mml_tr/IMDer/dataset/CHSIMS/", 
                 split: str = "train",
                 missing: bool = False,):
        self.datadir = dataset_path
        self.annotations = pd.read_csv(f"{dataset_path}label.csv")
        self.annotations = self.annotations[self.annotations.iloc[:, 8] == split]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_text_length = 128

        # Video preprocessing transforms
        self.video_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            #transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
            #                     std=[0.22803, 0.22145, 0.216989]),
        ])

        # Audio resampler
        self.audio_resampler = Resample()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Video
        video_foldername = str(self.annotations.iloc[idx][0])  # Adjust column name as needed
        video_filename = str(self.annotations.iloc[idx][1]).zfill(4) + ".mp4"
        video_path = os.path.join(self.datadir, video_foldername, video_filename)
        video_frames, audio_frames, info = read_video(video_path, pts_unit='sec')
        video_frames = self.process_video(video_frames)

        # Process audio frames
        audio_frames = self.process_audio(audio_frames, info['audio_fps'])

        # Process text data
        text = self.annotations.iloc[idx][2]
        input_ids, attention_mask = self.process_text(text)

        # Get label (if available)
        label = self.annotations.iloc[idx][3]  # Adjust column name as needed
        label = torch.tensor(label)  # dtype=torch.long

        return {
            'video': video_frames,                # Tensor: (Frames, Channels, Height, Width)
            'audio': audio_frames,                # Tensor: (Audio Length)
            'input_ids': input_ids,               # Tensor: (Seq Length)
            'attention_mask': attention_mask,     # Tensor: (Seq Length)
            'label': label                        # Tensor: ()
        }

    def process_video(self, video_frames):
        # videoframes shape Frames, Height, Width, Channels
        # to Channels, Frames, Height, Width
        video_frames = video_frames.permute(-1, 0, 1, 2).float()
        # Apply preprocessing to each frame
        video_frames = self.video_transform(video_frames)
        return video_frames

    def process_audio(self, audio_frames, orig_freq):
        audio_frames = self.audio_resampler(audio_frames)
        # Convert to mono if stereo
        if audio_frames.shape[0] > 1:
            audio_frames = torch.mean(audio_frames, dim=0, keepdim=True)
        return audio_frames

    def process_text(self, text):
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return input_ids, attention_mask

