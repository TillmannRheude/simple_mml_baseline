import torch
import torch.nn.functional as F
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask


class INSPECT(Dataset):
    def __init__(
        self, 
        dataset_path: str = "/path/to/data/inspect/",
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0],
        seed: int = 42,
        raw_vision: bool = False,
        use_preprocessed: bool = False
    ) -> None: 
        super().__init__()

        self.num_modalities = 2
        self.variant = variant
        self.raw_vision = raw_vision
        self.use_preprocessed = use_preprocessed

        splits_df = pd.read_pickle(f"/path/to/data/inspect/cv_splits.pkl")
        splits_df = splits_df[splits_df[f"split_{split_nr}"] == split]

        self.targets_df = pd.read_pickle(dataset_path + "PE_targets.pkl")
        self.targets_df["label"] = self.targets_df["label"].apply(lambda x: False if x == "False" else True)
        self.targets_df["label"] = self.targets_df["label"].astype(bool)
        self.ehr_df = pd.read_pickle(dataset_path + "ehr_motor_embeddings.pkl")
        self.images_pathlist = glob.glob(dataset_path + "vision_med3dvlm_embeddings/*.npz")

        self.targets_df = self.targets_df[self.targets_df["patient_id"].isin(splits_df["person_id"])]
        self.ehr_df = self.ehr_df[self.ehr_df["patient_ids"].isin(splits_df["person_id"])]
        self.image_ids = [file for file in self.images_pathlist if file.split("/")[-1].split(".")[0].split("_")[0] in self.ehr_df["image_id"].tolist()]

        if self.raw_vision:
            if self.use_preprocessed:
                self.images_pathlist = glob.glob("/sc-scratch/sc-scratch-dh-fu-swp-25/inspect_preprocessed/*.pt")
            else:
                self.images_pathlist = glob.glob("/sc-scratch/sc-scratch-dh-fu-swp-25/inspect/inspect2/CTPA/*.nii.gz")
            self.images_pathlist = [file for file in self.images_pathlist if file.split("/")[-1].split(".")[0] in self.ehr_df["image_id"].tolist()]
            self.image_ids = [id.split(".")[0].split("/")[-1] for id in self.images_pathlist]

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.image_ids),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Image
        if self.raw_vision:
            if self.use_preprocessed:
                image_path = f"/sc-scratch/sc-scratch-dh-fu-swp-25/inspect_preprocessed/{image_id}.pt"
                image = torch.load(image_path)
                image = image[None, None, ...]
                image = F.interpolate(image, size=(32, 112, 112), mode="trilinear").squeeze(0)
            else:
                image_path = f"/sc-scratch/sc-scratch-dh-fu-swp-25/inspect/inspect2/CTPA/{image_id}.nii.gz"
                image = nib.load(image_path).get_fdata()
                image = torch.from_numpy(image).float()
        else:
            image_path = f"{image_id}"
            image = np.load(image_path)["arr_0"]
            image = torch.from_numpy(image).float()  # (768, 2, 4, 4)
            image = torch.mean(image, dim=(1, 2, 3))  # (768)

        # EHR
        image_id = image_id.split("/")[-1].split("_")[0]
        ehr = self.ehr_df.loc[self.ehr_df["image_id"] == image_id, "embeddings"].values[0]
        ehr = torch.tensor(ehr)

        # Target
        target = self.targets_df.loc[self.targets_df["image_id"] == image_id, "label"].values[0]
        target = torch.from_numpy(np.array(target)).float().unsqueeze(0)

        # full multimodal datareturn
        datareturn = [image, ehr, target]

        # Missing Data
        if not "unimodal" in self.variant:
            datareturn[0] = apply_missing_mask(datareturn[0], self.zero_fill_masks[idx, 0])
            datareturn[1] = apply_missing_mask(datareturn[1], self.zero_fill_masks[idx, 1])

        return {
            "image": datareturn[0].contiguous(), 
            "ehr": datareturn[1].contiguous(), 
            "target": target
        }