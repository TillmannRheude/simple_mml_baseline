import torch
import torch.nn.functional as F
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask

def inspect_collate_fn(batch):
    """
    Custom collate function to handle variable z-dimension of raw vision data.
    Pads images to the maximum z-dimension in the batch.
    """
    # Check if raw_vision is used by inspecting the image tensor's dimension.
    # We expect a 3D tensor (H, W, Z) for raw vision.
    is_raw_vision = "image" in batch[0] and batch[0]["image"].ndim == 3

    if not is_raw_vision:
        # If not raw_vision, embeddings are 1D and default collate is fine.
        return default_collate(batch)

    # Separate images for custom padding.
    images = [item.pop("image") for item in batch]
    # Collate the rest of the data (ehr, target) using the default handler.
    collated_rest = default_collate(batch)
    
    # Find the maximum z-dimension in the batch.
    max_z = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        pad_amount = max_z - img.shape[2]
        # The padding tuple is for the last N dimensions, in reverse order.
        # For a 3D tensor (H, W, Z), F.pad expects a 6-element tuple, but
        # a shorter tuple like (pad_z_start, pad_z_end) pads only the last dimension.
        padding = (0, pad_amount) 
        padded_img = F.pad(img, padding, "constant", 0)
        padded_images.append(padded_img)
    
    # Stack the padded images to create the batch tensor.
    collated_images = torch.stack(padded_images)

    # Re-combine the collated images with the rest of the data.
    collated_rest["image"] = collated_images.permute(0, 3, 1, 2)
    return collated_rest


class INSPECT(Dataset):
    def __init__(
        self, 
        dataset_path: str = "/sc-projects/sc-proj-ukb-cvd/projects/data/inspect/",
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0],
        seed: int = 42,
        raw_vision: bool = True
    ) -> None: 
        super().__init__()

        self.num_modalities = 2
        self.variant = variant
        self.raw_vision = raw_vision

        splits_df = pd.read_pickle(f"/sc-projects/sc-proj-ukb-cvd/projects/data/inspect/cv_splits.pkl")
        splits_df = splits_df[splits_df[f"split_{split_nr}"] == split]

        self.targets_df = pd.read_pickle(dataset_path + "PE_targets.pkl")
        self.targets_df["label"] = self.targets_df["label"].apply(lambda x: False if x == "False" else True)
        self.targets_df["label"] = self.targets_df["label"].astype(bool)
        self.ehr_df = pd.read_pickle(dataset_path + "ehr_motor_embeddings.pkl")
        self.images_pathlist = glob.glob(dataset_path + "vision_radfm_embeddings/*.npz")

        # filter for split 
        self.targets_df = self.targets_df[self.targets_df["patient_id"].isin(splits_df["person_id"])]
        self.ehr_df = self.ehr_df[self.ehr_df["patient_ids"].isin(splits_df["person_id"])]
        self.images_pathlist = [file for file in self.images_pathlist if file.split("/")[-1].split(".")[0] in self.ehr_df["image_id"].tolist()]

        # still on a subset of the data
        #self.image_ids = self.targets_df["image_id"].tolist()
        self.image_ids = [id.split(".")[0].split("/")[-1] for id in self.images_pathlist]
        if self.raw_vision:
            self.images_pathlist = glob.glob("/sc-scratch/sc-scratch-dh-fu-swp-25/inspect/inspect2/CTPA/*.nii.gz")
            self.images_pathlist = [file for file in self.images_pathlist if file.split("/")[-1].split(".")[0] in self.ehr_df["image_id"].tolist()]

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
            image_path = f"/sc-scratch/sc-scratch-dh-fu-swp-25/inspect/inspect2/CTPA/{image_id}.nii.gz"
            image = nib.load(image_path).get_fdata()
            image = torch.from_numpy(image).float()
        else:
            image_path = f"/sc-projects/sc-proj-ukb-cvd/projects/data/inspect/vision_radfm_embeddings/{image_id}.npz"
            image = np.load(image_path)["layer_0"]
            image = torch.from_numpy(image).float()

        # EHR
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
            "image": datareturn[0], 
            "ehr": datareturn[1], 
            "target": target
        }