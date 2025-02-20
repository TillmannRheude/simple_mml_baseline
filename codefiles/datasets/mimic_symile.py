import torch
import warnings 

import numpy as np
import pandas as pd

from scipy import signal
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask

class MIMIC_Symile(Dataset):
    def __init__(self, 
        dataset_path: str = "/sc-resources/dh-mimic/mimic_symile/mimic_symile/",
        split: str = "train",
        split_nr: int = 1,
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0, 0.0, 0.0],
        seed: int = 42
    ) -> None: 
        super().__init__()

        self.num_modalities = 3
        self.variant = variant

        # whole df
        self.symile_dataset = pd.read_csv(f"{dataset_path}/{split}_split{split_nr}.csv")

        # whole lab values
        lab_cols = [
            51221, 51265, 50912, 50971, 51222, 51301, 51249, 51279, 51250, 51248, 
            51277, 51006, 50983, 50902, 50882, 50868, 50931, 50960, 50893, 50970, 
            51237, 51274, 51275, 51146, 51256, 51254, 51200, 51244, 52172, 50934, 
            51678, 50947, 50861, 50878, 50813, 50863, 50885, 50820, 50862, 50802, 
            50821, 50804, 50818, 52075, 52073, 52074, 52069, 51133, 50910, 52135
        ]
        lab_cols = [str(col) for col in lab_cols]

        # modalities as tensors
        self.cxrs = torch.load(f"{dataset_path}/{split}/cxr_{split}{split_nr}.pt")
        self.ecgs = torch.load(f"{dataset_path}/{split}/ecg_{split}{split_nr}.pt")
        self.labs = torch.load(f"{dataset_path}/{split}/labs_missingness_{split}{split_nr}.pt")
        self.hadm_ids = torch.load(f"{dataset_path}/{split}/hadm_id_{split}{split_nr}.pt")
        self.labs_full = torch.tensor(self.symile_dataset[lab_cols].values)

        # targets
        self.clf_labels = ["Fracture", "Enlarged Cardiomediastinum", "Consolidation", "Atelectasis", 
                           "Edema", "Cardiomegaly", "Lung Lesion", "Lung Opacity", 
                           "Pneumonia", "Pneumothorax"]
        self.symile_targets_df = self.symile_dataset[["Fracture", "Enlarged Cardiomediastinum", "Consolidation", "Atelectasis", 
                    "Edema", "Cardiomegaly", "Lung Lesion", "Lung Opacity", 
                    "Pneumonia", "Pneumothorax"]]
        self.targets = torch.tensor(self.symile_targets_df[self.clf_labels].values)
        self.targets[self.targets == -1] = float("nan")

        # normalization, images already normalized
        self.mean_ecg = torch.tensor([[-3.6575e-05], [-1.2807e-04], [-9.1384e-05], [ 7.4234e-05], [-1.0590e-04], [ 2.6687e-05], 
                                      [-1.2348e-04], [-4.4517e-04], [-5.6398e-04], [-3.6743e-04], [-1.7678e-04], [-1.0786e-04]])
        self.std_ecg = torch.tensor([[0.1523], [0.1539], [0.1703], [0.1283], [0.1434], [0.1436], 
                                     [0.2025], [0.2912], [0.3085], [0.2530], [0.2199], [0.1915]])
        self.mean_labs = torch.tensor([0.7007])
        self.std_labs = torch.tensor([0.4579])

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.symile_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )
    
    def __len__(self):
        return self.cxrs.shape[0]

    def remove_baseline_wander(self, ecg_data, sampling_rate=500, cutoff_freq=0.5):
        """
        Remove baseline wander from ECG data
        
        Parameters:
        -----------
        ecg_data : numpy.ndarray
            Input ECG data with shape (n_leads, n_timepoints)
        sampling_rate : float, optional
            Sampling rate of the ECG signal (default: 500 Hz)
        cutoff_freq : float, optional
            High-pass filter cutoff frequency (default: 0.5 Hz)
        
        Returns:
        --------
        numpy.ndarray
            Baseline wander removed ECG data with same shape as input
        """
        # Ensure input is numpy array
        ecg_data = np.asarray(ecg_data)
        
        # Create high-pass filter
        nyquist_freq = 0.5 * sampling_rate
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        # Design Butterworth high-pass filter
        order = 5
        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
        
        # Apply filter to each lead
        filtered_data = np.zeros_like(ecg_data)
        for j in range(ecg_data.shape[0]):  # Iterate over leads
            filtered_data[j, :] = signal.filtfilt(b, a, ecg_data[j, :])
        
        return filtered_data

    def __getitem__(self, idx):
        # CXR
        img = self.cxrs[idx]

        # Laboratory value
        lab = self.labs[idx].float()
        warnings.warn("Lab values are not normalized.")

        # ECG 
        ecg = self.ecgs[idx].permute(0, 2, 1)
        ecg = self.remove_baseline_wander(ecg, sampling_rate=500, cutoff_freq=0.5)
        ecg = torch.tensor(ecg)
        ecg = (ecg - self.mean_ecg) / self.std_ecg

        # Target
        target = self.targets[idx]

        # full multimodal datareturn
        datareturn = [img, lab, ecg]

        # Missing Data
        if not "unimodal" in self.variant:
            datareturn[0] = apply_missing_mask(datareturn[0], self.zero_fill_masks[idx, 0])
            datareturn[1] = apply_missing_mask(datareturn[1], self.zero_fill_masks[idx, 1])
            datareturn[2] = apply_missing_mask(datareturn[2], self.zero_fill_masks[idx, 2])

        return {
            "image": datareturn[0].float(),
            "lab": datareturn[1], 
            "ecg": datareturn[2].float().squeeze(),
            "target": target.float()
        }

