import torch
import cv2 

import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask

class MIMIC_Haim(Dataset):
    def __init__(
            self, 
            dataset_path: str = "/sc-resources/dh-mimic/mimic_haim/haim_img_report_lab_target_split.csv",
            split: str = "train",
            split_nr: int = 1, 
            variant: str = "unimodal_1",
            zero_fill_rates: list = [0.3, 0.0],
            seed: int = 42
    ) -> None: 
        super().__init__()

        self.num_modalities = 2
        self.variant = variant

        self.root_img = "/sc-resources/dh-mimic/mimic_cxr_jpg"
        self.root_lab = "/sc-resources/dh-mimic/mimic_haim"
        self.root_notes = "/sc-resources/dh-mimic/mimic_notes"
        # labels
        self.lab_labels = ["Glucose", "Potassium", "Sodium", "Chloride", "Creatinine", 
                            "Urea Nitrogen", "Bicarbonate", "Anion Gap", "Hemoglobin", 
                            "Hematocrit", "Magnesium", "Platelet Count", "Phosphate", 
                            "White Blood Cells", "Calcium", "MCH", "Red Blood Cells", 
                            "MCHC", "MCV", "RDW", "Neutrophils", "Vancomycin"]
        self.clf_labels = ["Fracture", "Enlarged Cardiomediastinum", "Consolidation", "Atelectasis", 
                           "Edema", "Cardiomegaly", "Lung Lesion", "Lung Opacity", 
                           "Pneumonia", "Pneumothorax"]

        # load dataframe
        self.haim_dataset_full = pd.read_csv(dataset_path).reset_index(drop=True)
        self.haim_dataset = self.haim_dataset_full[(self.haim_dataset_full[f"split_{str(split_nr)}"] == split)].reset_index(drop=True)

        # image, lab paths
        self.img_paths = [f"{self.root_img}/{self.haim_dataset.at[i, 'img_folder']}/{self.haim_dataset.at[i, 'img_id']}.jpg" for i in range(len(self.haim_dataset))]
        self.img_paths = [path.replace('files', 'files_resized') for path in self.img_paths]
        self.lab_paths = [f"{self.root_lab}/laboratory_for_images/tensors/{self.haim_dataset.at[i, 'laboratory_filename']}.pt" for i in range(len(self.haim_dataset))]
        self.max_first_dim = 358  # for uniform sequence length lab values

        # targets
        self.targets = self.haim_dataset.loc[:, self.clf_labels]  # .fillna(0)
        self.targets[self.targets == -1] = float("nan")
        self.targets = torch.from_numpy(self.targets.values)

        # normalization
        self.mean_lab = torch.tensor([-0.1233, -0.1240, -0.1253, -0.1279, -0.0924, -0.1000, -0.1161, -0.1126,
                                    -0.1195, -0.1202, -0.1201, -0.1227, -0.0554, -0.1181, -0.0974, -0.1257,
                                    -0.1216, -0.1142, -0.1280, -0.1026, -0.0400, -0.0786])
        self.std_lab = torch.tensor([0.7411, 0.7486, 0.7349, 0.7511, 0.7265, 0.7224, 0.7729, 0.7883, 0.7470,
                                    0.7460, 0.7447, 0.7531, 0.6917, 0.7317, 0.7716, 0.7548, 0.7266, 0.7640,
                                    0.7519, 0.7280, 0.6880, 0.5649])
        self.img_mean = torch.tensor([120.16300964355469])
        self.img_std = torch.tensor([77.33121490478516])

        # missing data 
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.haim_dataset),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return len(self.haim_dataset)

    def __getitem__(self, idx):
        # Image
        img = torch.from_numpy(cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)).float().unsqueeze(0)
        img = (img - self.img_mean) / self.img_std

        # Laboratory value
        lab = torch.load(self.lab_paths[idx], weights_only=True)
        lab = torch.nan_to_num(lab, 0.0)
        padding_needed = self.max_first_dim - lab.shape[0]
        lab = F.pad(lab, (0, 0, 0, padding_needed), 'constant', 0)
        lab = (lab - self.mean_lab) / self.std_lab

        # Target
        target = self.targets[idx]

        # full multimodal datareturn
        datareturn = [img, lab]

        # Missing Data
        if not "unimodal" in self.variant:
            datareturn[0] = apply_missing_mask(datareturn[0], self.zero_fill_masks[idx, 0])
            datareturn[1] = apply_missing_mask(datareturn[1], self.zero_fill_masks[idx, 1])

        return {
            "image": datareturn[0], 
            "lab": datareturn[1].float(), 
            "target": target.float()
        }