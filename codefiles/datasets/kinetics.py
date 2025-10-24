import os 
import torch 
import torchvision
import torchaudio
import pandas as pd 
from torch.utils.data import Dataset
from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask
from torchcodec.decoders import VideoDecoder, AudioDecoder

class Kinetics(Dataset):

    """ 
    https://github.com/cvdfoundation/kinetics-dataset
    """
    
    def __init__(
        self, 
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0],
        seed: int = 42,
        num_classes: int = 400  # 400 / 600 / 700
    ) -> None: 
        super().__init__()

        self.num_modalities = 2
        self.variant = variant
        self.split = split

        self.kinetics_dir = f"/sc-projects/sc-proj-ukb-cvd/projects/data/kinetics/kinetics-dataset/k{num_classes}/"
        self.kinetics_dir_videos_train = self.kinetics_dir + f"train/"
        self.kinetics_dir_videos_val = self.kinetics_dir + f"val/"
        self.kinetics_dir_videos_test = self.kinetics_dir + f"test/"
        self.kinetics_dir_videos_replacement = self.kinetics_dir + "replacement/replacement_for_corrupted_k400/"
        self.kinetics_dir_videos_preprocessed = self.kinetics_dir + "preprocessed"
        self.kinetics_dir_csv = self.kinetics_dir + "annotations/" + f"test.csv" if split == "test" else f"{self.kinetics_dir}/annotations/kinetics_trainval_splits.csv"
        self.list_of_videos_train = os.listdir(self.kinetics_dir_videos_train)
        self.list_of_videos_val = os.listdir(self.kinetics_dir_videos_val)
        self.list_of_videos_test = os.listdir(self.kinetics_dir_videos_test)

        # further corrupted files 
        corrupted_file_list = [
            "G1nWDa3jfBs_000168_000178.mp4",
            "wL1Bit-Gv40_000305_000315.mp4",
            "bOU2oGVBM_o_000030_000040.mp4",
            "QhF1i23vwps_000379_000389.mp4",
            "Df6CGDjUkAA_000151_000161.mp4",
            "gKBhQ-oe_9Q_000177_000187.mp4",
            "N74EWF0fs5c_000182_000192.mp4",
            "QzmhrYx15_E_000059_000069.mp4",
            "_cbZlhduYJY_000503_000513.mp4",
            "YCQlaH_Vy8I_000245_000255.mp4",
            "B6GxQKcL7IY_000213_000223.mp4",
            "28bTQiuymgs_000031_000041.mp4",
            "lm6qgrfJGmw_000027_000037.mp4",
            "MVWayhNpHr0_000065_000075.mp4",
            "E2kUsRIj4tM_000317_000327.mp4",
            "cgaMptpoY6Y_000095_000105.mp4",
            "rba-NkJjSNg_000167_000177.mp4",
            "aCcAcCE7Ixo_000034_000044.mp4",
            "ZtCk_0cMZ9U_000347_000357.mp4",
            "E2NeSaQieHk_000087_000097.mp4",
            "GN37yfNvQwM_000132_000142.mp4",
            "J5xNIJlfBAw_000156_000166.mp4",
            "GkGS69GCx4Q_000319_000329.mp4",
            "UfkWCSho6qg_000233_000243.mp4",
            "8iED0lhyrN8_000038_000048.mp4",
            "du6bfkBEfVs_000155_000165.mp4",
            "d_vQWquKtBg_000015_000025.mp4",
            "fXRNY6-s-7U_000112_000122.mp4",
            "VwVufLo7Mo0_000015_000025.mp4",
            "mtYFNsRcxY4_000063_000073.mp4",
            "uGQgxFHemyA_000059_000069.mp4",
            "co50KUHacYw_000005_000015.mp4",
            "w5ax4GiTkKg_000088_000098.mp4",
            "Zs0_2GMEPXo_000054_000064.mp4",
            "5l5Pdd96Pao_000161_000171.mp4",
            "cswoyWn6S_o_000167_000177.mp4",
            "-jGxlNQKkeo_000092_000102.mp4",
            "wKsuyr6Xk30_000094_000104.mp4",
            "LpaI8vPC0us_000288_000298.mp4",
            "jHODDw65G4A_000085_000095.mp4",
            "jQPSzhKkk-g_000028_000038.mp4",
            "eiZ8Hzc7FPU_000080_000090.mp4",
            "99ABSLQdgUc_000046_000056.mp4",
            "i-gzh_BPDa8_000154_000164.mp4",
            "uz5cIbBTf4Y_000049_000059.mp4",
            "cqUusXBODuw_001046_001056.mp4",
            "5_gyoV_sQXU_000001_000011.mp4",
            "1_nxfkY76mk_000001_000011.mp4",
            "3VvkoFtPCCU_000045_000055.mp4",
            "f4eb0wOlspM_000053_000063.mp4",
            "fNFXTBUF3nY_000230_000240.mp4",
            "4bhvIHWADVU_000022_000032.mp4",
            "J1x0XlWa6HM_000147_000157.mp4",
            "6-DyF8umej8_000097_000107.mp4",
            "FPZplLfJ6J8_000148_000158.mp4",
            "GajaQD6qRkw_000057_000067.mp4",
            "wL1Bit-Gv40_000305_000315.mp4",
            'gSjHCbS_u0Y_000058_000068.mp4',
            'oyj6TFAxpiw_000229_000239.mp4',
            'U_vYW90hFds_000042_000052.mp4',
            'lk5Ap5gZNj0_000009_000019.mp4',
            'UhkXeiMm_s4_000017_000027.mp4',
            'aj1bmhf-IyU_000118_000128.mp4',
            'SZtj2TEWiHc_000195_000205.mp4',
            'bgCrldl9pQ8_000027_000037.mp4',
            'efTAWmCkLKE_000418_000428.mp4',
            'ixQrfusr6k8_000001_000011.mp4',
            '_M6Ko0yRfD4_000097_000107.mp4',
            'pDPbETciXhw_000167_000177.mp4',
            'Ud9poTS_URE_000014_000024.mp4',
            'QfuO07EqYhI_000054_000064.mp4',
            '_6uq-NBo3Bk_000012_000022.mp4',
            'UdMCrOIUQrw_000005_000015.mp4',
            'UnGxFi0H5UA_000065_000075.mp4',
            'nfjWfoyGApo_000220_000230.mp4',
            'UbQsEI_KkBs_000049_000059.mp4',
            'I0luMKjIZyg_000422_000432.mp4',
            'sAA809R_u1E_000077_000087.mp4',
            'XFkykETgkoo_002967_002977.mp4',
            '305P2f9_lko_004145_004155.mp4',
            'IhanWvpHGu8_001243_001253.mp4',
            'Lw14NH9kAqE_000759_000769.mp4',
            'B4bn9G6__sY_000086_000096.mp4',
            'BvBVQmm2RcM_000082_000092.mp4',
            'y7cYaYX4gdw_000047_000057.mp4',
            'jJFqy6yiXzQ_000024_000034.mp4',
            'CxjipYE57Yo_000199_000209.mp4',
            '084k_RL3ApU_000109_000119.mp4',
            '74iWTzKsHPI_000110_000120.mp4',
            '2xWiEVNUvhE_000064_000074.mp4',
            'z35QkFl2tyU_000026_000036.mp4',
            'kinMMqkswUk_000120_000130.mp4',
            'geIeo7Y2xMg_000032_000042.mp4',
            'v4GSOx9EHpI_000084_000094.mp4',
            'UNiiEuISS3A_000115_000125.mp4',
            'ngJEC9BMQcQ_000085_000095.mp4',
            'jTVOrxyPSLc_000138_000148.mp4',
            '3yiTf7Q3FUs_000025_000035.mp4',
            'r0Q-lWtZGBk_000010_000020.mp4',
            'QKSClBuBPGQ_000003_000013.mp4',
            'bx6eP0p9cPk_000015_000025.mp4',
            'k-mz0d7KM18_000018_000028.mp4',
            'o_Kfrq_hTic_000025_000035.mp4',
            'aUxbLh4Js28_000252_000262.mp4',
            'Yr82KH4w-l4_000031_000041.mp4',
            'QNRVCEFp2fY_000049_000059.mp4',
            '3TXs3zYjv2c_000124_000134.mp4',
            'c-vOHTFXEz0_000001_000011.mp4',
            'W5o4aJs_tcg_000003_000013.mp4',
            'el4g_SqZrNE_000020_000030.mp4',
            'gXb1keQirmg_000017_000027.mp4',
            'cw9jFsrP4A4_000024_000034.mp4',
            'YKedtoKNyvg_000010_000020.mp4',
            'jYOZ9sVYXJM_000078_000088.mp4',
            'anppA4Z0ggc_000004_000014.mp4',
            'Y6f1eQJeLno_000009_000019.mp4',
            'cJcNloXVQgY_000003_000013.mp4',
            'Ujj5r_9aWvI_000010_000020.mp4',
            'bklH07wHVuw_000000_000010.mp4',
            'R-iIR5VmA5M_000277_000287.mp4',
            'l3cb5semtKw_000030_000040.mp4',
            'bSyYyzLZqw8_000004_000014.mp4',
            'eZHnt63IVgw_000003_000013.mp4',
            'iv-PDkv1Jhc_000033_000043.mp4',
            'Qd7uHH_R_Vc_000004_000014.mp4',
            '2vvxK-Yaxgg_000012_000022.mp4',
            'mDoBBtoqWsA_000001_000011.mp4',
            'TUQJckFdm30_000024_000034.mp4',
            'ZJ8M2Gi6LCM_000001_000011.mp4',
            'mqZHDq6H8JU_000006_000016.mp4',
            'chc7dbiZGoE_000012_000022.mp4',
            'rgXo71dN_3c_000032_000042.mp4',
            'tRLsWm93X0o_000001_000011.mp4',
            'U4lx3a9A_ag_000001_000011.mp4',
            'f50m2RxLG7U_000006_000016.mp4',
            'Xv__0v03BiY_000327_000337.mp4',
            'plTGBi6c-PE_000003_000013.mp4',
            'ZYOGH7szNq0_000081_000091.mp4',
            'kHwmtbitWSI_000001_000011.mp4',
            'iORqFBGSoqk_000042_000052.mp4',
            'e_3dsgTRJNc_000002_000012.mp4',
            'Yg0tDrq1Qng_000023_000033.mp4',
            'fozN3imw53M_000106_000116.mp4',
            'wAi33oNRB9I_000051_000061.mp4',
            '2BvI5wyu29w_000001_000011.mp4',
            'lgCKjPGnB1s_000028_000038.mp4',
            'g0OMTL0EcPA_000008_000018.mp4',
            'a3_TgjVgEtU_000007_000017.mp4',
            'VfoIi5dRSec_000097_000107.mp4',
            'X2ybNDFKnGc_000001_000011.mp4',
            'kgJPbRSkdtw_000001_000011.mp4',
            'w4X72YQCtM8_000006_000016.mp4',
            'VEpXASQ7seY_000004_000014.mp4',
            'ZmY7OdK8ODY_000005_000015.mp4',
            'WbcufY5RRJA_000343_000353.mp4',
            'eP8k6GotHtw_000005_000015.mp4',
            'ghxiG0Dnrt0_000004_000014.mp4',
            'VjJ4kiaPdNY_000004_000014.mp4',
            'fTYNfiXKaaE_000366_000376.mp4',
            '3__grgZHp74_000032_000042.mp4',
            'YpSTiYzK34E_000001_000011.mp4',
            '9mec-ejMBqQ_000015_000025.mp4',
            'yZ8b0DLeJcU_000005_000015.mp4',
            '-C_s9oUnFek_000003_000013.mp4',
            'ZuCWeUQeAkI_000007_000017.mp4',
            'piYKF7yTMOA_000001_000011.mp4',
            '1V-YjQsZjeM_000010_000020.mp4',
            'riKnfJj545o_000035_000045.mp4',
            'NdjLKFhn9j0_000004_000014.mp4',
            'gbdN0yYj_pQ_000002_000012.mp4',
            'cJaHXPAWPp0_000223_000233.mp4',
            'lKHLOjqyXs8_000021_000031.mp4',
            '56E-e8wzAkY_000182_000192.mp4',
            'jZNgDwNjoW8_000005_000015.mp4',
            'SyNDPEY4PmQ_000001_000011.mp4',
            'Sl7hJv0mafE_000002_000012.mp4',
            'iEzpnawqqZ8_000036_000046.mp4',
            '2qN_X0TpTNs_000003_000013.mp4',
            'RDgQIuHTWGM_000006_000016.mp4',
            'klEvBVoT4FI_000047_000057.mp4',
            'jLe_mgtzwDc_000014_000024.mp4',
            'UqqVA3Ujhjc_000020_000030.mp4',
            'XDGhv1b5zfA_000004_000014.mp4',
            'wKBg2hlH9vY_000018_000028.mp4',
            'JfpMbV8SeDg_000280_000290.mp4',
            'mYHR1Dk_mCI_000073_000083.mp4',
            'DABrjXSnNzg_000062_000072.mp4',
            'bLw2JvO8SV4_000002_000012.mp4',
            '1MfnMW_EnQw_000002_000012.mp4',
            'VXiloz_G1A0_000006_000016.mp4',
            'H8NBB1qswB0_000003_000013.mp4',
            'Fv2cEF3qdVM_000023_000033.mp4',
            'kbXP2cAhoPY_000027_000037.mp4',
            'D1elvFRNXZM_000013_000023.mp4',
            'T7r_72lrZ_g_000032_000042.mp4',
            '1VkXX8YSb4w_000120_000130.mp4',
            'RpsVCX_N1Xg_000002_000012.mp4',
            '6OcY0XYhkms_000042_000052.mp4',
            'GA9--6Q_DMU_000008_000018.mp4',
            'LCkrcw67MD4_000007_000017.mp4',
            'k4onnrKfAQM_000001_000011.mp4',
            'PvdFJB769_4_000001_000011.mp4',
            'oqHQJpdgsME_000001_000011.mp4',
            'upa9PQmviqY_000042_000052.mp4',
            'UsJFVEKIhzw_000008_000018.mp4',
            'YuVnKG6dgB4_000147_000157.mp4',
            'Eym3-bOno1U_000001_000011.mp4',
            'ON_dl3S166c_000004_000014.mp4',
            'Nwl5k-RhU7Q_000065_000075.mp4',
            'HS1s802yOCs_000001_000011.mp4',
            'IIKAz1SSjfE_000024_000034.mp4',
            'Es4uCQUV5II_000002_000012.mp4',
            'Swiqnw1B0lU_000002_000012.mp4',
            'cIc21QwXdgE_000004_000014.mp4',
            'pUNcqaSxpbk_000001_000011.mp4',
            'UNx9YpFzgHI_000053_000063.mp4',
            'nWYQYQLzYiM_000045_000055.mp4',
            "vXQqmXXrA74_000268_000278.mp4"
        ]
        
        if split == "test":
            self.list_of_videos = self.list_of_videos_test
        else:
            self.list_of_videos = self.list_of_videos_train + self.list_of_videos_val
        
        if not os.path.exists(self.kinetics_dir_videos_preprocessed):
            os.makedirs(self.kinetics_dir_videos_preprocessed)

        self.kinetics_csv = pd.read_csv(self.kinetics_dir_csv)
        self.kinetics_csv = self.kinetics_csv[self.kinetics_csv[f"split_{split_nr}"] == split] if split != "test" else self.kinetics_csv

        # only include videos in list_of_videos
        # https://github.com/cvdfoundation/kinetics-dataset/issues/14
        self.kinetics_csv["time_start"] = self.kinetics_csv["time_start"].astype(str).str.zfill(6)
        self.kinetics_csv["time_end"] = self.kinetics_csv["time_end"].astype(str).str.zfill(6)
        self.kinetics_csv["full_file_name"] = self.kinetics_csv["youtube_id"] + "_" + self.kinetics_csv["time_start"] + "_" + self.kinetics_csv["time_end"] + ".mp4"
        self.kinetics_csv = self.kinetics_csv[self.kinetics_csv["full_file_name"].isin(self.list_of_videos)]
        self.kinetics_csv = self.kinetics_csv[~self.kinetics_csv["full_file_name"].isin(corrupted_file_list)].reset_index(drop=True)

        all_unique_classes = sorted(self.kinetics_csv["label"].dropna().unique())
        class_to_int = {class_name: idx for idx, class_name in enumerate(all_unique_classes)}
        self.kinetics_csv["label_number"] = self.kinetics_csv["label"].map(class_to_int)

        self.max_video_len = 32
        self.take_every = 8
        self.video_transform = torchvision.transforms.Resize((224, 224), antialias=False)
        self.sample_rate = 16000
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=159,  # 512 (win_length) - 353 (overlap)
        )

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.kinetics_csv),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return len(self.kinetics_csv)

    def __getitem__(self, idx):
        # load video and audio
        videofilename = self.kinetics_csv["full_file_name"][idx]
        if self.split == "test":
            video_path = os.path.join(self.kinetics_dir_videos_test, videofilename)
        else:
            video_path = os.path.join(self.kinetics_dir_videos_train, videofilename)
            if not os.path.exists(video_path):
                video_path = os.path.join(self.kinetics_dir_videos_val, videofilename)

        # if videofilename also exists in replacement, use this one instead
        if videofilename in os.listdir(self.kinetics_dir_videos_replacement):
            video_path = os.path.join(self.kinetics_dir_videos_replacement, videofilename)

        try: 
            image = torch.load(os.path.join(self.kinetics_dir_videos_preprocessed, videofilename.replace(".mp4", "_image.pt")))
            audio_spectrogram = torch.load(os.path.join(self.kinetics_dir_videos_preprocessed, videofilename.replace(".mp4", "_audio.pt")))
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
            audio_spectrogram = torchaudio.transforms.AmplitudeToDB()(audio_spectrogram)

            torch.save(image, os.path.join(self.kinetics_dir_videos_preprocessed, videofilename.replace(".mp4", "_image.pt")))
            torch.save(audio_spectrogram, os.path.join(self.kinetics_dir_videos_preprocessed, videofilename.replace(".mp4", "_audio.pt")))

        # Normalization
        # TODO 

        # Missing data
        if not "unimodal" in self.variant:
            image = apply_missing_mask(image, self.zero_fill_masks[idx, 0])
            audio_spectrogram = apply_missing_mask(audio_spectrogram, self.zero_fill_masks[idx, 1])

        target = torch.tensor(self.kinetics_csv["label_number"][idx]) 

        item = {
            "vision": image, 
            "audio": audio_spectrogram, 
            "target": target.long(),
            "target_name": self.kinetics_csv["label"][idx]
        }

        return item