import torch
import copy 

import numpy as np

from typing import List, Tuple, Union

def create_missing_data_masks(
    total_samples: int,
    num_modalities: int,
    missing_rates: List[float],
    random_seed: int = None
) -> Tuple[np.ndarray, dict]:
    """
    Creates masks for simulating missing data across different modalities in a dataset.
    
    Parameters:
    -----------
    total_samples : int
        Total number of samples in the dataset
    num_modalities : int
        Number of modalities in the dataset
    missing_rates : List[float]
        List of rates for missing modalities [one_missing_rate, two_missing_rate, ...]
        The rates should sum to <= 1.0
    random_seed : int, optional
        Seed for reproducibility
        
    Returns:
    --------
    Tuple[np.ndarray, dict]
        - numpy array of shape (total_samples, num_modalities) with boolean masks
        - dictionary containing statistics about the missing data distribution
    
    Example:
    --------
    >>> masks, stats = create_missing_data_masks(1000, 3, [0.2, 0.1])
    >>> # This will create masks where:
    >>> # 70% of samples have no missing modalities
    >>> # 20% of samples have one missing modality
    >>> # 10% of samples have two missing modalities
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Validate inputs
    if not 0 <= sum(missing_rates) <= 1:
        raise ValueError("Sum of missing rates must be between 0 and 1")
    if len(missing_rates) >= num_modalities:
        raise ValueError("Number of missing rates must be less than number of modalities")
        
    # Initialize masks
    zero_fill_masks = np.zeros((total_samples, num_modalities), dtype=bool)
    
    # Calculate sample counts for each category
    no_missing = int((1 - sum(missing_rates)) * total_samples)
    missing_counts = [int(rate * total_samples) for rate in missing_rates]
    
    current_idx = 0
    
    # Samples with no missing modalities
    for i in range(no_missing):
        zero_fill_masks[i] = [False] * num_modalities
    current_idx += no_missing
    
    # Create masks for each missing category
    for n_missing, count in enumerate(missing_counts, 1):
        for i in range(count):
            if current_idx >= total_samples:
                break
                
            # Select n_missing random modalities to be missing
            missing_modalities = np.random.choice(
                num_modalities, 
                size=n_missing, 
                replace=False
            )
            mask = [False] * num_modalities
            for mod_idx in missing_modalities:
                mask[mod_idx] = True
            zero_fill_masks[current_idx] = mask
            current_idx += 1
    
    # Shuffle the masks
    np.random.shuffle(zero_fill_masks)

    # ensure that there is no case where all modalities are missing
    assert not np.all(np.all(zero_fill_masks, axis=1)), "All modalities are missing in some samples"

    # Calculate statistics
    stats = {
        'total_samples': total_samples,
        'no_missing': np.sum(np.all(~zero_fill_masks, axis=1)),
    }
    
    for n_missing in range(1, len(missing_rates) + 1):
        stats[f'{n_missing}_missing'] = np.sum(np.sum(zero_fill_masks, axis=1) == n_missing)
        
    for i in range(num_modalities):
        stats[f'modality_{i}_missing'] = np.sum(zero_fill_masks[:, i])
    
    # Print statistics
    for key, value in stats.items():
        print(f"{key}: {value} / {stats['total_samples']}")
        
    return zero_fill_masks, stats

def apply_missing_mask(
    data: Union[torch.Tensor, np.ndarray],
    mask: np.ndarray,
    fill_value: Union[float, int] = float('nan')
) -> Union[torch.Tensor, np.ndarray]:
    """
    Applies a missing data mask to the input data.
    
    Parameters:
    -----------
    data : Union[torch.Tensor, np.ndarray]
        Input data to mask
    mask : np.ndarray
        Boolean mask indicating which samples should be marked as missing
    fill_value : Union[float, int], optional
        Value to use for masked elements (default: nan)
        
    Returns:
    --------
    Union[torch.Tensor, np.ndarray]
        Data with missing values masked
    """
    mask = torch.full_like(data, mask, dtype=torch.bool)
    masked_data = copy.copy(data)
    masked_data[mask] = fill_value
    return masked_data