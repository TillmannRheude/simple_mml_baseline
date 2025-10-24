# This file is part of RegBN: Batch Normalization of Multimodal Data
# with Regularization.
#
# RegBN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RegBN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RegBN. If not, see <https://www.gnu.org/licenses/>.


from typing import Union, List, Sequence, Tuple
import os
import math
import operator

import torch
import torch.nn as nn
from   torch import Tensor
from functools import reduce


# Check PyTorch version for compatibility
torchV = int(torch.__version__.split('.')[1])
if torchV < 13:
    from torch import inverse as inv_torch
    from torch import svd     as svd_torch
else:
    from torch.linalg import inv as inv_torch
    from torch.linalg import svd as svd_torch

# L1 Loss function
L1torch = nn.L1Loss()
# Small epsilon value for numerical stability
epsilon_ = 1e-8


class RegBN(nn.Module):
    r""" Implements Batch Normalization via Tikhonov Regularizer (RegBN) module
    for multiple modalities, compatible with PyTorch Lightning (no explicit device management).

    RegBN removes the influence of secondary modalities (metadata) from a primary
    modality at the batch level.

    Args:
        modalities_channels (List[int]): List of channel numbers for each modality.
                                         The first element is the primary modality.
        modalities_dims (List[List[Sequence[int]]]): List of spatial dimensions for each modality.
                                                    Corresponds to modalities_channels.
        beta1 (float): Beta1 for Adam-like update of projection matrices (default: 0.9).
        beta2 (float): Beta2 for Adam-like update of projection matrices (default: 0.99).
        momentum (float): Learning rate/momentum for projection matrix updates (default: 0.02).
        normalize_input (bool): Whether to normalize input modalities (default: False).
        normalize_output (bool): Whether to normalize output modalities (default: True).
        affine (bool): If True, normalization layers have learnable affine parameters (default: False).
        sigma_THR (float): Threshold ratio for singular values from SVD (default: 0.0).
        sigma_MIN (float): Minimum cutoff value for singular values (default: 0.0).
        verbose (bool): Print messages during execution (default: False).

    Example:
        >>> batchSize = 100
        >>> # Example with 1 primary (128 channels) and 2 secondary modalities (16, 32 channels)
        >>> modalities = [
        ...     torch.rand([batchSize, 128]),         # Primary modality (f)
        ...     torch.rand([batchSize, 16]),          # Secondary modality 1 (g1)
        ...     torch.rand([batchSize, 8, 8])       # Secondary modality 2 (g2, with spatial dims)
        ... ]
        >>> kwargs = {
        ...     'modalities_channels': [128, 16, 8],
        ...     'modalities_dims': [[], [], [8]], # Spatial dims matching channels (8x8 -> [8])
        ...     'normalize_input': True,
        ...     'normalize_output': True,
        ...     'affine': False,
        ... }
        >>> regbn_module = RegBN(**kwargs)
        ... # Assuming regbn_module is on the correct device via Lightning

        # Training:
        >>> kwargs_train = {"is_training": True, 'n_epoch': 1, 'steps_per_epoch': 100}
        >>> processed_modalities = regbn_module(modalities, **kwargs_train)
        >>> f_n = processed_modalities[0] # Processed primary modality
        >>> g_n_list = processed_modalities[1:] # Processed secondary modalities

        # Validation/Inference:
        >>> kwargs_test = {"is_training": False}
        >>> processed_modalities = regbn_module(modalities, **kwargs_test)
        >>> f_n = processed_modalities[0]
        >>> g_n_list = processed_modalities[1:]

    """
    # __constants__ = ['modalities_channels'] # Cannot make list constant
    modalities_channels: List[int]

    def __init__(self,
                 modalities_channels: List[int],
                 modalities_dims: List[List[Sequence[int]]],
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 momentum: float = 0.02,
                 normalize_input: bool = False,
                 normalize_output: bool = True,
                 affine: bool = False,
                 sigma_THR: float = 0.0,
                 sigma_MIN: float = 0.0,
                 verbose: bool = False,
        ):
        super(RegBN, self).__init__()

        if not modalities_channels or not modalities_dims or len(modalities_channels) < 2:
            raise ValueError("RegBN requires at least one primary and one secondary modality.")
        if len(modalities_channels) != len(modalities_dims):
             raise ValueError("Length of modalities_channels and modalities_dims must match.")

        self.modalities_channels = modalities_channels
        self.modalities_dims = modalities_dims
        self.num_modalities = len(modalities_channels)
        self.num_secondary_modalities = self.num_modalities - 1

        assert 0.0 <= beta1 < 1.0, f"Invalid beta1: {beta1}"
        assert 0.0 <= beta2 < 1.0, f"Invalid beta2: {beta2}"
        assert momentum > 0.0, f"Invalid momentum: {momentum}"

        self.beta1 = beta1
        self.beta2 = beta2
        self.Momentum = momentum # Renamed from momentum to avoid conflict
        self.sigma_THR = sigma_THR
        self.sigma_MIN = sigma_MIN
        self.verbose = verbose

        # Calculate flattened dimensions
        self.modalities_flat_dims = [
            reduce(operator.mul, [ch] + dims, 1)
            for ch, dims in zip(modalities_channels, modalities_dims)
        ]
        self.f_dim_flat = self.modalities_flat_dims[0]
        self.g_dims_flat_list = self.modalities_flat_dims[1:]

        # Store lambda+ values (shared across secondary modalities for now)
        self.register_buffer('lambda_set', torch.tensor(()))
        self.is_nan_ = False

        # LBFGS-solver params for projection matrix estimation
        lbfgs_max_iter = 25
        lbfgs_kwargs = {'max_iter': lbfgs_max_iter,
                        'max_eval': lbfgs_max_iter * 7 // 4,
                        'history_size': lbfgs_max_iter * 20,
                        'line_search_fn': "strong_wolfe",
                        'tolerance_grad': 1e-05,
                        'tolerance_change': 1e-09,
                       }
        # Pass verbose flag to estimator
        self.W_calc = proj_matrix_estimator(lbfgs_kwargs, verbose=self.verbose)

        # Initialize projection matrices (W_i maps g_i to f space)
        # Shape: (g_dim_i, f_dim)
        self.W_list = nn.ParameterList([
            nn.Parameter(torch.zeros(g_dim, self.f_dim_flat))
            for g_dim in self.g_dims_flat_list
        ])

        # Initialize Adam-like parameters for updating projection weights
        self._reset_adam_params()

        # Initialize normalization layers for inputs and outputs
        self.norm_in_list = nn.ModuleList([
            _get_norm_inp(normalize_input, ch, dims, affine)
            for ch, dims in zip(modalities_channels, modalities_dims)
        ])
        self.norm_out_list = nn.ModuleList([
             # Use LayerNorm for output normalization as spatial structure might change
            _get_norm_out(normalize_output, ch, dims, affine)
             for ch, dims in zip(modalities_channels, modalities_dims)
        ])


    def _reset_adam_params(self) -> None:
        """ Resets the Adam-like parameters for updating projection weights. """
        # Using simple lists to store floats/tensors, manage device implicitly
        self.m_list = [0. for _ in range(self.num_secondary_modalities)]
        self.v_list = [0. for _ in range(self.num_secondary_modalities)]
        # Keep track of steps per W matrix for bias correction
        self.adam_steps = [0 for _ in range(self.num_secondary_modalities)]


    @torch.enable_grad() # Ensure gradients are enabled if called externally
    def update_W_i(self,
                   i: int,
                   W_cur_i: torch.Tensor,
                   n_epoch: int # n_epoch is used for bias correction term
        ) -> None:
        """
        Updates the i-th projection matrix (self.W_list[i]) using an Adam-like step.
        Args:
            i (int): Index of the secondary modality and its corresponding W matrix.
            W_cur_i (Tensor): The newly estimated projection matrix for this batch.
            n_epoch (int): Current epoch number (used for bias correction).
                           Note: A step counter might be more accurate than epoch.
        """
        with torch.no_grad():
            # Ensure W_cur_i is on the same device as self.W_list[i]
            if W_cur_i.device != self.W_list[i].device:
                 W_cur_i = W_cur_i.to(self.W_list[i].device)

            # L1 difference between current estimate and running average W
            g = L1torch(self.W_list[i].data, W_cur_i) # Use .data to avoid graph issues

            # Update biased first moment estimate
            self.m_list[i] = self.beta1 * self.m_list[i] + (1.0 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v_list[i] = self.beta2 * self.v_list[i] + (1.0 - self.beta2) * g**2

            # Increment step count for this W
            self.adam_steps[i] += 1
            step = self.adam_steps[i] # Use step count instead of epoch for better bias correction

            # Compute bias-corrected first moment estimate
            mhat = self.m_list[i] / (1.0 - self.beta1**step + epsilon_) # Add epsilon for stability
            # Compute bias-corrected second raw moment estimate
            vhat = self.v_list[i] / (1.0 - self.beta2**step + epsilon_) # Add epsilon for stability

            # Calculate update step size (eta)
            # Ensure vhat is non-negative before sqrt
            vhat = torch.clamp(vhat, min=0.0)
            eta = self.Momentum * mhat / (torch.sqrt(vhat) + epsilon_)

            # Apply update: W = (1 - eta) * W + eta * W_cur
            # Ensure eta is compatible type/device if m/v were tensors
            if isinstance(eta, torch.Tensor):
                # Clamp eta to prevent excessively large updates if mhat is large and vhat is tiny
                eta = torch.clamp(eta, -1.0, 1.0).item() # Example clamping, adjust as needed

            self.W_list[i].data = (1. - eta) * self.W_list[i].data + eta * W_cur_i


    def update_Lambda(self, lambda_: Tensor, n_keep: int = 21) -> None:
        """ Updates the set of recent lambda values. """
        # Ensure lambda_ is detached and on the correct device
        lambda_val = lambda_.detach().float().cpu() # Store lambdas on CPU
        current_lambdas = self.lambda_set.cpu()

        # Keep the last n_keep values
        if len(current_lambdas) >= n_keep:
            updated_lambdas = torch.cat((current_lambdas[-(n_keep-1):], lambda_val.unsqueeze(0)), dim=0)
        else:
            updated_lambdas = torch.cat((current_lambdas, lambda_val.unsqueeze(0)), dim=0)

        # Move back to buffer's device if necessary (though CPU storage is fine)
        self.lambda_set = updated_lambdas.to(self.lambda_set.device)


    def forward(self,
                modalities: List[Tensor],
                is_training: bool = False,
                n_epoch: int = 0,
                steps_per_epoch: int = 1, # Default or estimate if not provided
        ) -> List[Tensor]:
        """
        Forward pass for RegBN.

        Args:
            modalities (List[Tensor]): List of input tensors. modalities[0] is the
                                       primary modality (f), modalities[1:] are
                                       secondary modalities (g_i).
            is_training (bool): Indicates if the model is in training mode.
            n_epoch (int): Current epoch number (used for W update).
            steps_per_epoch (int): Number of steps per epoch (used for lambda history).

        Returns:
            List[Tensor]: List containing the processed primary modality and
                          the (potentially normalized) secondary modalities.
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, but got {len(modalities)}")

        # Check for NaNs in input
        if any(torch.isnan(mod).any() for mod in modalities): # Use .any() for efficiency
            if self.verbose: print("RegBN: NaN detected in input, returning original modalities.")
            # Return original modalities without processing or normalization
            # Or consider returning normalized versions if normalize_output is True?
            # For safety, returning originals.
            return modalities

        # Separate primary (f) and secondary (g) modalities
        f = modalities[0]
        g_list = modalities[1:]
        f_sz = f.size()
        g_sz_list = [g.size() for g in g_list]

        # --- 1. Normalize and Flatten Inputs ---
        # Ensure normalization happens on the correct device
        f_flat_norm = self.norm_in_list[0](f).view(f_sz[0], -1)
        g_flat_norm_list = [
            self.norm_in_list[i+1](g).view(g_sz[0], -1)
            for i, (g, g_sz) in enumerate(zip(g_list, g_sz_list))
        ]

        # --- 2. Estimate/Apply Projections and Combine ---
        f_mapped_combined = torch.zeros_like(f_flat_norm)
        self.is_nan_ = False # Reset NaN flag for this forward pass

        for i in range(self.num_secondary_modalities):
            g_flat_norm_i = g_flat_norm_list[i]
            W_hat_i = None # Initialize W_hat_i for this modality

            if is_training:
                # --- 2a. Training: Estimate and Update W_i ---
                # SVD decomposition of the current secondary modality
                g_u_i, g_s_diag_i, g_v_i = _svd_decomposition(g_flat_norm_i,
                                                              self.sigma_THR,
                                                              self.sigma_MIN)
                if g_u_i is None or g_s_diag_i is None or g_v_i is None: # Check all SVD outputs
                    if self.verbose: print(f"RegBN: SVD failed for modality {i+1}. Skipping projection update.")
                    self.is_nan_ = True # Flag potential issue
                    W_hat_i = self.W_list[i].detach() # Use existing W without update
                else:
                    # Estimate W_plus_i (intermediate matrix) using LBFGS
                    # Pass lambda_set buffer (ensure it's on the right device if needed by W_calc)
                    lambda_set_device = self.lambda_set.to(g_flat_norm_i.device) if len(self.lambda_set) > 0 else self.lambda_set

                    # **** FIX: Pass f_flat_norm to compute ****
                    W_plus_i, lambda_i, not_found = self.W_calc.compute(
                        f_flat_norm, # Pass primary modality features
                        g_flat_norm_i, # Pass secondary modality features (for SVD components)
                        g_u_i,
                        g_s_diag_i,
                        g_v_i,
                        lambda_set_device
                    )

                    if not not_found and W_plus_i is not None and W_plus_i.numel() > 0:
                         # W_hat_i projects f onto g_i's space via W_plus_i
                         # W_hat = W_plus @ f -> shape (g_dim, batch) @ (batch, f_dim) = (g_dim, f_dim)
                        W_hat_i = torch.mm(W_plus_i, f_flat_norm)

                        # Update the persistent W_list[i]
                        self.update_W_i(i, W_hat_i.detach(), n_epoch) # Detach W_hat_i before update

                        # Update the lambda set history
                        if lambda_i is not None: # Ensure lambda was found
                            self.update_Lambda(lambda_i.detach(), n_keep=steps_per_epoch or 1)

                    else:
                        # If estimation failed, use the existing W_list[i]
                        W_hat_i = self.W_list[i].detach()
                        if self.verbose:
                            print(f'RegBN: Lambda/W_plus not found or invalid for modality {i+1}! Using previous W.')
                        # Still need to update lambda history even if W_plus failed?
                        # Maybe update with a default value if lambda_i exists?
                        if lambda_i is not None:
                           self.update_Lambda(lambda_i.detach(), n_keep=steps_per_epoch or 1)


            else:
                # --- 2b. Inference: Use Stored W_i ---
                W_hat_i = self.W_list[i].detach()


            # --- 2c. Calculate Projection and Accumulate ---
            if W_hat_i is not None:
                 # Project g_i onto f's space using W_hat_i
                 # f_mapped2g_i = g_i @ W_hat_i -> (batch, g_dim_i) @ (g_dim_i, f_dim) = (batch, f_dim)
                f_mapped2g_i = torch.mm(g_flat_norm_i, W_hat_i)
                f_mapped_combined = f_mapped_combined + f_mapped2g_i # Accumulate projections
            elif self.verbose:
                 print(f"RegBN: W_hat_i for modality {i+1} was None. Skipping projection.")


        # If any SVD failed during training, might return originals for safety
        if self.is_nan_ and is_training:
             if self.verbose: print("RegBN: SVD failed during training. Returning original modalities.")
             return modalities # Or apply output norm? Returning originals for now.


        # --- 3. Subtract Combined Projection ---
        # f_r = f_norm - sum(projections of g_i onto f)
        f_r_flat = f_flat_norm - f_mapped_combined
        f_r = f_r_flat.view(f_sz) # Reshape back to original primary modality shape

        # --- 4. Normalize Outputs ---
        f_r_norm = self.norm_out_list[0](f_r)
        # Normalize original g's, not the flattened/normalized ones
        g_norm_list = [self.norm_out_list[i+1](g) for i, g in enumerate(g_list)]

        return [f_r_norm] + g_norm_list

    def extra_repr(self) -> str:
        return (f'modalities_channels={self.modalities_channels}, '
                f'num_secondary_modalities={self.num_secondary_modalities}')


class proj_matrix_estimator(object):
    """ Computes the projection matrix component W_plus based on eq. (4) logic. """

    def __init__(self,
                 lbfgs_kwargs: dict,
                 figure: bool = False, # Unused?
                 pred_tolerance: float = 0.05,
                 verbose: bool = False, # Added verbose flag
                 ) -> None:

        self.figure = figure
        self.pred_tolerance = pred_tolerance
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose # Store verbose flag


    def get_usx(self,
                f_flatten: Tensor, # Primary modality features (batch, f_dim)
                lambda_: Tensor,   # Regularization parameter (scalar tensor)
                u: Tensor,         # Left singular vectors of g (batch, rank)
                s_diag: Tensor     # Singular values of g (rank,)
                ) -> Tensor:
        """
        Helper function for lambda_fn objective calculation.
        Computes (S^2 + lambda*I)^-1 @ S @ U.T @ F
        """
        # Ensure lambda_ is on the same device as s_diag and u
        lambda_dev = lambda_.to(s_diag.device)

        sl = torch.pow(s_diag, 2) + lambda_dev
        # Clamp values to avoid NaN/Inf in inverse
        sl = torch.clamp(sl, min=epsilon_)

        try:
            # Efficiently compute diag(1 / sl)
            sl_inv_diag = 1.0 / sl
            # sl_inv = torch.diag(sl_inv_diag) # Avoid forming full diagonal matrix if possible
        except RuntimeError as e:
             # This should not happen with the clamp, but handle just in case
            if self.verbose: print(f"RegBN (get_usx): Error inverting sl: {e}. Returning zeros.")
            rank = len(s_diag)
            f_dim = f_flatten.shape[1]
            return torch.zeros(rank, f_dim, device=f_flatten.device, dtype=f_flatten.dtype)

        # Compute diag(s_diag) @ U.T @ f_flatten efficiently
        # (diag(s) @ U.T) has shape (rank, batch)
        # (diag(s) @ U.T) @ f_flatten has shape (rank, f_dim)
        s_uT = torch.diag_embed(s_diag) @ u.t() # Shape (rank, rank) @ (rank, batch) -> (rank, batch)
        s_uT_f = torch.mm(s_uT, f_flatten)      # Shape (rank, batch) @ (batch, f_dim) -> (rank, f_dim)

        # Compute sl_inv @ (s_uT_f) efficiently using broadcasting
        # sl_inv_diag is (rank,), s_uT_f is (rank, f_dim)
        usx = sl_inv_diag.unsqueeze(1) * s_uT_f # Element-wise multiplication along rank dim

        # Shape check:
        # sl_inv_diag.unsqueeze(1): (rank, 1)
        # s_uT_f: (rank, f_dim)
        # Result usx: (rank, f_dim) - This matches the original code's output shape

        return usx


    def lambda_fn(self,
                  f_flatten: Tensor, # Primary modality features (batch, f_dim)
                  lambda_: Tensor,   # Regularization parameter
                  u: Tensor,         # Left singular vectors of g (batch, rank)
                  s_diag: Tensor     # Singular values of g (rank,)
                  ) -> torch.Tensor:
        """
        Objective function to find the optimal lambda.
        Calculates L1Loss(|| (S^2+lambda*I)^-1 @ S @ U.T @ F ||_F^2, 1.0)
        """
        usx = self.get_usx(f_flatten, lambda_, u, s_diag) # Shape (rank, f_dim)

        # Calculate squared Frobenius norm: trace(usx @ usx.H)
        # For real matrices: trace(usx @ usx.T)
        # Efficient calculation: sum(usx * usx.conj()) or sum(usx**2) for real
        objective = torch.sum(usx * usx.conj()).real # Sum over all elements, ensure real output
        # objective = torch.sum(usx**2) # If only real needed

        # L1 distance to target value (1.0)
        target = torch.tensor(1.0, device=objective.device, dtype=objective.dtype)
        loss = L1torch(objective, target)
        return loss


    def get_W_plus(self,
                   lambda_: Tensor,
                   u: Tensor,      # Left singular vectors of g (batch, rank)
                   s_diag: Tensor, # Singular values of g (rank,)
                   v: Tensor       # Right singular vectors (transposed) of g (g_dim, rank)
                   ) -> Tensor:
        """
        Calculates the intermediate projection matrix W_plus.
        W_plus = V @ diag(s / (s^2+lambda)) @ U.H
        Shape: (g_dim, batch)
        """
        # Ensure lambda_ is on the correct device
        lambda_dev = lambda_.to(s_diag.device)

        sl_diag = torch.pow(s_diag, 2.) + lambda_dev
        # Clamp values before division
        sl_diag = torch.clamp(sl_diag, min=epsilon_)

        try:
            # Calculate diag(s / sl_diag) = diag(s / (s^2 + lambda))
            s_reg_diag = s_diag / sl_diag
            # s_reg = torch.diag(s_reg_diag) # Avoid forming full matrix
        except RuntimeError as e:
            if self.verbose: print(f"RegBN (get_W_plus): Error calculating s_reg_diag: {e}. Returning zeros.")
            g_dim = v.shape[0]
            batch_size = u.shape[0]
            return torch.zeros(g_dim, batch_size, device=u.device, dtype=u.dtype)

        # Compute V @ diag(s_reg_diag) efficiently
        # V is (g_dim, rank), s_reg_diag is (rank,)
        # Result V_s_reg is (g_dim, rank)
        V_s_reg = v * s_reg_diag # Broadcasting element-wise along rank dim

        # Compute (V_s_reg) @ U.H
        # V_s_reg is (g_dim, rank), U.H is (rank, batch)
        # Result W_plus is (g_dim, batch)
        W_plus = torch.mm(V_s_reg, u.conj().t()) # V @ S_reg @ U.H

        return W_plus


    def compute(self,
                f_flatten: Tensor,  # Primary modality (batch, f_dim)
                g_flatten: Tensor,  # Secondary modality (batch, g_dim_i)
                u: Tensor,          # SVD of g_flatten (batch, rank)
                s_diag: Tensor,     # SVD of g_flatten (rank,)
                v: Tensor,          # SVD of g_flatten (g_dim_i, rank)
                lambda_set: Tensor, # Recent lambda values (on correct device)
        ) -> Tuple[Union[Tensor, None], Union[Tensor, None], bool]:
        """
        Computes the optimal lambda and the resulting W_plus matrix.

        Args:
            f_flatten (Tensor): Flattened primary modality (batch, f_dim).
            g_flatten (Tensor): Flattened secondary modality (batch, g_dim_i).
            u (Tensor): Left singular vectors of g_flatten (batch, rank).
            s_diag (Tensor): Singular values of g_flatten (rank,).
            v (Tensor): Right singular vectors (transposed) of g_flatten (g_dim_i, rank).
            lambda_set (Tensor): History of recent lambda values.

        Returns:
            Tuple[Union[Tensor, None], Union[Tensor, None], bool]:
                - W_plus (Tensor or None): The computed intermediate matrix (g_dim_i, batch).
                - lambda_hat (Tensor or None): The estimated optimal lambda.
                - closed_form_sol_not_found (bool): True if optimization failed.
        """

        # --- LBFGS Closure ---
        # Capture necessary variables for the objective function, detaching them
        # from the main computation graph as this is a self-contained optimization problem.
        f_flatten_detached = f_flatten.detach()
        u_mat = u.detach()
        s_diag_mat = s_diag.detach()

        # --- Initialize Lambda Search ---
        if lambda_set is not None and len(lambda_set) > 3:
             # Use median, ensuring it's positive and on the correct device
             median_lambda = max(epsilon_, torch.median(lambda_set.to(f_flatten.device)).item())
             lambda_init = [max(epsilon_, coef * median_lambda) for coef in [0.01, 0.1, 1., 10., 100.]]
        else:
            lambda_init = [1e-2, 1e-1, 1.0, 1e1, 1e2] # Default initial positive guesses

        lr_init = [0.1, 1.0] # Learning rates to try for LBFGS

        best_lambda = None
        min_loss = float('inf')

        # --- LBFGS Optimization Loop ---
        for lr_i in lr_init:
            for lambd_ini_val in lambda_init:
                # Initialize lambda_ for this attempt on the correct device
                lambda_ = torch.tensor([lambd_ini_val], requires_grad=True, device=f_flatten.device)

                # Define closure here to capture the current lambda_ and other inputs
                def lbfgs_closure():
                    # Zero grad the optimizer target (lambda_)
                    if lambda_.grad is not None:
                         lambda_.grad.zero_()
                    # Calculate loss using f_flatten
                    objective = self.lambda_fn(f_flatten_detached, lambda_, u_mat, s_diag_mat)
                    # Backpropagate
                    if objective.requires_grad: # Avoid backward on non-leaf tensors if error occurs
                        objective.backward()
                    return objective

                # Create optimizer for this attempt
                lbfgs_optim = torch.optim.LBFGS(
                    params=[lambda_],
                    lr=lr_i,
                    **self.lbfgs_kwargs
                )

                try:
                    # Run optimization steps
                    lbfgs_optim.step(lbfgs_closure)

                    # Evaluate final loss for this attempt (use detached lambda)
                    final_loss = self.lambda_fn(f_flatten_detached, lambda_.detach(), u_mat, s_diag_mat).item()

                    # Store result if it's the best valid one so far
                    current_lambda_val = lambda_.item()
                    # Check for valid lambda range and improvement in loss
                    if epsilon_ <= current_lambda_val < 1e8 and final_loss < min_loss:
                         min_loss = final_loss
                         best_lambda = lambda_.detach() # Store the detached tensor

                except Exception as e:
                     # Catch potential errors during optimization
                     if self.verbose: print(f"RegBN (compute): LBFGS optimization failed for init {lambd_ini_val:.2e}, lr {lr_i}. Error: {e}")
                     continue # Try next initialization


        # --- Final Check and W_plus Calculation ---
        closed_form_sol_not_found = True
        W_plus = None
        lambda_hat = None

        if best_lambda is not None:
            lambda_hat = best_lambda
            W_plus = self.get_W_plus(lambda_hat, u, s_diag, v) # u, s, v are from g

            # Check if W_plus calculation succeeded and is valid
            if W_plus is not None and not torch.isnan(W_plus).any() and not torch.isinf(W_plus).any():
                 # Check if loss tolerance is met
                 if min_loss < self.pred_tolerance:
                     closed_form_sol_not_found = False # Success
                 else:
                     # Found a lambda, but loss is high. Still use it but flag it.
                     if self.verbose: print(f"RegBN (compute): Lambda found ({lambda_hat.item():.2e}), but loss ({min_loss:.4f}) > tolerance ({self.pred_tolerance:.4f}).")
                     closed_form_sol_not_found = False # Mark as found, but maybe suboptimal
            else:
                 # W_plus calculation failed (e.g., NaN/Inf)
                 if self.verbose: print(f"RegBN (compute): NaN/Inf detected in W_plus for lambda {lambda_hat.item():.2e}.")
                 W_plus = None # Reset W_plus
                 lambda_hat = None # Reset lambda_hat
                 closed_form_sol_not_found = True # Mark as not found


        # If no suitable lambda/W_plus was found at all
        if closed_form_sol_not_found:
            if self.verbose: print("RegBN (compute): Could not find suitable lambda/W_plus via LBFGS.")
            # Handle failure case: Use a default small lambda
            lambda_hat = torch.tensor([epsilon_], device=f_flatten.device) # Small positive default
            W_plus = self.get_W_plus(lambda_hat, u, s_diag, v)
            # Still return closed_form_sol_not_found = True to signal optimization failure

        # Ensure lambda_hat is always a tensor, even if optimization failed
        if lambda_hat is None:
             lambda_hat = torch.tensor([epsilon_], device=f_flatten.device)


        return W_plus, lambda_hat, closed_form_sol_not_found


# --- Helper Functions ---

def _svd_decomposition(data: Tensor, sigma_THR: float, sigma_MIN: float) -> Tuple[Union[Tensor, None], Union[Tensor, None], Union[Tensor, None]]:
    """ Calculates SVD decomposition with thresholding for singular values. """
    # Check for NaNs or Infs before SVD
    if torch.isnan(data).any() or torch.isinf(data).any():
        print("RegBN (_svd): NaN or Inf detected in input data to SVD.")
        return None, None, None
    # Check if data is empty
    if data.numel() == 0:
        print("RegBN (_svd): Input data to SVD is empty.")
        return None, None, None

    # Set full_matrices based on PyTorch version
    svd_kwargs = {'full_matrices': False} if torchV >= 13 else {}

    try:
        # Perform SVD
        u, s_diag, vh = svd_torch(data, **svd_kwargs)

        # Check for NaNs in results (can happen with ill-conditioned matrices)
        if torch.isnan(u).any() or torch.isnan(s_diag).any() or torch.isnan(vh).any():
            print("RegBN (_svd): NaN detected in SVD results.")
            raise RuntimeError("SVD resulted in NaNs")

        # Threshold singular values
        if len(s_diag) > 0: # Avoid error on empty s_diag
            # Ensure sigma_MIN is positive
            sigma_MIN_eff = max(epsilon_, sigma_MIN)
            thr = torch.max(s_diag) * sigma_THR
            # Ensure thresholding doesn't create NaNs if thr is NaN
            if not torch.isnan(thr):
                 s_diag = torch.where(s_diag > thr.item(), s_diag, torch.tensor(sigma_MIN_eff, device=s_diag.device, dtype=s_diag.dtype))
            else:
                 print("RegBN (_svd): NaN threshold detected. Using sigma_MIN for all singular values.")
                 s_diag.fill_(sigma_MIN_eff)
            # Clamp very small singular values after thresholding
            s_diag = torch.clamp(s_diag, min=epsilon_)
        else:
            # Handle case with zero singular values if necessary
            pass # s_diag is already empty

        # Get V from Vh (Hermitian transpose)
        v = vh.mH if torchV >= 13 else vh # Use mH for >=1.13, else vh is already V

        return u, s_diag, v

    except Exception as e: # Catch LinAlgError or other issues
        print(f"RegBN (_svd): SVD failed. Error: {e}")
        # Attempt SVD with added noise as a fallback - Use with caution
        # Adding noise might not always be the best solution
        # Consider returning None or raising an error depending on desired behavior
        # For now, returning None to indicate failure
        # try:
        #     print("RegBN (_svd): Retrying SVD with noise perturbation.")
        #     noise = 1e-4 * data.mean() * torch.rand_like(data)
        #     u, s_diag, vh = svd_torch(data + noise, **svd_kwargs)
        #     # ... (rest of retry logic) ...
        # except Exception as e_retry:
        #     print(f"RegBN (_svd): SVD retry also failed. Error: {e_retry}. Returning None.")
        return None, None, None


def _get_norm_inp(normalize_input: bool,
                  num_channels: int,
                  layer_dim: List,
                  affine: bool) -> nn.Module:
    """ Returns input normalization layer based on dimensions. """
    if not normalize_input:
        return nn.Identity()

    # Ensure num_channels is valid
    if num_channels <= 0:
        raise ValueError(f"Number of channels must be positive, got {num_channels}")

    if not layer_dim: # No spatial dimensions (e.g., [batch, channels])
        # Use LayerNorm over the channel dimension
        norm_ = nn.LayerNorm(num_channels, elementwise_affine=affine)
    elif len(layer_dim) == 1: # 1D spatial (e.g., [batch, channels, length])
        norm_ = nn.BatchNorm1d(num_channels, affine=affine)
    elif len(layer_dim) == 2: # 2D spatial (e.g., [batch, channels, height, width])
        norm_ = nn.BatchNorm2d(num_channels, affine=affine)
    elif len(layer_dim) == 3: # 3D spatial (e.g., [batch, channels, depth, height, width])
        norm_ = nn.BatchNorm3d(num_channels, affine=affine)
    else:
        # For >3D, LayerNorm is generally safer and applicable
        normalized_shape = [num_channels] + layer_dim
        print(f"RegBN (_get_norm_inp): Using LayerNorm for high ({len(layer_dim)}) spatial dimensions.")
        norm_ = nn.LayerNorm(normalized_shape, elementwise_affine=affine)

    # Initialize affine parameters if they exist
    if affine and hasattr(norm_, 'weight') and norm_.weight is not None:
        nn.init.constant_(norm_.weight, 1.0)
    if affine and hasattr(norm_, 'bias') and norm_.bias is not None:
        nn.init.constant_(norm_.bias, 0.0)

    return norm_


def _get_norm_out(normalize_output: bool,
                  num_channels: int,
                  layer_dim: List,
                  affine: bool) -> nn.Module:
    """ Returns output normalization layer (typically LayerNorm). """
    if not normalize_output:
        return nn.Identity()

    # Ensure num_channels is valid
    if num_channels <= 0:
        raise ValueError(f"Number of channels must be positive, got {num_channels}")

    # Use LayerNorm for output normalization
    normalized_shape = [num_channels] + layer_dim
    norm_ = nn.LayerNorm(normalized_shape, elementwise_affine=affine)

    # Initialize affine parameters if they exist
    if affine and hasattr(norm_, 'weight') and norm_.weight is not None:
        nn.init.constant_(norm_.weight, 1.0)
    if affine and hasattr(norm_, 'bias') and norm_.bias is not None:
        nn.init.constant_(norm_.bias, 0.0)

    return norm_

