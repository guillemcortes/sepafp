from torch.nn.modules.loss import _Loss
import torch
import numpy as np

class LogL2Time(_Loss):
    
    def forward(self, est_targets, targets):
        # if targets.size() != est_targets.size() or targets.ndim < 2:
        #     raise TypeError(
        #         f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead"
        #     )

        _, number_of_sources, length_of_sources = est_targets.shape
        min_time = np.minimum(length_of_sources,targets.shape[-1])
        est_targets = est_targets[...,:min_time]
        targets = targets[...,:min_time]
        squared_abs_dif = torch.abs((est_targets - targets) ** 2)
        sum_of_squared_abs_dif = torch.sum(squared_abs_dif, dim=2)
        sum_of_log = torch.sum(torch.log10(sum_of_squared_abs_dif), dim=1)
        loss = 10 / (number_of_sources * length_of_sources) * sum_of_log
        loss = loss.mean(dim=0)
        return loss

class LogL2Time_weighted(_Loss):
    def __init__(self, weights):
        super(LogL2Time_weighted, self).__init__()
        self.weights=torch.from_numpy(np.array(weights))


    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError(
                f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead"
            )
        _, number_of_sources, length_of_sources = est_targets.shape
        if (self.weights.size(0) != est_targets.size(1) or self.weights.size(0) != targets.size(1)):
            raise TypeError(
                f"Weights should have the same length as number of sources, got {weights.size()} while number of sources is {number_of_sources}"
            )

        weights = self.weights.to(device=targets.get_device())
        squared_abs_dif = torch.abs((est_targets - targets) ** 2)
        sum_of_squared_abs_dif = torch.sum(squared_abs_dif, dim=2)
        sum_of_squared_abs_dif = weights * sum_of_squared_abs_dif
        sum_of_log = torch.sum(torch.log10(sum_of_squared_abs_dif), dim=1)
        loss = 10 / (number_of_sources * length_of_sources) * sum_of_log
        loss = loss.mean(dim=0)
        return loss



class SingleSrcNegSDR_weighted(_Loss):
    """
    """

    def __init__(self, weights, sdr_type='sisdr', zero_mean=True, take_log=True, reduction="none", eps=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        #self.weights=torch.tensor(np.array(weights),requires_grad=False)
        self.eps = eps


    def forward(self, est_target, target):
        assert target.size() == est_target.size()
        # weights = self.weights.to(device=target.get_device())
        # #import pdb;pdb.set_trace()
        # target = weights * target.transpose(1,2)
        # target = target.transpose(1,2)
        # est_target = weights * est_target.transpose(1,2)
        # est_target = est_target.transpose(1,2)

        # # Step 1. Zero-mean norm
        # if self.zero_mean:
        #     mean_source = torch.mean(target, dim=1, keepdim=True)
        #     mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
        #     target = target - mean_source
        #     est_target = est_target - mean_estimate
        # # Step 2. Pair-wise SI-SDR.
        # if self.sdr_type in ["sisdr", "sdsdr"]:
        #     # [batch, 1]
        #     dot = torch.sum(est_target * target, dim=1, keepdim=True)
        #     # [batch, 1]
        #     s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + self.eps
        #     # [batch, time]
        #     scaled_target = dot * target / s_target_energy
        # else:
        #     # [batch, time]
        #     scaled_target = target
        # if self.sdr_type in ["sdsdr", "snr"]:
        #     e_noise = est_target - target
        # else:
        #     e_noise = est_target - scaled_target

        scaled_target = target
        e_noise = est_target - scaled_target

        losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + self.eps)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.eps)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses