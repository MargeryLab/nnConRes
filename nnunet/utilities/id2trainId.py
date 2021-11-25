import torch


def id2trainId(label):
    shape = label.shape
    results_map = torch.zeros((shape[0], 2, shape[2], shape[3], shape[4]), device='cuda')

    TUMOR = (label[:, 0, :, :, :] == 1)
    RECTAL = (label[:, 0, :, :, :] == 2)

    results_map[:, 0, :, :, :] = torch.where(TUMOR, 1, 0)
    results_map[:, 1, :, :, :] = torch.where(RECTAL, 1, 0)
    return results_map