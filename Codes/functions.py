import torch
import torch.nn.functional as F

def AoA(TD1, TD2):

    # batchsize * wordnum * hiddensize
    td1 = TD1.permute([1, 0, 2])
    td2 = TD2.permute([1, 0, 2])

    norm1 = torch.norm(td1, dim=2, keepdim=True)
    norm2 = torch.norm(td2, dim=2, keepdim=True)

    norm_td1 = td1 / norm1
    norm_td2 = td2 / norm2

    # batchsize * wordnum1 * wordnum2
    sim = torch.matmul(norm_td1, norm_td2.permute([0, 2, 1]))
    sim = torch.cos(sim)

    # batchsize * wordnum1 * wordnum2
    alpha = F.softmax(sim, dim=1)

    # batchsize * wordnum1 * 1
    sim_1 = torch.max(sim, dim=2, keepdim=True)[0]
    beta = F.softmax(sim_1, dim=1)

    # batchsize * wordnum2 * 1
    gamma = torch.matmul(alpha.permute([0, 2, 1]), beta)

    # batchsize * hiddensize * 1
    td = torch.matmul(td2.permute([0, 2, 1]), gamma)

    return td




