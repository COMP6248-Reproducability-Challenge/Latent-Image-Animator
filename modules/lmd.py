import torch

from torch import nn


def initial_directions(components, dimensions, device):
    M = torch.randn(dimensions, components).to(device)
    Q,R = torch.linalg.qr(M)
    return Q


def QR_Decomposition(A):
    n = A.size(0)  # get the shape of A
    m = A.size(1)
    Q = torch.empty((n, n))  # initialize matrix Q
    u = torch.empty((n, n))  # initialize matrix u
    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / torch.linalg.norm(u[:, 0])
    for i in range(1, n):
        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j]  # get each u vector
        Q[:, i] = u[:, i] / torch.linalg.norm(u[:, i])  # compute each e vetor
    R = torch.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]
    return Q, R


def diag_sign(A):
    "Compute the signs of the diagonal of matrix A"
    D = torch.diag(torch.sign(torch.diag(A)))
    return D


def adjust_sign(Q, R):
    """
    Adjust the signs of the columns in Q and rows in R to
    impose positive diagonal of Q
    """
    D = diag_sign(Q)
    Q[:, :] = Q @ D
    R[:, :] = D @ R
    return Q, R


class LinearMotionDecomposition(nn.Module):
    MOTION_DICTIONARY_SIZE = 20
    MOTION_DICTIONARY_DIMENSION = 512

    def __init__(self):
        super(LinearMotionDecomposition, self).__init__()
        self.motion_dictionary = torch.nn.Parameter(
            initial_directions(self.MOTION_DICTIONARY_SIZE, self.MOTION_DICTIONARY_DIMENSION, device='cpu'),
            requires_grad=True)

    def generate_latent_path(self, magnitudes):
        magnitudes = magnitudes.unsqueeze(0)
        Z = torch.empty(1, 512)
        M = torch.transpose(self.motion_dictionary.data, 0, 1)
        for i in range(magnitudes.shape[0]):
            for j in range(M.shape[1]):
                total = 0
                for k in range(magnitudes.shape[1]):
                    total += magnitudes[i, k] * M[k, j]
                Z[i, j] = total
        return Z

    def generate_target_code(self, source_latent_code, magnitudes):
        return source_latent_code + self.generate_latent_path(magnitudes=magnitudes)

    def forward(self, x, magnitudes):
        self.motion_dictionary = adjust_sign(*QR_Decomposition(self.motion_dictionary))
        target_code = self.generate_target_code(x, magnitudes)
        return target_code
