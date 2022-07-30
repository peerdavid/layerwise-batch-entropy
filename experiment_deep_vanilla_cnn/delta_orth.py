import math
import torch


""" The implementation below corresponds to Tensorflow implementation.
    Refer https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py for details.
    From https://github.com/yl-1993/ConvDeltaOrthogonal-Init/

    We tried this version as well as the version implemented in train.py, but failed.
"""


def init_delta_orthogonal_1(tensor, gain=1.):
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
      raise ValueError("The tensor to initialize must be at least "
                       "three-dimensional and at most five-dimensional")

    if tensor.size(1) > tensor.size(0):
      raise ValueError("In_channels cannot be greater than out_channels.")

    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2)-1)//2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2] = q
        else:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2, (tensor.size(4)-1)//2] = q
        tensor.mul_(math.sqrt(gain))
    return tensor


#
# Test also a second method with orthogonal initialization
#
def _gen_orthogonal(dim):
    """ Thanks to https://github.com/JiJingYu/delta_orthogonal_init_pytorch/
    """
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def init_delta_orthogonal_2(weights, gain):
    """ Thanks to https://github.com/JiJingYu/delta_orthogonal_init_pytorch/
    """
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = _gen_orthogonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
        weights.mul_(gain)
