#
# nearest_embedding.py, doom-net
#
# Inspired by https://arxiv.org/abs/1711.00937
#
# Created by Andrey Kolishchak on 12/2/17.
#
import torch
from torch.autograd import Variable
from torch.autograd.function import Function
from torch.nn.parameter import Parameter
import torch.nn as nn


class NearestEmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        assert input.dim() == 2
        assert weight.dim() == 2
        ctx._weight_size = weight.size()
        # repeat each row by number of embeddings
        input = input.view(input.size(0), 1, -1)
        input = input.expand(input.size(0), weight.size(0), -1)
        # compute distance between all embeddings and input
        distance = torch.pow(input - weight.expand_as(input), 2).sum(2)
        # select embeddings with minimal distance
        _, indices = distance.min(1)
        ctx._indices = indices
        output = torch.index_select(weight, 0, indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        tensor_type = type(grad_output).__name__
        if grad_output.is_cuda:
            SparseTensor = getattr(torch.cuda.sparse, tensor_type)
        else:
            SparseTensor = getattr(torch.sparse, tensor_type)

        grad_input = grad_output
        indices = ctx._indices
        indices = indices.view(1, -1)
        grad_weight = SparseTensor(indices, grad_output, ctx._weight_size).to_dense()
        return grad_input, grad_weight


class NearestEmbedding(nn.Module):
    def __init__(self, embedding_num, embedding_dim):
        super(NearestEmbedding, self).__init__()
        self.weight = Parameter(torch.Tensor(embedding_num, embedding_dim))
        self.reset_parameters()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_num = embedding_num

    def reset_parameters(self):
        self.weight.normal_(0, 1)

    def forward(self, input):
        input = self.bn(input)
        return NearestEmbeddingFunction.apply(input, self.weight)[0]

    def forward_onehot(self, input):
        input = self.bn(input)
        indices = NearestEmbeddingFunction.apply(input, self.weight)[1]
        onehot = input.new(input.size(0), self.embedding_num)
        onehot.zero_()
        onehot.scatter_(1, indices.view(-1, 1), 1)
        return onehot

