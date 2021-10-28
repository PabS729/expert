import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from MOE import SparseDispatcher

class Net_sub(nn.Module):
    def __init__(self, linear, in_size, out_size, layer = 1):
        super(Net_sub, self).__init__()
        self.linear = linear
        if linear == True:
            self.module = nn.Linear(in_size, out_size)
        else:
            if layer == 1:
                self.module = nn.Conv2d(in_size, 16, 3, 1)
                self.bn = nn.BatchNorm2d(16)
            else:
                self.module = nn.Conv2d(2**(layer+2), 2**(layer+3), 3, 1)
                self.bn = nn.BatchNorm2d(2**(layer+3), 3, 1)

    def forward(self, x):
        if self.linear == True:
            x = self.module(x)
            x = F.relu(x)
        else:
            x = self.module(x)
            x = F.relu(x)
            x = self.bn(x)
        return x


class SparseDispatcher_mod(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        #print(self._batch_index)
        # calculate num samples that each expert gets

        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        #print("inp:", inp_exp.shape)
        dim_l = inp_exp.shape[2]
        if len(inp_exp.shape) > 3:
            inp_exp = inp_exp.reshape((inp_exp.shape[0], inp_exp.shape[1],inp_exp.shape[2], inp_exp.shape[2]))
        else:
            inp_exp = inp_exp.reshape((inp_exp.shape[0], 1, inp_exp.shape[1],inp_exp.shape[2]))
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        stitched = stitched.reshape(stitched.shape[0], stitched.shape[1] * stitched.shape[2] * stitched.shape[3] )
        #stitched = expert_out
        #print(stitched.size())
        # print(self._nonzero_gates.shape)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), stitched.size(1), requires_grad=True).cuda()
        #print(zeros.shape)
        #print(self._batch_index)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE_mod(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, model, noisy_gating=False, k=4):
        super(MoE_mod, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        #self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([model for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)




    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        #print(threshold_positions_if_in.get_device(), noisy_top_values.get_device())
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.cuda()), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out.cuda()), 1)
        # is each value currently in the top k.
        #print(clean_values.get_device(), noisy_top_values.get_device(), noise_stddev.get_device())
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in.cuda())/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out.cuda())/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]**2))
        #print(x.shape, self.w_gate.shape)
        #print(x.shape, self.w_gate.shape)
        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k] 
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load



    def forward(self, x, train=True, loss_coef=3e-1):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, train)
        #print(gates.shape)
        # calculate importance loss
        importance = gates.sum(0)
        sd = gates

        dispatcher = SparseDispatcher_mod(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        #print(expert_inputs[2].shape)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        _,c,h,w = expert_outputs[0].shape
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(y.shape[0], c,h,w)
        #print(y.shape)

        return y, sd
