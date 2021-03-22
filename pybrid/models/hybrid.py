import logging

import torch

from pybrid import utils
from pybrid.models.base import BaseModel
from pybrid.layers import FCLayer


class HybridModel(BaseModel):
    def __init__(
        self,
        nodes,
        amort_nodes,
        act_fn,
        train_thresh=None,
        test_thresh=None,
        mu_dt=0.01,
        use_bias=False,
        kaiming_init=False,
    ):
        self.nodes = nodes
        self.amort_nodes = amort_nodes
        self.mu_dt = mu_dt
        self.train_thresh = train_thresh
        self.test_thresh = test_thresh

        self.num_nodes = len(nodes)
        self.num_layers = len(nodes) - 1
        self.total_params = 0

        self.layers = []
        self.amort_layers = []
        for l in range(self.num_layers):
            layer_act_fn = utils.Linear() if (l == self.num_layers - 1) else act_fn
            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=layer_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            self.layers.append(layer)
            self.total_params = self.total_params + (nodes[l] * nodes[l + 1]) + nodes[l + 1]

            # TODO use same layer_act_fn ?
            amort_layer = FCLayer(
                in_size=amort_nodes[l],
                out_size=amort_nodes[l + 1],
                act_fn=layer_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
                is_amortised=True,
            )
            self.amort_layers.append(amort_layer)

        self.mean_weights, self.mean_biases = self.get_weight_stats()

    def reset(self):
        self.preds = [[] for _ in range(self.num_nodes)]
        self.errs = [[] for _ in range(self.num_nodes)]
        self.q_preds = [[] for _ in range(self.num_nodes)]
        self.q_errs = [[] for _ in range(self.num_nodes)]
        self.mus = [[] for _ in range(self.num_nodes)]

    def reset_mu(self, batch_size, init_std=0.05):
        for l in range(self.num_layers):
            tensor = torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            self.mus[l] = utils.set_tensor(tensor)

    def set_img_batch(self, img_batch):
        self.mus[-1] = img_batch.clone()

    def set_img_batch_amort(self, img_batch):
        self.q_preds[0] = img_batch.clone()

    def set_label_batch(self, label_batch):
        self.mus[0] = label_batch.clone()

    def forward(self, val):
        for layer in self.amort_layers:
            val = layer.forward(val)
        return val

    def backward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def forward_mu(self):
        for l in range(1, self.num_nodes):
            self.q_preds[l] = self.amort_layers[l - 1].forward(self.q_preds[l - 1])

        mus = self.q_preds[::-1]
        for l in range(self.num_nodes):
            self.mus[l] = mus[l].clone()

    def backward_mu(self):
        for l in range(1, self.num_layers):
            self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_batch(self, img_batch, label_batch, num_iters=20, fixed_preds=False, use_amort=True):
        self.reset()
        if use_amort:
            self.set_img_batch_amort(img_batch)
            self.forward_mu()
            self.set_img_batch(img_batch)
        else:
            self.set_label_batch(label_batch)
            self.backward_mu()
            self.set_img_batch(img_batch)

        self.set_label_batch(label_batch)
        num_iter = self.train_updates(num_iters, fixed_preds=fixed_preds)
        self.update_grads()
        if use_amort:
            self.update_amort_grads()
        return num_iter

    def test_batch(
        self, img_batch, num_iters=100, init_std=0.05, fixed_preds=False, use_amort=True
    ):
        self.reset()
        if use_amort:
            self.set_img_batch_amort(img_batch)
            self.forward_mu()
        else:
            self.reset_mu(img_batch.size(0), init_std)

        self.set_img_batch(img_batch)
        num_iter = self.test_updates(num_iters, fixed_preds=fixed_preds)
        return self.mus[0], num_iter

    def train_updates(self, num_iters, fixed_preds=False):
        for n in range(1, self.num_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        itr = 0
        for itr in range(num_iters):
            for l in range(1, self.num_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * (2 * delta)

            for n in range(1, self.num_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            avg_loss = self.get_loss()[0] / self.total_params
            if self.train_thresh is not None and avg_loss < self.train_thresh:
                break

        return itr

    def test_updates(self, num_iters, fixed_preds):
        for n in range(1, self.num_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        itr = 0
        for itr in range(num_iters):
            delta = self.layers[0].backward(self.errs[1])
            self.mus[0] = self.mus[0] + self.mu_dt * (2 * delta)
            for l in range(1, self.num_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * (2 * delta)

            for n in range(1, self.num_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            if self.test_thresh is not None and self.get_loss()[0] < self.test_thresh:
                break

        return itr

    def update_grads(self):
        for l in range(self.num_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def update_amort_grads(self):
        for q, l in zip(reversed(range(1, self.num_nodes)), range(self.num_layers)):
            self.q_errs[q] = self.mus[l] - self.q_preds[q]

        for l in range(self.num_layers):
            self.amort_layers[l].update_gradient(self.q_errs[l + 1])

    def get_loss(self):
        try:
            return torch.sum(self.errs[-1] ** 2).item(), torch.sum(self.q_errs[-1] ** 2).item()
        except:
            return torch.sum(self.errs[-1] ** 2).item(), 0

    def get_weight_stats(self):
        mean_abs_weights = []
        mean_abs_biases = []
        for l in range(self.num_layers):
            mean_abs_weights.append(torch.mean(torch.abs(self.layers[l].weights)).item())
            mean_abs_biases.append(torch.mean(torch.abs(self.layers[l].bias)).item())
        return mean_abs_weights, mean_abs_biases

    def normalize_weights(self):
        for l in range(self.num_layers):
            mean_weights = torch.mean(torch.abs(self.layers[l].weights))
            self.layers[l].weights = self.layers[l].weights * self.mean_weights[l] / mean_weights
            mean_bias = torch.mean(torch.abs(self.layers[l].bias))
            self.layers[l].bias = self.layers[l].bias * self.mean_biases[l] / mean_bias

    @property
    def params(self):
        return self.layers + self.amort_layers

    def __str__(self):
        return f"<HybridModel> {self.nodes}"
