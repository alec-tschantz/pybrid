import torch

from pybrid import utils
from pybrid.models.base import BaseModel
from pybrid.layers import FCLayer


class PCModel(BaseModel):
    def __init__(self, nodes, act_fn, mu_dt=0.01, use_bias=False, kaiming_init=False):
        self.nodes = nodes
        self.mu_dt = mu_dt

        self.num_nodes = len(nodes)
        self.num_layers = len(nodes) - 1

        self.layers = []
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

    def reset(self):
        self.preds = [[] for _ in range(self.num_nodes)]
        self.errs = [[] for _ in range(self.num_nodes)]
        self.mus = [[] for _ in range(self.num_nodes)]

    def reset_mu(self, batch_size, init_std=0.05):
        for l in range(self.num_layers):
            tensor = torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            self.mus[l] = utils.set_tensor(tensor)

    def set_img_batch(self, img_batch):
        self.mus[-1] = img_batch.clone()

    def set_label_batch(self, label_batch):
        self.mus[0] = label_batch.clone()

    def forward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def backward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def forward_mu(self):
        for l in range(1, self.num_layers):
            self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_batch(self, img_batch, label_batch, num_iters=20, fixed_preds=False):
        self.reset()
        self.set_img_batch(img_batch)
        self.forward_mu()
        self.set_label_batch(label_batch)
        self.train_updates(num_iters, fixed_preds=fixed_preds)
        self.update_grads()

    def test_batch(self, img_batch, num_iters=100, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mu(batch_size, init_std)
        self.set_img_batch(img_batch)
        self.test_updates(num_iters, fixed_preds=fixed_preds)
        return self.mus[0]

    def train_updates(self, num_iters, fixed_preds=False):
        for n in range(1, self.num_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for _ in range(num_iters):
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.num_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            # TODO check convergence

    def test_updates(self, num_iters, fixed_preds=False):
        for n in range(1, self.num_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for _ in range(num_iters):
            delta = self.layers[0].backward(self.errs[1])
            self.mus[0] = self.mus[0] + self.mu_dt * delta
            for l in range(1, self.num_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.num_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

    def update_grads(self):
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def get_loss(self):
        return torch.sum(self.errs[-1] ** 2).item()

    @property
    def params(self):
        return self.layers
