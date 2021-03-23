from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reset_mu(self, batch_size):
        pass

    @abstractmethod
    def set_img_batch(self, img_batch):
        pass

    @abstractmethod
    def set_label_batch(self, label_batch):
        pass

    @abstractmethod
    def forward(self, val):
        pass

    @abstractmethod
    def backward(self, val):
        pass

    @abstractmethod
    def train_batch(self, img_batch, label_batch):
        pass

    @abstractmethod
    def test_batch(self, img_batch):
        pass

    @abstractmethod
    def update_grads(self):
        pass

    @property
    def params(self):
        pass
