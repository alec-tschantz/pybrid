import logging
import pprint

import torch

from pybrid import utils
from pybrid import datasets
from pybrid import optim
from pybrid.models.hybrid import HybridModel


def main(cfg):
    utils.setup_logging()
    utils.seed(cfg.exp.seed)
    pprint.pprint(cfg)

    train_dataset = datasets.MNIST(
        train=True,
        scale=cfg.data.label_scale,
        size=cfg.data.train_size,
        normalize=cfg.data.normalize,
    )
    test_dataset = datasets.MNIST(
        train=False,
        scale=cfg.data.label_scale,
        size=cfg.data.test_size,
        normalize=cfg.data.normalize,
    )
    train_loader = datasets.get_dataloader(train_dataset, cfg.optim.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cfg.optim.batch_size)
    msg = f"Loaded MNIST dataset (train {len(train_loader)} test {len(test_loader)})"
    logging.info(msg)

    model = HybridModel(
        nodes=cfg.model.nodes,
        amort_nodes=cfg.model.amort_nodes,
        mu_dt=cfg.infer.mu_dt,
        act_fn=utils.get_act_fn(cfg.model.act_fn),
        use_bias=cfg.model.use_bias,
        kaiming_init=cfg.model.kaiming_init,
    )
    optimizer = optim.get_optim(
        model.params,
        cfg.optim.name,
        cfg.optim.lr,
        amort_lr=cfg.optim.amort_lr,
        batch_scale=cfg.optim.batch_scale,
        grad_clip=cfg.optim.grad_clip,
        weight_decay=cfg.optim.weight_decay,
    )
    logging.info(f"Loaded model {model}")

    with torch.no_grad():
        # metrics = {"acc": [], "pc_acc": [], "amortised_acc": []}
        for epoch in range(1, cfg.exp.num_epochs + 1):

            logging.info(f"Train @ epoch {epoch} ({len(train_loader)} batches)")
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                model.train_batch(
                    img_batch,
                    label_batch,
                    cfg.infer.num_train_iters,
                    fixed_preds=cfg.infer.fixed_preds_train,
                )
                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

                if batch_id % 100 == 0:
                    logging.info(f"batch [{batch_id}/{len(train_loader)}] loss {model.get_loss()}")

            if epoch % cfg.exp.test_every == 0:
                logging.info(f"=== Test @ epoch {epoch} === ")
                acc, pc_acc, q_acc = 0, 0, 0
                for _, (img_batch, label_batch) in enumerate(test_loader):
                    label_preds = model.test_batch(
                        img_batch, cfg.infer.num_test_iters, fixed_preds=cfg.infer.fixed_preds_test
                    )
                    acc = acc + datasets.accuracy(label_preds, label_batch)

                    label_preds = model.test_batch(
                        img_batch,
                        cfg.infer.num_test_iters,
                        init_std=cfg.infer.init_std,
                        fixed_preds=cfg.infer.fixed_preds_test,
                        use_amort=False,
                    )
                    pc_acc = pc_acc + datasets.accuracy(label_preds, label_batch)

                    label_preds = model.forward(img_batch)
                    q_acc = datasets.accuracy(label_preds, label_batch)

                acc = acc / len(test_loader)
                pc_acc = pc_acc / len(test_loader)
                q_acc = q_acc / len(test_loader)
                # metrics["acc"].append(acc)
                # metrics["pc_acc"].append(pc_acc)
                # metrics["q_acc"].append(q_acc)
                msg = "accuracy: {:.4f} pc accuracy {:.4f} amortised accuracy {:.4f} "
                logging.info(msg.format(acc, pc_acc, q_acc))

                _, label_batch = next(iter(test_loader))
                img_preds = model.backward(label_batch)
                datasets.plot_imgs(img_preds, cfg.exp.img_path + f"/{epoch}.png")

            # utils.save_json(metrics, cfg.logdir + "metrics.json")


if __name__ == "__main__":
    cfg = utils.load_config("./config.json")
    main(cfg)
