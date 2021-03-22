import logging
import pprint

import torch

from pybrid import utils
from pybrid import datasets
from pybrid import optim
from pybrid.models.hybrid import HybridModel


def main(cfg):
    utils.setup_logging()
    cfg.exp.log_dir = utils.setup_logdir(cfg.exp.log_dir, cfg.exp.seed)
    utils.seed(cfg.exp.seed)
    pprint.pprint(cfg)
    logging.info(f"using {utils.get_device()}")
    utils.save_json(cfg, cfg.exp.log_dir + "/config.json")

    datasets.download_mnist()
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
    msg = f"Loaded MNIST ({len(train_loader)} train batches {len(test_loader)} test batches)"
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
        metrics = {"hybrid_acc": [], "pc_acc": [], "amort_acc": []}
        for epoch in range(1, cfg.exp.num_epochs + 1):
            pc_losses, amort_losses = [], []
            logging.info(f"=== Train @ epoch {epoch} ({len(train_loader)} batches) ===")
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                model.train_batch(
                    img_batch,
                    label_batch,
                    cfg.infer.num_train_iters,
                    fixed_preds=cfg.infer.fixed_preds_train,
                    use_amort=cfg.model.train_amortised,
                )
                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

                pc_loss, amort_loss = model.get_loss()
                pc_losses.append(pc_loss)
                amort_losses.append(amort_loss)

                if batch_id % 100 == 0:
                    pc_loss = sum(pc_losses) / (batch_id + 1)
                    amort_loss = sum(amort_losses) / (batch_id + 1)
                    msg = f"batch [{batch_id}/{len(train_loader)}]: pc [{pc_loss:.4f}] amortised [{amort_loss:.4f}]"
                    logging.info(msg)

            if epoch % cfg.exp.test_every == 0:
                logging.info(f"=== Test @ epoch {epoch} ({len(test_loader)} batches) === ")
                hybrid_acc, pc_acc, amort_acc = 0, 0, 0
                for _, (img_batch, label_batch) in enumerate(test_loader):

                    # Test hybrid inference
                    label_preds = model.test_batch(
                        img_batch, cfg.infer.num_test_iters, fixed_preds=cfg.infer.fixed_preds_test
                    )
                    hybrid_acc = hybrid_acc + datasets.accuracy(label_preds, label_batch)

                    # Test predictive coing
                    label_preds = model.test_batch(
                        img_batch,
                        cfg.infer.num_test_iters,
                        init_std=cfg.infer.init_std,
                        fixed_preds=cfg.infer.fixed_preds_test,
                        use_amort=False,
                    )
                    pc_acc = pc_acc + datasets.accuracy(label_preds, label_batch)

                    # Test amortised inference
                    label_preds = model.forward(img_batch)
                    amort_acc = amort_acc + datasets.accuracy(label_preds, label_batch)

                hybrid_acc = hybrid_acc / len(test_loader)
                pc_acc = pc_acc / len(test_loader)
                amort_acc = amort_acc / len(test_loader)
                metrics["hybrid_acc"].append(hybrid_acc)
                metrics["pc_acc"].append(pc_acc)
                metrics["amort_acc"].append(amort_acc)
                msg = "hybrid accuracy: {:.4f} pc accuracy {:.4f} amortised accuracy {:.4f} "
                logging.info(msg.format(hybrid_acc, pc_acc, amort_acc))

                _, label_batch = next(iter(test_loader))
                img_preds = model.backward(label_batch)
                datasets.plot_imgs(img_preds, cfg.exp.log_dir + f"/{epoch}.png")

            utils.save_json(metrics, cfg.exp.log_dir + "/metrics.json")


if __name__ == "__main__":
    cfg = utils.load_config("./config.json")
    main(cfg)
