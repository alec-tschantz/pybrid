import logging
from pprint import pprint
import torch
from pybrid import utils, datasets, optim
from pybrid.models.hybrid import HybridModel


def main(cfg):
    cfg = utils.setup_experiment(cfg)

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
    logging.info(f"Loaded MNIST [train {len(train_loader)}] [test {len(test_loader)}]")

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
        metrics = utils.to_attr_dict(
            {
                "hybrid_acc": [],
                "pc_acc": [],
                "amort_acc": [],
                "pc_losses": [],
                "pc_errs": [],
                "amort_losses": [],
                "amort_errs": [],
                "num_train_iters": [],
                "num_test_iters": [],
                "num_test_iters_pc": [],
                "init_errs": [],
                "final_errs": [],
            }
        )
        for epoch in range(1, cfg.exp.num_epochs + 1):
            logging.info(f"Epoch {epoch}/{cfg.exp.num_epochs + 1}")

            logging.info(f"Train @ epoch {epoch} [{len(train_loader)} batches]")
            pc_losses, pc_errs, amort_losses, amort_errs, num_train_iters = [], [], [], [], []
            final_errs, init_errs = [], []
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                num_train_iter, avg_err = model.train_batch(
                    img_batch,
                    label_batch,
                    cfg.infer.num_train_iters,
                    init_std=cfg.infer.init_std,
                    fixed_preds=cfg.infer.fixed_preds_train,
                    use_amort=cfg.model.train_amort,
                    thresh=cfg.infer.train_thresh,
                )
                final_errs.append(avg_err[-1])
                init_errs.append(avg_err[0])

                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

                pc_loss, amort_loss = model.get_losses()
                pc_err, amort_err = model.get_errors()
                pc_losses.append(pc_loss)
                pc_errs.append(pc_err)
                amort_losses.append(amort_loss)
                amort_errs.append(amort_err)
                num_train_iters.append(num_train_iter)

                if batch_id % cfg.exp.log_batch_every == 0:
                    pc_loss = sum(pc_losses) / (batch_id + 1)
                    pc_err = sum(pc_errs) / (batch_id + 1)
                    amort_loss = sum(amort_losses) / (batch_id + 1)
                    amort_err = sum(amort_errs) / (batch_id + 1)
                    num_iter = sum(num_train_iters) / (batch_id + 1)
                    msg = f"Batch [{batch_id}/{len(train_loader)}]: "
                    msg = msg + f"losses PC {pc_loss:.1f} Q {amort_loss:.1f} "
                    msg = msg + f"errors PC {pc_err:.1f} Q {amort_err:.1f} "
                    msg = msg + f"iterations: {num_iter}"
                    logging.info(msg)

                if cfg.optim.normalize_weights:
                    model.normalize_weights()

            metrics.final_errs.append(sum(final_errs) / len(train_loader))
            metrics.pc_losses.append(sum(pc_losses) / len(train_loader))
            metrics.pc_errs.append(sum(pc_errs) / len(train_loader))
            metrics.amort_losses.append(sum(amort_losses) / len(train_loader))
            metrics.amort_errs.append(sum(amort_errs) / len(train_loader))
            metrics.num_train_iters.append(sum(num_train_iters) / len(train_loader))
            metrics.init_errs.append(sum(init_errs) / len(train_loader))

            if epoch % cfg.exp.test_every == 0:
                logging.info(f"Test @ epoch {epoch} [{len(test_loader)} batches]")
                hybrid_acc, pc_acc, amort_acc, num_test_iters, num_test_iters_pc = 0, 0, 0, [], []
                for _, (img_batch, label_batch) in enumerate(test_loader):

                    if cfg.exp.test_hybrid:
                        label_preds, num_test_iter, __path__ = model.test_batch(
                            img_batch,
                            cfg.infer.num_test_iters,
                            fixed_preds=cfg.infer.fixed_preds_test,
                            use_amort=True,
                            thresh=cfg.infer.test_thresh,
                        )
                        hybrid_acc = hybrid_acc + datasets.accuracy(label_preds, label_batch)
                        num_test_iters.append(num_test_iter)

                    if cfg.exp.test_pc:
                        label_preds, num_test_iter_pc, _ = model.test_batch(
                            img_batch,
                            cfg.infer.num_test_iters,
                            init_std=cfg.infer.init_std,
                            fixed_preds=cfg.infer.fixed_preds_test,
                            use_amort=False,
                            thresh=cfg.infer.test_thresh,
                        )
                        pc_acc = pc_acc + datasets.accuracy(label_preds, label_batch)
                        num_test_iters_pc.append(num_test_iter_pc)

                    if cfg.exp.test_amort:
                        label_preds = model.forward(img_batch)
                        amort_acc = amort_acc + datasets.accuracy(label_preds, label_batch)

                metrics.hybrid_acc.append(hybrid_acc / len(test_loader))
                metrics.pc_acc.append(pc_acc / len(test_loader))
                metrics.amort_acc.append(amort_acc / len(test_loader))
                metrics.num_test_iters.append(sum(num_test_iters) / len(test_loader))
                metrics.num_test_iters_pc.append(sum(num_test_iters_pc) / len(test_loader))
                logging.info("Metrics:")
                pprint({k: v[-1] for k, v in metrics.items()})

                logging.info(f"Generating image @ {cfg.exp.img_dir}/{epoch}.png")
                _, label_batch = next(iter(test_loader))
                img_preds = model.backward(label_batch)
                datasets.plot_imgs(img_preds, cfg.exp.img_dir + f"/{epoch}.png")

            utils.save_json(metrics, cfg.exp.log_dir + "/metrics.json")
            logging.info(f"Saved metrics @ {cfg.exp.log_dir}/metrics.json")
