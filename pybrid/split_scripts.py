import logging
from pprint import pprint
import torch
from pybrid import utils, datasets, optim
from pybrid.models.hybrid import HybridModel


def setup(cfg):
    cfg = utils.setup_experiment(cfg)

    datasets.download_mnist()
    train_dataset_1 = datasets.MNIST(
        train=True,
        scale=cfg.data.label_scale,
        size=cfg.data.train_size,
        normalize=cfg.data.normalize,
        labels=[0, 1, 2, 3, 4],
    )
    test_dataset_1 = datasets.MNIST(
        train=False,
        scale=cfg.data.label_scale,
        size=cfg.data.test_size,
        normalize=cfg.data.normalize,
        labels=[0, 1, 2, 3, 4],
    )
    train_dataset_2 = datasets.MNIST(
        train=True,
        scale=cfg.data.label_scale,
        size=cfg.data.train_size,
        normalize=cfg.data.normalize,
        labels=[5, 6, 7, 8, 9],
    )
    test_dataset_2 = datasets.MNIST(
        train=False,
        scale=cfg.data.label_scale,
        size=cfg.data.test_size,
        normalize=cfg.data.normalize,
        labels=[5, 6, 7, 8, 9],
    )
    train_loader_1 = datasets.get_dataloader(train_dataset_1, cfg.optim.batch_size)
    test_loader_1 = datasets.get_dataloader(test_dataset_1, cfg.optim.batch_size)
    train_loader_2 = datasets.get_dataloader(train_dataset_2, cfg.optim.batch_size)
    test_loader_2 = datasets.get_dataloader(test_dataset_2, cfg.optim.batch_size)
    logging.info(f"Loaded MNIST [train {len(train_loader_1)}] [test {len(test_loader_1)}]")
    logging.info(f"Loaded MNIST [train {len(train_loader_2)}] [test {len(test_loader_2)}]")

    model = HybridModel(
        nodes=cfg.model.nodes,
        amort_nodes=cfg.model.amort_nodes,
        mu_dt=cfg.infer.mu_dt,
        act_fn=utils.get_act_fn(cfg.model.act_fn),
        use_bias=cfg.model.use_bias,
        kaiming_init=cfg.model.kaiming_init,
    )
    logging.info(f"Loaded model {model}")
    optimizer = optim.get_optim(
        model.params,
        cfg.optim.name,
        cfg.optim.lr,
        amort_lr=cfg.optim.amort_lr,
        batch_scale=cfg.optim.batch_scale,
        grad_clip=cfg.optim.grad_clip,
        weight_decay=cfg.optim.weight_decay,
    )
    return cfg, train_loader_1, train_loader_2, test_loader_1, test_loader_2, model, optimizer


def main(cfg):
    cfg, train_loader_1, train_loader_2, test_loader_1, test_loader_2, model, optimizer = setup(cfg)

    train_loader = train_loader_1
    test_loader = test_loader_1

    with torch.no_grad():
        metrics = utils.to_attr_dict(
            {
                "batch_idx": [],
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
        global_batch_id = 0
        curr_epoch = 0
        pc_losses, pc_errs, amort_losses, amort_errs, num_train_iters = [], [], [], [], []
        final_errs, init_errs = [], []
        while global_batch_id < cfg.exp.num_batches:

            for batch_id, (img_batch, label_batch) in enumerate(train_loader):

                if global_batch_id == cfg.exp.switch:
                    logging.info("Switched dataset")
                    train_loader = train_loader_2
                    test_loader = test_loader_2

                global_batch_id = global_batch_id + 1
                num_train_iter, avg_err = model.train_batch(
                    img_batch,
                    label_batch,
                    cfg.infer.num_train_iters,
                    init_std=cfg.infer.init_std,
                    fixed_preds=cfg.infer.fixed_preds_train,
                    use_amort=cfg.model.train_amort,
                    thresh=cfg.infer.train_thresh,
                    no_backward=cfg.infer.no_backward
                )

                optimizer.step(
                    curr_epoch=curr_epoch,
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
                final_errs.append(avg_err[-1])
                init_errs.append(avg_err[0])

                if (global_batch_id % cfg.exp.batches_per_epoch == 0) and global_batch_id > 0:
                    metrics.batch_idx.append(global_batch_id)
                    metrics.final_errs.append(sum(final_errs) / cfg.exp.batches_per_epoch)
                    metrics.pc_losses.append(sum(pc_losses) / cfg.exp.batches_per_epoch)
                    metrics.pc_errs.append(sum(pc_errs) / cfg.exp.batches_per_epoch)
                    metrics.amort_losses.append(sum(amort_losses) / cfg.exp.batches_per_epoch)
                    metrics.amort_errs.append(sum(amort_errs) / cfg.exp.batches_per_epoch)
                    metrics.num_train_iters.append(sum(num_train_iters) / cfg.exp.batches_per_epoch)
                    metrics.init_errs.append(sum(init_errs) / cfg.exp.batches_per_epoch)

                    logging.info(f"Test @ epoch {curr_epoch} [batch {global_batch_id}]")
                    hybrid_acc, pc_acc, amort_acc = 0, 0, 0
                    num_test_iters, num_test_iters_pc = [], []

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

                    logging.info(f"Generating image @ {cfg.exp.img_dir}/{curr_epoch}.png")
                    _, label_batch = next(iter(test_loader))
                    img_preds = model.backward(label_batch)
                    datasets.plot_imgs(img_preds, cfg.exp.img_dir + f"/{curr_epoch}.png")

                    utils.save_json(metrics, cfg.exp.log_dir + "/metrics.json")
                    logging.info(f"Saved metrics @ {cfg.exp.log_dir}/metrics.json")

                    pc_losses, pc_errs, amort_losses, amort_errs, num_train_iters = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                    final_errs, init_errs = [], []
                    curr_epoch = curr_epoch + 1
