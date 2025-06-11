import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(
    version_base="1.3", config_path="cli/conf/pretrain/", config_name="default.yaml"
)
def main(cfg: DictConfig):
    pretrain_lm = instantiate(cfg.model)

    # if cfg.ckpt_path:
    #     checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.trainer.accelerator)
    #     pretrain_lm.load_state_dict(checkpoint["state_dict"])

    if cfg.trainer.accelerator == "gpu":
        pretrain_lm = torch.compile(pretrain_lm)

    train_dataloader = instantiate(cfg.train_dataloader).get_dataloder()

    val_dataloader = instantiate(cfg.val_dataloader).get_dataloder()

    trainer = instantiate(cfg.trainer)

    trainer.fit(
        pretrain_lm, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
