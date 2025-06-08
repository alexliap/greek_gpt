import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(
    version_base="1.3", config_path="cli/conf/pretrain/", config_name="default.yaml"
)
def main(cfg: DictConfig):
    pretrain_lm = instantiate(cfg.model)

    train_dataloader = instantiate(cfg.train_dataloader).get_dataloder()

    # print(pretrain_lm.model.count_params() / int(1e6))

    trainer = instantiate(cfg.trainer)

    trainer.fit(pretrain_lm, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    main()
