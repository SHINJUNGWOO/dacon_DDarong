from model.trainer import Trainer


trainer = Trainer(config_pth="./config.yaml")
trainer.fit()