from model.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(config_pth="./config.yaml")
    trainer.test()
