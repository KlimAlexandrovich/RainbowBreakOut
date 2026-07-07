import os
import sys

if os.path.abspath("package") not in sys.path:
    sys.path.append(os.path.abspath("package"))

from package.trainer import Trainer, TrainConfig

if __name__ == "__main__":
    config = TrainConfig(device=None)  # None -> авто: cuda (Kaggle) / mps / cpu
    print(config)
    trainer = Trainer(config)
    trainer.run()
