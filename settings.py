class Config():
    def __init__(self, **entries):
        self.__dict__.update(entries)

configs = {
              "roberta-base1": {
                  "pretrained_model_name": "roberta-base",
                  "lr": 1e-5,
                  "batch_size": 32,
                  "betas": (0.9, 0.98),
                  "weight_decay": 0.01
              },
              "roberta-large1": {
                  "pretrained_model_name": "roberta-large",
                  "lr": 1e-5,
                  "batch_size": 8,
                  "betas": (0.9, 0.98),
                  "weight_decay": 0.01
              },
              "roberta-large2": {
                  "pretrained_model_name": "roberta-large",
                  "lr": 1e-6,
                  "batch_size": 8,
                  "betas": (0.9, 0.98),
                  "weight_decay": 0.01
              }
          }

def get_config_by_name(config_name):
    return Config(**configs[config_name])
