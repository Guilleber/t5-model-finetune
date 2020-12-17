class Config():
    def __init__(self, **entries):
        self.__dict__.update(entries)

configs = {
              "roberta-base1": {
                  "pretrained_model_name": "roberta-base",
                  "lr": 1e-5,
                  "batch_size": 32
              },
              "roberta-large1": {
                  "pretrained_model_name": "roberta-large",
                  "lr": 1e-5,
                  "batch_size": 8
              }
          }

def get_config_by_name(config_name):
    return Config(**configs[config_name])
