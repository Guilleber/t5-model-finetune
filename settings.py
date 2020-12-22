class Config():
    def __init__(self, **entries):
        self.__dict__.update(entries)

configs = {
              "roberta-base-1": {
                  "pretrained_model_type": "bert",
                  "pretrained_model_name": "roberta-base",
                  "lr": 1e-5,
                  "batch_size": 32,
                  "adam_betas": (0.9, 0.98),
                  "weight_decay": 0.01
              },
              "roberta-large-1": {
                  "pretrained_model_type": "bert",
                  "pretrained_model_name": "roberta-large",
                  "lr": 1e-5,
                  "batch_size": 8,
                  "adam_betas": (0.9, 0.98),
                  "weight_decay": 0.01
              },
              "unifiedqa-base": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-t5-base",
                  "lr": 1e-3,
                  "batch_size": 8,
                  "max_len_in": 512,
                  "max_len_out": 100,
                  "optimizer_type": "AdaFactor"
              },
              "unifiedqa-large": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-t5-large",
                  "lr": 1e-3,
                  "batch_size": 8,
                  "max_len_in": 512,
                  "max_len_out": 100,
                  "optimizer_type": "AdaFactor"
              }
          }

def get_config_by_name(config_name):
    return Config(**configs[config_name])
