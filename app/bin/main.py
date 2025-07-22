from bridge.application.handlers import LoggingConfigHandler, TrainModelHandler
from bridge.domain.model.utils import (
    load_configs,
    load_configs_dict,
)
from bridge.domain.model.utils import (
    print_scalable_config_pretty as print_config,
)


def main():
    print_config(load_configs())
    configs_dict = load_configs_dict()
    TrainModelHandler(**configs_dict).initiate_model_training()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
