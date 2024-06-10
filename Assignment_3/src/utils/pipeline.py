from os.path import join as ospj
from copy import deepcopy
from src.utils.constants import PROJECT_ROOT

def build_pipeline(cnn_config):
    datamodule_class = cnn_config['datamodule']
    data_args = cnn_config['data_args']

    dm = datamodule_class(**data_args)

    train_data_loader = dm.get_loader()
    valid_data_loader = dm.get_heldout_loader()

    test_data_args = deepcopy(data_args)
    test_data_args['training'] = False
    test_data_args['shuffle'] = False
    test_data_args['heldout_split'] = 0.0

    test_dm = datamodule_class(**test_data_args)
    test_loader = test_dm.get_loader()

    trainer_class = cnn_config['trainer_module']
    trainer_cnn = trainer_class(
        config = cnn_config, 
        log_dir = ospj(PROJECT_ROOT,'Logs'),
        train_loader=train_data_loader, 
        eval_loader=valid_data_loader,
    )

    return trainer_cnn, test_loader