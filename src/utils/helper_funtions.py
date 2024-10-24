from datetime import datetime
import getpass
import os


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_user():
    return getpass.getuser().replace("_", "")


def get_new_model_id():
    current_time = datetime.now()
    date_time = current_time.strftime("%Y-%m-%d_%Hh%Mm")
    milliseconds = int(current_time.timestamp() * 1000) % 1000
    date_time += f"{milliseconds:03d}ms"
    return f"{get_user()}_{date_time}"


def get_model_file_name(model_id: str, epoch_num: int):
    file_name = f"{model_id}_chkpt{epoch_num:03d}.pth"
    return f"{file_name}"


def setup_new_run():
    model_id = get_new_model_id()
    model_file_name = get_model_file_name(model_id, 0)
    return model_id, model_file_name


def handle_model_continuation(model_config):
    ##############################
    #  This function will be used to handle the scenario where we wish
    #  to continue training an existing model. This is not yet implemented.
    ##############################

    # TODO: Add logic here to handle continuation of a model

    # For now, setup new simulation with new model_id

    model_id, model_file_name = setup_new_run()
    if not model_config.model_id:
        model_config.model_id = model_id

    print(f"New run: {model_file_name}")

    return model_id, model_file_name
