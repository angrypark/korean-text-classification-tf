import os


def create_dirs(config):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        dir = "runs/{}".format(config.name)
        if not os.path.exists(dir):
            os.makedirs(dir)
            os.makedirs(dir + "/checkpoint")
            os.makedirs(dir + "/summaries")
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
