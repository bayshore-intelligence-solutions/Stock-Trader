import silence_tensorflow
from tensorflow.compat.v1 import ConfigProto


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


# def validate_tf_1_14_plus(version: str) -> bool:
#     versions = version.split('.')
#     major = int(versions[0])
#     minor = int(versions[1])
#     if major != 1 and minor < 14:
#         return False
#     return True


def get_tensorflow_config():
    run_config = ConfigProto()
    run_config.gpu_options.allow_growth = False
    run_config.gpu_options.per_process_gpu_memory_fraction = 1