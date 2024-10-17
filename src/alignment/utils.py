from datetime import datetime

from transformers import TrainerCallback

from pynvml import *


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")


class GpuUtilPrintCallBack(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and (state.global_step + 1) % 100 == 0:
            print(
                "[", datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"), "]\t", logs
            )

        if (state.global_step + 1) % 500 == 0:
            print(get_gpu_utilization())

    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        print(get_gpu_utilization())

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(get_gpu_utilization())


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
