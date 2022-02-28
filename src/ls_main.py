from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait, ALL_COMPLETED
from detectron2.config import LazyConfig
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)
from queue import Queue
from pymoo.factory import get_reference_directions
import torch
import subprocess


def start_ls(preference_vector_idx: int, gpu: int):
    # TODO: Change output dir
    command = f'CUDA_VISIBLE_DEVICES={gpu} python src/main.py --config-file /src/configs/adult/adult_mlp_ls.py ' \
              f'MODEL.PREFERENCE_VECTOR_IDX={preference_vector_idx} ' \
              f'TRAIN.OUTPUT_DIR=./output/ls/adult/{preference_vector_idx}'
    command = command.split(' ')
    subprocess.run(command)
    return gpu

def main():
    # Need to change output dir and device and preference vector index
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    gpu_q = Queue(cfg.train.gpus)
    num_ref_dirs = get_reference_directions('das-dennis', len(cfg.model.loss_fn),
                                            n_partitions=cfg.train.num_preference_vector_partitions).shape[0]
    futures = set()

    with ProcessPoolExecutor(max_workers=len(cfg.train.gpus), mp_context=torch.multiprocessing.get_context('spawn')) \
            as executor:
        for idx in range(num_ref_dirs):
            # Check if there's an available gpu to use
            if gpu_q.empty():       # If empty, then wait until a job finishes
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                futures -= done
                for future in futures:
                    gpu_q.put(future.result())
            # now that we have gpus to use, start a new task
            futures.add(executor.submit(start_ls, idx, gpu_q.get()))
        wait(futures, return_when=ALL_COMPLETED)


if __name__ == '__main__':
    main()