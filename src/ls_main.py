from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait, ALL_COMPLETED
from detectron2.config import LazyConfig
from detectron2.engine import (
    default_argument_parser,
)
from queue import Queue
from pymoo.factory import get_reference_directions
import torch
import subprocess


def start_ls(preference_vector_idx: int, gpu: int):
    # TODO: Change output dir
    command = f'CUDA_VISIBLE_DEVICES={gpu} python src/main.py --config-file src/configs/adult/adult_mlp_ls.py ' \
              f'train.preference_vector_idx={preference_vector_idx} ' \
              f'train.output_dir=./output/ls/adult/{preference_vector_idx}'
    subprocess.run(command, shell=True, check=True)
    return gpu


def main():
    # Need to change output dir and device and preference vector index
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    print(cfg.train.gpus)
    gpu_q = Queue()
    for gpu in cfg.train.gpus:
        gpu_q.put(gpu)
    num_ref_dirs = get_reference_directions('das-dennis', len(cfg.model.loss_fn),
                                            n_partitions=cfg.train.num_preference_vector_partitions).shape[0]
    futures = set()

    with ProcessPoolExecutor(max_workers=len(cfg.train.gpus), mp_context=torch.multiprocessing.get_context('spawn')) \
            as executor:
        for idx in range(num_ref_dirs):
            # Check if there's an available gpu to use
            if gpu_q.empty():  # If empty, then wait until a job finishes
                print('No GPUs available, waiting for a process to finish...')
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                print(f'{len(done)} processes finished. Adding their GPUs to the available queue.')
                futures -= done
                for future in futures:
                    gpu_q.put(future.result())
            # now that we have gpus to use, start a new task
            gpu = gpu_q.get()
            print(f'Going to submit new job on gpu {gpu} with preference vector index {idx}')
            futures.add(executor.submit(start_ls, idx, gpu))
        wait(futures, return_when=ALL_COMPLETED)


if __name__ == '__main__':
    main()
