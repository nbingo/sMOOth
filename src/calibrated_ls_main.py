import subprocess

from src.ls_main import main


def start_ls(preference_vector_idx: int, gpu: int):
    # TODO: Change output dir
    command = f'CUDA_VISIBLE_DEVICES={gpu} python src/main.py ' \
              f'--config-file src/configs/adult/adult_mlp_calibrated_ls.py ' \
              f'train.preference_ray_idx={preference_vector_idx} ' \
              f'train.output_dir=./output/adult/ls/{preference_vector_idx}'
    subprocess.run(command, shell=True, check=True)
    return gpu


if __name__ == '__main__':
    main()
