# from https://github.com/BigRedT/deep_income/blob/master/global_constants.py

import yaml


income_const = yaml.load(
    open('/lfs/local/0/nomir/sMOOth/data/Adult/income.yml'),
    Loader=yaml.FullLoader)