# from https://github.com/BigRedT/deep_income/blob/master/global_constants.py

import yaml


income_const = yaml.load(
    open('income.yml'),
    Loader=yaml.FullLoader)