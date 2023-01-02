from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import smac_maps
from . import mt_maps


def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]


def get_mt_scenario_params(scenario_name):
    mt_scenario_param_registry = mt_maps.get_mt_scenario_registry()
    return mt_scenario_param_registry[scenario_name]()
