from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib

from functools import partial
from itertools import combinations_with_replacement, product


class SMACMap(lib.Map):
    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


mt_map_param_registry = {
    "empty_passive": {}
}

for name in mt_map_param_registry.keys():
    globals()[name] = type(name, (SMACMap,), dict(filename=name))


def get_all_unique_teams(all_types, min_len, max_len):
    all_com_teams = []
    for i in range(min_len, max_len + 1):
        all_com_teams += list(combinations_with_replacement(all_types, i))

    all_cnt_teams = []
    for team in all_com_teams:
        team_types = list(set(team))
        team_counts = list(zip([team.count(typ) for typ in team_types], team_types))
        all_cnt_teams.append(team_counts)
    return all_cnt_teams


type_name2health = {'Baneling': 30.0, 'Colossus': 200.0, 'Hydralisk': 80.0,
                    'Marauder': 125.0, 'Marine': 45.0, 'Medivac': 150.0,
                    'Stalker': 80.0, 'Zealot': 100.0, 'Zergling': 35.0}


type_name2shield = {'Baneling': 0.0, 'Colossus': 150.0, 'Hydralisk': 0.0,
                    'Marauder': 0.0, 'Marine': 0.0, 'Medivac': 0.0,
                    'Stalker': 80.0, 'Zealot': 50.0, 'Zergling': 0.0}


def symmetric_armies(army_spec, ally_centered=False, rotate=False, separation=10,
                     jitter=0, episode_limit=100, n_extra_tags=0,
                     map_name="empty_passive", map_type=None):

    reward_health_shield_max = 0

    unique_sub_teams = []
    for unit_types, n_unit_range in army_spec:
        unique_sub_teams.append(get_all_unique_teams(unit_types, n_unit_range[0], n_unit_range[1]))
        reward_health_shield_max += max([type_name2health[unit_type] + type_name2shield[unit_type] for unit_type in unit_types]) * n_unit_range[1]
    unique_teams = [sum(prod, []) for prod in product(*unique_sub_teams)]

    scenarios = list(zip(unique_teams, unique_teams))
    max_types_and_units_team = sorted(unique_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_scenario = (max_types_and_units_team, max_types_and_units_team)

    scenario_dict = {'scenarios': scenarios,
                     'max_types_and_units_scenario': max_types_and_units_scenario,
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'n_extra_tags': n_extra_tags,
                     'map_name': map_name,
                     'map_type': map_type,
                     'reward_health_shield_max': reward_health_shield_max}

    return scenario_dict


mt_scenario_registry = {
    "3-8sz_symmetric": partial(symmetric_armies,
                               [(('Stalker', 'Zealot'), (3, 8))],
                               ally_centered=False,
                               rotate=True,
                               separation=14,
                               jitter=1,
                               episode_limit=150,
                               n_extra_tags=0,
                               map_name="empty_passive",
                               map_type='stalkers_and_zealots'),
    "5-11sz_symmetric": partial(symmetric_armies,
                                [(('Stalker', 'Zealot'), (5, 11))],
                                ally_centered=False,
                                rotate=True,
                                separation=14,
                                jitter=1,
                                episode_limit=150,
                                n_extra_tags=0,
                                map_name="empty_passive",
                                map_type='stalkers_and_zealots'),
    "3-8MMM_symmetric": partial(symmetric_armies,
                                [(('Marine', 'Marauder'), (3, 6)), (('Medivac',), (0, 2))],
                                ally_centered=False,
                                rotate=True,
                                separation=14,
                                jitter=1,
                                episode_limit=150,
                                n_extra_tags=0,
                                map_name="empty_passive",
                                map_type='MMM'),
    "5-11MMM_symmetric": partial(symmetric_armies,
                                 [(('Marine', 'Marauder'), (5, 8)), (('Medivac',), (0, 3))],
                                 ally_centered=False,
                                 rotate=True,
                                 separation=14,
                                 jitter=1,
                                 episode_limit=150,
                                 n_extra_tags=0,
                                 map_name="empty_passive",
                                 map_type='MMM'),
    "3-8csz_symmetric": partial(symmetric_armies,
                                [(('Stalker', 'Zealot'), (3, 6)), (('Colossus',), (0, 2))],
                                ally_centered=False,
                                rotate=True,
                                separation=14,
                                jitter=1,
                                episode_limit=150,
                                n_extra_tags=0,
                                map_name="empty_passive",
                                map_type='colossus_stalkers_zealots'),
    "5-11csz_symmetric": partial(symmetric_armies,
                                 [(('Stalker', 'Zealot'), (5, 8)), (('Colossus',), (0, 3))],
                                 ally_centered=False,
                                 rotate=True,
                                 separation=14,
                                 jitter=1,
                                 episode_limit=150,
                                 n_extra_tags=0,
                                 map_name="empty_passive",
                                 map_type='colossus_stalkers_zealots'),
}


def get_mt_scenario_registry():
    return mt_scenario_registry
