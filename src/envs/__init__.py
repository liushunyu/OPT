from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env, StarCraft2MTEnv
from .prey import PreyEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2mt"] = partial(env_fn, env=StarCraft2MTEnv)
REGISTRY["prey"] = partial(env_fn, env=PreyEnv)
