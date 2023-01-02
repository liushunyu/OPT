from .entity_opt_q_learner import QLearner as EntityOPTQLearner
from .token_opt_q_learner import QLearner as TokenOPTQLearner

REGISTRY = {}

REGISTRY["entity_opt_q_learner"] = EntityOPTQLearner
REGISTRY["token_opt_q_learner"] = TokenOPTQLearner
