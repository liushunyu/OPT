REGISTRY = {}

from .entity_opt_agent import EntityOPTAgent
REGISTRY["entity_opt_agent"] = EntityOPTAgent

from .token_opt_agent import TokenOPTAgent
REGISTRY["token_opt_agent"] = TokenOPTAgent
