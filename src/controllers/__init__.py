REGISTRY = {}

from .basic_controller import BasicMAC
from .token_controller import TokenMAC
from .entity_controller import EntityMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["token_mac"] = TokenMAC
REGISTRY["entity_mac"] = EntityMAC
