from minihack import MiniHackNavigation
from gym.envs.registration import register

class MiniHackN3(MiniHackNavigation):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            max_num_rooms=3,
            room_min_size=3,
            room_max_size=4,
            **kwargs,
        )

register(
    id="MiniHack-N3-v0",
    entry_point="__main__:MiniHackN3",
)