import numpy as np


def power_perp_price(spot, time, vol, drift, power):
    return spot ** power * np.exp(
        (power - 1) * (drift + power / 2 * vol ** 2) * time
    )

def everlasting_power_perp_price(spot, funding_period, vol, drift, power, time_to_tick=0):
    """
    `time_to_tick` determines the first term in the geometric series.
    The formula located at https://www.paradigm.xyz/2021/08/power-perpetuals/
    assumes that `time_to_tick = funding_period`.
    """
    k = (power - 1) * (2 * drift + power * vol ** 2) / 2
    return (spot ** power) * np.exp(time_to_tick * k) / (2 - np.exp(funding_period * k))
