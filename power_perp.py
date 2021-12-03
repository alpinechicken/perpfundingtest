import numpy as np


def power_perp_price(spot, time, vol, drift, power):
    return spot ** power * np.exp(
        (power - 1) * (drift + power / 2 * vol ** 2) * time
    )

def everlasting_power_perp_price(spot, funding_period, vol, drift, power):
    return spot ** power * (
            1 / (2* np.exp(-funding_period/2 * (power - 1) * (2 * drift + power * vol **2)) - 1)
    )


def everlasting_power_perp_price_(spot, funding_period, vol, drift, power):
    """
    Solving directly: 
    """
    f = 1/funding_period
    k = (power - 1) * (2 * drift + power * vol ** 2) / 2
    return (spot ** power)/f  * 1 / (1 - f/(f-1) * np.exp(k/f)) 


def everlasting_power_perp_price3(spot, funding_frequency, vol, drift, power):
    """
    `time_to_tick` determines the first term in the geometric series.
    The formula located at https://www.paradigm.xyz/2021/08/power-perpetuals/
    assumes that `time_to_tick = funding_period`.
    """
    k = (power - 1) * (2 * drift + power * vol ** 2) / 2
    return (spot ** power) * np.exp(time_to_tick * k) / (2 - np.exp(funding_period * k))