# %%
import numpy as np
from gbm import *
from black_scholes import black_scholes
from power_perp import power_perp_price, everlasting_power_perp_price

# Demonstrate sampling gives the same result as black scholes
def test_black_scholes():
    num_samples = 10000
    spot = 2
    time = 0.5
    vol = 0.8
    drift = 0.2
    samples = get_samples(num_samples,spot,time,vol,drift)

    strike = spot * 1.1

    discounted_payoff = np.mean(np.maximum(samples-strike,0) * np.exp(time*drift*-1))

    calced = black_scholes(spot, strike, time, drift, vol)

    assert np.abs(discounted_payoff - calced)/calced < 0.05, (discounted_payoff, calced)

# Demonstrate our power perp pricing is the same as sampling
def test_power_perp():
    num_samples = 100000
    spot = 2
    time = 0.5
    vol = 1.2
    drift = 0.2
    power = 3
    samples = get_samples(num_samples,spot,time,vol,drift)

    discounted_payoff = np.mean(samples**power * np.exp(time*drift*-1))

    calced = power_perp_price(spot, time, vol, drift, power)

    assert np.abs(discounted_payoff - calced)/calced < 0.05, (discounted_payoff, calced)

def test_everlasting_power_perp():
    spot = 2
    vol = 1.2
    drift = 0.2
    power = 3
    funding_period = 1/7

    num_iters = 1000
    est = 0
    for i in range(1,num_iters,1):
        est += power_perp_price(spot, i * funding_period, vol, drift, power) / 2**i

    calced = everlasting_power_perp_price(spot, funding_period, vol, drift, power)

    assert np.abs(est - calced)/est < 0.05, (est, calced)


def test_everlasting_power_perp2():
    num_samples = 10000
    spot = 2
    rate = 0
    time = 0.5
    freq = 1/365
    vol = 1.2
    drift = 0.2
    power = 2
    
    # GBM path
    # Funding factor (perp form)
    FF = 1/(2*np.exp( -freq* 0.5* (power-1) * (2 * rate + power * vol**2)) - 1)
    # Funding factor (expiring form)
    #FF = np.exp(freq* 0.5* (power - 1) * (2 * rate + power * vol**2))
    # Price function
    Minf = lambda S: S**power *FF

    def simPerp(time):
        # GBM path
        S = liteGBM(S0=spot, mu=freq*rate, sigma=vol*np.sqrt(freq), T=np.floor(time/freq))
        # Funding path (Mark - Index)
        d = [Minf(s)-s**2 for s in S[1:]]
        # Total cash for long power perp (cost - funding + sale) 
        return -Minf(spot) - np.sum(d) + Minf(S[-1])

    total_return = [simPerp(time) for _ in range(num_samples)]
    assert np.abs(np.mean(total_return/Minf(spot))) < 0.01

if __name__ == "__main__":
    test_everlasting_power_perp()
    # Test along paths
    test_everlasting_power_perp2()

    for period in range(1,100):
        print((period,everlasting_power_perp_price(2,1/period,1.2,0,3)))

# %%
test_everlasting_power_perp2()
# %%
