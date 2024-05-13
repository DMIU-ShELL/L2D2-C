import numpy as np

def cusum(data, threshold=1, drift=0.0, ending=False):
    """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

    Parameters:
    ----------
    data : 1D array_like
        data to analyze (one-dimensional array).
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.

    Returns:
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).

    """
    data = np.atleast_1d(data).astype('float64')
    if data.ndim > 1:
        raise ValueError("CUSUM data must be 1D.")
    if not threshold > 0:
        raise ValueError("CUSUM threshold must be positive.")
    if not drift >= 0:
        raise ValueError("CUSUM drift must be non-negative.")
    if not isinstance(ending, bool):
        raise ValueError("Ending must be a Boolean.")

    # Get the sign (+/-) of the data
    s = np.sign(np.diff(data))

    # Start with a value at the threshold for fast comparison
    mask = np.hstack((s, 0))
    mask[mask > 0] = 1
    mask[mask < 0] = -1

    # Apply CUSUM algorithm
    ta = np.cumsum(mask)

    # Detection of change
    change_indices = np.where(np.abs(ta) > threshold)[0]

    # Adding drift
    if drift > 0:
        # Recursive method to estimate when the change started
        if len(change_indices) > 1:
            d = np.diff(change_indices)
            breakpoints = np.where(d > 1)[0]
            if len(breakpoints) > 0:
                breakpoints += 1
                breakpoints = np.insert(breakpoints, 0, 0)
                for bp in breakpoints:
                    ta[change_indices[bp:]] -= drift * (np.arange(len(change_indices[bp:])) + 1)
        # Estimate when the change started
        ta[ta < 0] = 0

    if ending:
        # Get ending of change
        if drift == 0:
            _, d = np.histogram(change_indices, bins=2)
            border = int(d[1])
            change_indices = change_indices[change_indices < border]
            if len(change_indices) > 0:
                ta = np.zeros_like(data, dtype=bool)
                ta[change_indices] = True
                ta[0] = False
                change_indices += 1
        if len(change_indices) > 0:
            mask = np.hstack((s, 0))
            mask[mask < 0] = 1
            mask[mask > 0] = -1
            ta = np.cumsum(mask)
            change_indices = np.where(np.abs(ta) > threshold)[0]
            taf = change_indices.copy()
            # Adding drift
            if drift > 0:
                for i, index in enumerate(change_indices):
                    ta[index:] -= drift * (np.arange(len(data[index:])) + 1)
                ta[ta < 0] = 0
            tai = np.zeros(len(change_indices), dtype=int)
            for i, index in enumerate(change_indices):
                index = np.where(ta[:index + 1][::-1] == 0)[0][0]
                tai[i] = index
                taf[i] -= 1
            return (tai, taf, np.diff(data[change_indices]))

    return change_indices



import matplotlib.pyplot as plt
import pandas as pd

# Generate some example data
np.random.seed(0)
df = pd.read_csv('log/ctgraph_seq_img_seed_sanity_check/MetaCTgraph-shell-dist-upz-seed-6652/agent_4/240411-184231/maha_cov_mean.csv')
data = df['distance'].to_numpy()

# Apply CUSUM algorithm
change_indices = cusum(data=data, threshold=6, drift=0, ending=False)

# Plotting
plt.figure(figsize=(200, 6))
plt.plot(data, color='blue', label='Data')
plt.plot(change_indices, data[change_indices], 'ro', markersize=8, label='Change Point')
plt.title('CUSUM Change Point Detection')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('cusumdata.pdf')
