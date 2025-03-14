#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiscale Entropy (MSE) Analysis for EEG Data

This module provides functions for calculating various entropy measures on time series data,
particularly focused on EEG signal analysis. It includes implementations of:
- Shannon Entropy
- Sample Entropy
- Multiscale Entropy
- Permutation Entropy

Originally from a Jupyter notebook, this script has been optimized for use as a Python module.
"""

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import itertools
from math import factorial
import datetime
import math
from scipy import signal
import pandas as pd
from scipy.signal import hilbert, chirp

def _embed(x, order=3, delay=1):
    """
    Time-delay embedding of a time series.
    
    Parameters
    ----------
    x : 1d-array
        Time series data
    order : int
        Embedding dimension (order)
    delay : int
        Time delay between consecutive embedded points
        
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series
    """
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


def util_pattern_space(time_series, lag, dim):
    """
    Create a set of sequences with given lag and dimension.
    
    Parameters
    ----------
    time_series : array-like
        Input time series data
    lag : int
        Lag between beginning of sequences
    dim : int
        Dimension (number of patterns)
        
    Returns
    -------
    pattern_space : 2D array
        Matrix of embedded vectors
    """
    n = len(time_series)

    if lag * dim > n:
        raise ValueError('Result matrix exceeded size limit, try to change lag or dim.')
    elif lag < 1:
        raise ValueError('Lag should be greater or equal to 1.')

    pattern_space = np.empty((n - lag * (dim - 1), dim))
    for i in range(n - lag * (dim - 1)):
        for j in range(dim):
            pattern_space[i][j] = time_series[i + j * lag]

    return pattern_space


def util_standardize_signal(time_series):
    """
    Standardize a time series (z-score normalization).
    
    Parameters
    ----------
    time_series : array-like
        Input time series data
        
    Returns
    -------
    standardized_series : array-like
        Z-score normalized time series
    """
    return (time_series - np.mean(time_series)) / np.std(time_series)


def util_granulate_time_series(time_series, scale):
    """
    Extract coarse-grained time series for multiscale entropy analysis.
    
    Parameters
    ----------
    time_series : array-like
        Input time series data
    scale : int
        Scale factor for coarse-graining
        
    Returns
    -------
    coarse_grained : array-like
        Coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    temp = np.reshape(time_series[0:b*scale], (b, scale))
    cts = np.mean(temp, axis=1)
    return cts


def shannon_entropy(time_series):
    """
    Calculate the Shannon Entropy of the sample data.
    
    Parameters
    ----------
    time_series : array-like or string
        Input time series data
        
    Returns
    -------
    entropy : float
        Shannon entropy value
    """
    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency distribution
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy calculation
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    ent = -ent
    return ent


def sample_entropy(time_series, sample_length, tolerance=None):
    """
    Calculate the sample entropy of degree m of a time series.
    
    This method uses Chebyshev norm for distance calculations.
    It is quite fast for random data, but can be slower if there is
    structure in the input time series.
    
    Parameters
    ----------
    time_series : array-like
        Input time series data
    sample_length : int
        Length of longest template vector (m+1 in literature)
    tolerance : float, optional
        Tolerance for matching (defaults to 0.1 * std(time_series))
        
    Returns
    -------
    entropy : array-like
        Array of sample entropies:
        SE[k] is ratio "#templates of length k+1" / "#templates of length k"
        where "#templates of length 0" = n*(n - 1) / 2, by definition
    
    References
    ----------
    [1] http://en.wikipedia.org/wiki/Sample_Entropy
    [2] http://physionet.incor.usp.br/physiotools/sampen/
    [3] Costa, M., Goldberger, A.L., Peng, C.K. (2005). Multiscale entropy analysis
        of biological signals. Physical Review E, 71(2), 021906.
    """
    # The code below follows the sample length convention of Ref [1] so:
    M = sample_length - 1

    time_series = np.array(time_series)
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)

    # Ntemp is a vector that holds the number of matches
    # N[k] holds matches for templates of length k
    Ntemp = np.zeros(M + 2)
    # Templates of length 0 matches by definition:
    Ntemp[0] = n * (n - 1) / 2

    for i in range(n - M - 1):
        template = time_series[i:(i+M+1)]  # We have 'M+1' elements in the template
        rem_time_series = time_series[i+1:]

        searchlist = np.nonzero(np.abs(rem_time_series - template[0]) < tolerance)[0]

        go = len(searchlist) > 0
        length = 1

        Ntemp[length] += len(searchlist)

        while go:
            length += 1
            nextindxlist = searchlist + 1
            nextindxlist = nextindxlist[nextindxlist < n - 1 - i]  # Remove candidates too close to the end
            nextcandidates = rem_time_series[nextindxlist]
            hitlist = np.abs(nextcandidates - template[length-1]) < tolerance
            searchlist = nextindxlist[hitlist]

            Ntemp[length] += np.sum(hitlist)

            go = any(hitlist) and length < M + 1

    sampen = -np.log(Ntemp[1:] / Ntemp[:-1])
    return sampen


def multiscale_entropy(time_series, sample_length, tolerance=None, maxscale=None):
    """
    Calculate the Multiscale Entropy of the given time series.
    
    Parameters
    ----------
    time_series : array-like
        Input time series data
    sample_length : int
        Length of template vector (m+1 in literature)
    tolerance : float, optional
        Tolerance for matching (default = 0.1*std(time_series))
    maxscale : int, optional
        Maximum scale to calculate (default = length of time_series)
        
    Returns
    -------
    mse : array-like
        Multiscale entropy values for different scales
        
    References
    ----------
    [1] Costa, M., Goldberger, A.L., Peng, C.K. (2005). Multiscale entropy analysis
        of biological signals. Physical Review E, 71(2), 021906.
    """
    if tolerance is None:
        # Fix the tolerance at this level
        tolerance = 0.5 * np.std(time_series)  # Originally 0.1, changed to 0.5
    if maxscale is None:
        maxscale = len(time_series)

    mse = np.zeros(maxscale)

    for i in range(maxscale):
        now = datetime.datetime.now()
        if i == 0:
            print(f"Starting MSE calculation: {now.strftime('%H:%M:%S')}")
        elif i % 10 == 0:
            print(f"Processing scale {i}: {now.strftime('%H:%M:%S')}")
            
        # Create coarse-grained time series for current scale
        temp = util_granulate_time_series(time_series, i+1)
        
        # Calculate sample entropy for this scale
        mse[i] = sample_entropy(temp, sample_length, tolerance)[-1]
        
    return mse


def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    """
    Calculate Permutation Entropy of a time series.
    
    Parameters
    ----------
    time_series : array-like
        Input time series data
    order : int
        Order of permutation entropy (embedding dimension)
    delay : int
        Time delay between points in the embedding
    normalize : bool
        If True, divide by log2(factorial(order)) to normalize between 0 and 1
        
    Returns
    -------
    pe : float
        Permutation Entropy value
        
    References
    ----------
    [1] Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
        measure for time series. Physical Review Letters, 88(17), 174102.
    [2] Zanin, M., et al. (2012). Permutation entropy and its main biomedical and
        econophysics applications: a review. Entropy, 14(8), 1553-1577.
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    
    # Embed the time series
    embedded = _embed(x, order=order, delay=delay)
    
    # Get the permutation patterns by sorting each embedded vector
    sorted_idx = embedded.argsort(axis=1)
    
    # Convert permutation patterns to unique integers
    hashval = (np.multiply(sorted_idx, hashmult)).sum(axis=1)
    
    # Count the frequency of each pattern
    _, c = np.unique(hashval, return_counts=True)
    
    # Calculate the probabilities
    p = c / c.sum()
    
    # Calculate the entropy
    pe = -np.sum(p * np.log2(p))
    
    if normalize:
        pe /= np.log2(factorial(order))
        
    return pe


def weighted_permutation_entropy(time_series, order=3, delay=1, normalize=False):
    """
    Calculate Weighted Permutation Entropy of a time series.
    
    Parameters
    ----------
    time_series : array-like
        Input time series data
    order : int
        Order of permutation entropy (embedding dimension)
    delay : int
        Time delay between points in the embedding
    normalize : bool
        If True, divide by log2(factorial(order)) to normalize between 0 and 1
        
    Returns
    -------
    wpe : float
        Weighted Permutation Entropy value
        
    References
    ----------
    [1] Fadlallah, B., et al. (2013). Weighted-permutation entropy: A complexity
        measure for time series incorporating amplitude information. Physical
        Review E, 87(2), 022911.
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    
    # Embed the time series
    embedded = _embed(x, order=order, delay=delay)
    
    # Calculate the variance (weight) of each embedded vector
    weights = np.var(embedded, axis=1)
    
    # Get the permutation patterns by sorting each embedded vector
    sorted_idx = embedded.argsort(axis=1)
    
    # Convert permutation patterns to unique integers
    hashval = (np.multiply(sorted_idx, hashmult)).sum(axis=1)
    
    # Count the weighted frequency of each pattern
    unique_patterns, inv_idx = np.unique(hashval, return_inverse=True)
    weighted_counts = np.zeros(len(unique_patterns))
    
    for i, w in enumerate(weights):
        weighted_counts[inv_idx[i]] += w
    
    # Calculate the probabilities
    p = weighted_counts / weighted_counts.sum()
    
    # Calculate the entropy
    wpe = -np.sum(p * np.log2(p))
    
    if normalize:
        wpe /= np.log2(factorial(order))
        
    return wpe


def calculate_entropy_measures(signal, sample_length=2, max_scale=20):
    """
    Calculate multiple entropy measures for a given signal.
    
    Parameters
    ----------
    signal : array-like
        Input time series data
    sample_length : int, optional
        Length of template vector for sample entropy (default=2)
    max_scale : int, optional
        Maximum scale for multiscale entropy (default=20)
        
    Returns
    -------
    results : dict
        Dictionary containing various entropy measures
    """
    # Standardize the signal
    std_signal = util_standardize_signal(signal)
    
    # Calculate Shannon entropy
    se = shannon_entropy(std_signal)
    
    # Calculate Sample entropy
    samp_e = sample_entropy(std_signal, sample_length)[-1]
    
    # Calculate Multiscale entropy (first 20 scales)
    mse = multiscale_entropy(std_signal, sample_length, maxscale=max_scale)
    
    # Calculate Permutation entropy
    pe = permutation_entropy(std_signal, order=3, delay=1, normalize=True)
    
    # Calculate Weighted Permutation entropy
    wpe = weighted_permutation_entropy(std_signal, order=3, delay=1, normalize=True)
    
    return {
        'shannon_entropy': se,
        'sample_entropy': samp_e,
        'multiscale_entropy': mse,
        'permutation_entropy': pe,
        'weighted_permutation_entropy': wpe
    }


def plot_entropy_results(results, title="Entropy Analysis Results"):
    """
    Plot the results of entropy analysis.
    
    Parameters
    ----------
    results : dict
        Dictionary containing entropy measures
    title : str, optional
        Title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot Multiscale Entropy
    axes[0, 0].plot(range(1, len(results['multiscale_entropy'])+1), results['multiscale_entropy'], 'o-')
    axes[0, 0].set_title('Multiscale Entropy')
    axes[0, 0].set_xlabel('Scale Factor')
    axes[0, 0].set_ylabel('Sample Entropy')
    axes[0, 0].grid(True)
    
    # Bar plot for other entropy measures
    other_entropies = {k: v for k, v in results.items() if k != 'multiscale_entropy'}
    axes[0, 1].bar(other_entropies.keys(), other_entropies.values())
    axes[0, 1].set_title('Entropy Measures')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True)
    
    # Hide unused subplots
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("MSE_EEG.py - Multiscale Entropy Analysis for EEG Data")
    print("This module provides functions for calculating various entropy measures.")
    print("Import this module in your script to use its functions.")
    
    # Generate a sample signal for demonstration
    np.random.seed(42)
    sample_signal = np.random.randn(1000)
    
    print("\nCalculating entropy measures for a random signal...")
    results = calculate_entropy_measures(sample_signal, max_scale=10)
    
    print("\nEntropy Results:")
    for k, v in results.items():
        if k != 'multiscale_entropy':
            print(f"{k}: {v:.4f}")
    print(f"multiscale_entropy: array of length {len(results['multiscale_entropy'])}")
    
    print("\nPlotting results...")
    plot_entropy_results(results, "Random Signal Entropy Analysis")

a1 = read("/Users/cogmech/Documents/Neuro Complexity Matching/ResampledAudio/ResampledAudio/BerliozSymphResamp.wav")
arr1 = np.array(a1[1],dtype=float)
arr1 = np.mean(arr1,axis=1)

a2 = read("/Users/cogmech/Documents/Neuro Complexity Matching/ResampledAudio/ResampledAudio/CalvinHarrisResamp.wav")
arr2 = np.array(a2[1],dtype=float)
arr2 = np.mean(arr2,axis=1)

a3 = read("/Users/cogmech/Documents/Neuro Complexity Matching/ResampledAudio/ResampledAudio/HermitThrushResamp.wav")
arr3 = np.array(a3[1],dtype=float)
arr3 = np.mean(arr3,axis=1)

a4 = read("/Users/cogmech/Documents/Neuro Complexity Matching/ResampledAudio/ResampledAudio/TEDSynthResamp.wav")
arr4 = np.array(a4[1],dtype=float)
#arr4 = np.mean(arr4,axis=1)

a5 = read("/Users/cogmech/Documents/Neuro Complexity Matching/ResampledAudio/ResampledAudio/TEDTalkResamp.wav")
arr5 = np.array(a5[1],dtype=float)
arr5 = np.mean(arr5,axis=1)

amp1 = np.abs(hilbert(arr1))
amp2 = np.abs(hilbert(arr2))
amp3 = np.abs(hilbert(arr3))
amp4 = np.abs(hilbert(arr4))
amp5 = np.abs(hilbert(arr5))

plt.plot(range(len(amp2)),amp2)

samp1_mse = multiscale_entropy(amp1[0:143437], 3, None, 100)
samp2_mse = multiscale_entropy(amp2[0:143437], 3, None, 100)
samp3_mse = multiscale_entropy(amp3[0:143437], 3, None, 100)
samp4_mse = multiscale_entropy(amp4[0:143437], 3, None, 100)
samp5_mse = multiscale_entropy(amp5[0:143437], 3, None, 100)



plt.plot(range(1,101),samp1_mse, range(1,101),samp2_mse,range(1,101),samp3_mse,range(1,101),samp4_mse,range(1,101),samp5_mse)
plt.legend(['Symph','EDM','Bird','Ted','Synth'])
plt.title('2048hz Amplitude MSE')

"""## EEG"""

##EDM##
cl3_edm = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c3_edm.csv",header=None)
cl4_edm = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c4_edm.csv",header=None)

cl3_mse = []
cl4_mse = []

for i in range(len(cl3_edm)):
    ts = cl3_edm.loc[i,:].values
    amp = np.abs(hilbert(ts))
    halt = int(len(ts)/4)
    cl3_mse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

for i in range(len(cl4_edm)):
    ts = cl4_edm.loc[i,:].values
    amp = np.abs(hilbert(ts))
    halt = int(len(ts)/4)
    cl4_mse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

cl3_mean = np.mean(cl3_mse, axis=0)
cl4_mean = np.mean(cl4_mse, axis=0)

plt.plot(range(1,101),cl3_mean,range(1,101),samp2_mse)
plt.legend(['cluster3','edm'])

plt.plot(range(1,101),cl4_mean,range(1,101),samp2_mse)
plt.legend(['cluster4','edm'])

##Bird Song##
cl3_bird = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c3_hermit.csv",header=None)
cl4_bird = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c4_hermit.csv",header=None)

#cl3_birdmse = []
cl4_birdmse = []
#cl3_birdamp = []
cl4_birdamp = []

#for i in range(len(cl3_bird)):
    #ts = cl3_bird.loc[i,:].values
    #amp = np.abs(hilbert(ts))
    #halt = int(len(ts)/4)
    #cl3_birdamp.append(amp)
    #cl3_birdmse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

for i in range(len(cl4_edm)):
    ts = cl4_bird.loc[i,:].values
    amp = np.abs(hilbert(ts))
    halt = int(len(ts)/4)
    cl4_birdamp.append(amp)
    cl4_birdmse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

cl3_birdmean = np.mean(cl3_birdmse, axis=0)
cl4_birdmean = np.mean(cl4_birdmse, axis=0)

plt.plot(range(1,101),cl3_birdmean,range(1,101),cl3_mean)
plt.legend(['cl3_bird','cl3_edm'])
plt.title('Cluster3')

plt.plot(range(1,101),cl4_birdmean,range(1,101),cl4_mean)
plt.legend(['cl4_bird','cl4_edm'])
plt.title('Cluster 4')

plt.plot(range(1,101),cl3_birdmean,range(1,101),samp3_mse)
plt.legend(['eeg','stim'])
plt.title('Bird Song')

##Sine Ted##
cl3_sine = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c3_sine.csv",header=None)
cl3_ted = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c3_ted.csv",header=None)
cl4_sine = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c4_sine.csv",header=None)
cl4_ted = pd.read_csv("/Users/cogmech/Documents/Neuro Complexity Matching/comps/comps/c4_ted.csv",header=None)

cl3_sinemse = []
cl3_tedmse = []
cl4_sinemse = []
cl4_tedmse = []

cl3_sineamp = []
cl3_tedamp = []
cl4_sinedamp = []
cl4_tedamp = []

for i in range(len(cl3_sine)):
    ts = cl3_sine.loc[i,:].values
    amp = np.abs(hilbert(ts))
    halt = int(len(ts)/4)
    cl3_sineamp.append(amp)
    cl3_sinemse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

for i in range(len(cl3_ted)):
    ts = cl3_ted.loc[i,:].values
    amp = np.abs(hilbert(ts))
    halt = int(len(ts)/4)
    cl3_tedamp.append(amp)
    cl3_tedmse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

for i in range(len(cl4_sine)):
    ts = cl4_sine.loc[i,:].values
    amp = np.abs(hilbert(ts))
    halt = int(len(ts)/4)
    cl4_sineamp.append(amp)
    cl4_sinemse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

for i in range(len(cl4_ted)):
    ts = cl4_ted.loc[i,:].values
    amp = np.abs(hilbert(ts))
    halt = int(len(ts)/4)
    cl4_tedamp.append(amp)
    cl4_tedmse.append(multiscale_entropy( amp[0:halt], 3, None, 100))

