""" Quick plot for FLAT_QA
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def aux_main():
    # Plotting flat_qa results
    fnm = 'flatqa_20180815t1015.csv'
    df = pd.read_csv(fnm)
    df.columns = df.columns.map(str.lower)
    
    # Call the plotting
    if False:
        plot_flatqa(df)

    # Write out the selected expnums
    c1 = df['nite'] >= 20180913
    c2 = df['nite'] <= 20180923
    c3 = df['rms'] < 0.02
    c4 = (df['band'] == 'u') | (df['band'] == 'Y') | (df['band'] == 'VR') 
    # c4 = (df['band'] == 'g') | (df['band'] =='r') | (df['band'] == 'i') | (df['band'] == 'z')
    dfaux = df.loc[c1 & c2 & c3 & c4]
    dfaux['expnum'].to_csv('expnum_uYVR_20180913t0923.csv', index=False, 
                           header=False)
    h = dfaux.groupby(['band']).agg('count')
    print(h)


def plot_flatqa(df):
    # Add a new date-column
    df['nite_dt'] = pd.to_datetime(df['nite'], format='%Y%m%d')
    df['ratio'] = df['factor'] / df['rms']

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
   
    kw = {
        'linewidth': 0.5,
        'edgecolor': 'w',
        'alpha': 0.5,
        'sizes': 'exptime',
    }

    plt.sca(ax[0])
    ts = sns.scatterplot(x='nite_dt', y='factor', hue='band',
                         data=df, **kw)
    
    plt.sca(ax[1])
    ts = sns.scatterplot(x='nite_dt', y='rms', hue='band',
                         data=df, **kw)

    plt.sca(ax[2])
    ts = sns.scatterplot(x='nite_dt', y='ratio', hue='band',
                         data=df, **kw)

    # Rotate and align the tick labels
    fig.autofmt_xdate()
    # Date string
    for axis in ax: 
        axis.fmt_xdata = mdates.DateFormatter('%Y%m%d')
        axis.set_xlim([min(df['nite_dt']), max(df['nite_dt'])])
        axis.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    
    plt.subplots_adjust(right=0.8)
    plt.show()


if __name__ == '__main__':
    aux_main()
