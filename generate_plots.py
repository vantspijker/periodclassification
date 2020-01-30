# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 02:27:09 2020

@author: Bas van 't Spijker
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import librosa
import librosa.display
import time
import pickle
import math
import sklearn

from python_speech_features import mfcc, logfbank
from concurrent.futures import ProcessPoolExecutor
from pylab import show, figure, boxplot, axes

                
hop_length = 512
df = pd.read_csv('maestro-v2.0.0.csv')
classes = ['Baroque', 'Classical', 'Romantic', 'Modern']
sampling_rate = 8000
nfft = math.ceil(sampling_rate/40)
step_size = 4
max_plots = step_size * 8

def superplot(signals, spectral_centroids, fbank, mfccs, chromagrams, alignment = 'horizontal'):
    rows = 8
    cols = 4
    if alignment == 'horizontal':
        figure = plt.figure(figsize=(40,20))
        xtextpos = -4
        ytextpos = 0.9
        
    if alignment == 'vertical':
        figure = plt.figure(figsize=(20,28))
        xtextpos = -6
        ytextpos = 0.7
        
    axes = {}
    
    #load variables to plot
    signal_labels, signal_data = [*zip(*signals.items())]
    spectral_labels, spectral_data = [*zip(*spectral_centroids.items())]
    chroma_labels, chroma_data = [*zip(*chromagrams.items())]
    
    #set the spacing between plots
    plt.subplots_adjust(hspace=0.1, wspace =0.15)
    
    plt.tight_layout()
    
    #for formatting the ax labels
    signal_formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1)))
    fbank_formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 100)))
    
    #create labels for every row in the final figure
    letters = ['a1', 'a2', 'b1', 'b2', 'c1', 'c2' ,'d1', 'd2']
    
    for i in range(0, max_plots, step_size):
        # a normalised for loop step for some lists
        normalised_step = int(i/step_size)
        
        #calculate spectral plot values
        spectral_frames = range(len(spectral_data[normalised_step]))
        spectral_t = librosa.frames_to_time(spectral_frames, sr = sampling_rate, hop_length=hop_length)
        #calculate signal plot values
        signal_frames = range(len(signal_data[normalised_step]))
        signal_t = librosa.frames_to_time(signal_frames, sr = sampling_rate*hop_length, hop_length=hop_length)
        
        #plot the first graph
        axes[i] = figure.add_subplot(rows, cols, i+1)
        axes[i].plot(signal_t, signal_data[normalised_step])
        axes[i].plot(spectral_t, normalize(spectral_data[normalised_step]))
        axes[i].get_yaxis().set_visible(True)
        axes[i].xaxis.set_major_formatter(signal_formatter)
        axes[i].grid(True)
        
        #put the letter label in front of the first graph of every row
        axes[i].text(x=xtextpos, y=ytextpos, s=letters[normalised_step], bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center', fontsize=16, weight = 'bold')
        axes[i].set_ylim(-0.7, 1.1)

        #do layout edits to make it look nice
        fix_layout(x_label='Time(s)', y_label='Amplitude', title='Signal and Spectral Rolloff', i=i, axes=axes)
        
        #plot the second graph
        axes[i+1] = figure.add_subplot(rows, cols, i+2)
        axes[i+1].imshow(list(fbank.values())[normalised_step],
                    cmap='hot', interpolation='nearest', aspect='auto', origin='lower')
        axes[i+1].get_xaxis().set_visible(True)
        axes[i+1].get_yaxis().set_visible(True)
        axes[i+1].xaxis.set_major_formatter(fbank_formatter)
        fix_layout(x_label='Time(s)', y_label='F. Coefficients', title='Filterbank', i=i+1, axes=axes)

        #plot the third graph
        axes[i+2] = figure.add_subplot(rows, cols, i+3)
        axes[i+2].imshow(list(mfccs.values())[normalised_step],
                    cmap='hot', interpolation='nearest', aspect='auto', origin='lower')
        axes[i+2].get_xaxis().set_visible(True)
        axes[i+2].get_yaxis().set_visible(True)
        axes[i+2].xaxis.set_major_formatter(fbank_formatter)
        fix_layout(x_label='Time(s)', y_label='MFCCs', title='Mel-Cepstrum Coefficients', i=i+2, axes=axes)
        
        #plot the fourth graph
        axes[i+3] = figure.add_subplot(rows, cols, i+4)
        librosa.display.specshow(chroma_data[normalised_step], sr=sampling_rate, y_axis='chroma', x_axis='time', hop_length=hop_length, cmap='coolwarm') 
        axes[i+3].xaxis.set_major_formatter(signal_formatter)
        fix_layout(x_label='Time(s)', y_label='Key', title='Chromagram', i=i+3, axes=axes)
    
    #crop and cut outer whitespace in the saved .pdf figure (or any other format)
    plt.tight_layout()
    #save the figure using the desired file format
    plt.savefig('superplot_' + alignment +'.pdf')


def fix_layout(x_label, y_label, title, i, axes):
    #set the title only for the first plots (on the top of the figure)
    if i < step_size:
        axes[i].set_title(title, size=20)
    #disable x tick labels for everything but the bottom plots
    if i < max_plots-step_size:
        for tic in axes[i].xaxis.get_major_ticks():
                tic.label1.set_visible(False)
                tic.label2.set_visible(False)
        axes[i].set_xlabel('')
    #always set the y label
    axes[i].set_ylabel(y_label, labelpad=-2, fontsize=14)
    #set the x labels for the last 4 plots only at the bottom fo the figure
    if i >= max_plots - step_size:
        axes[i].set_xlabel(x_label, fontsize = 14)
        axes[i].get_xaxis().set_visible(True)

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#plot eight example signals
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,5))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    fig.suptitle('Time Series', size=16)
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // sampling_rate)))
    i = 0
    for y in range(4):
        for x in range(2):
            axes[x,y].set_title(list(signals.keys())[i], fontsize = 17)
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].xaxis.set_major_formatter(formatter)
            axes[x,y].grid(True)
            if y == 0:
                axes[x,y].set_ylabel("Amplitude", fontsize=16)
            if x==1:
                axes[x,y].set_xlabel("Time", fontsize=16)
            i += 1
    plt.savefig('signals.pdf')
    show()

#plot eight example signals but with spectral centroids
def plot_signals_with_centroids(signals, spectral_centroids):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,5))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    fig.suptitle('Time Series', size=16)
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1)))
    labels, data = [*zip(*spectral_centroids.items())]
    labels2, data2 = [*zip(*signals.items())]   
    i = 0
    for y in range(4):
        for x in range(2):
            frames = range(len(data[i]))
            t = librosa.frames_to_time(frames, sr = sampling_rate, hop_length=hop_length)
            frames2 = range(len(data2[i]))
            t2 = librosa.frames_to_time(frames2, sr = sampling_rate*hop_length, hop_length=hop_length)
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(t, normalize(data[i]), color='r')
            axes[x,y].plot(t2, data2[i])
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].xaxis.set_major_formatter(formatter)
            axes[x,y].grid(True)
            i += 1
    plt.savefig('signals_with_centroids.pdf')
    show()

# plot eight boxplots           
def plot_boxplots(signals):
    boxwidth = 0.9
    figure(figsize=(10,5))
    ax = axes()
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_xlabel('Period', fontsize=14)
    labels, data = [*zip(*signals.items())]
    
    boxplot([data[0], data[1]], positions = [1, 2], widths = boxwidth, showfliers=False)
    boxplot([data[2], data[3]], positions = [3.5, 4.5], widths = boxwidth, showfliers=False)
    boxplot([data[4], data[5]], positions = [6, 7], widths = boxwidth, showfliers=False)
    boxplot([data[6], data[7]], positions = [8.5, 9.5], widths = boxwidth, showfliers=False)
    
    #set the labels (hardcoded)
    ax.set_xticklabels(['a1', 'Baroque', 'a2', 'b1', 'Classical', 'b2', 'c1', 'Romantic', 'c2', 'd1', 'Modern', 'd2'])
    ax.set_xticks([1, 1.5, 2, 3.5, 4, 4.5, 6, 6.5, 7, 8.5, 9, 9.5])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.grid(True, axis='y')


    plt.savefig('boxplots.pdf')
    show()

#plot eight example chromagrams
def plot_chromagrams(chromagrams):
    plt.figure(figsize=(20,5))
    labels, data = [*zip(*chromagrams.items())]
    i = 0
    for y in range(4):
        for x in range(2):
            plt.subplot(2, 4, i+1)
            librosa.display.specshow(data[i], x_axis='time', y_axis='chroma', sr=sampling_rate, hop_length=hop_length, cmap='coolwarm') 
            i += 1

#plot eight example fast fourier transforms
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,5))
    plt.subplots_adjust(hspace=0.4, wspace = 0.1)
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for y in range(4):
        for x in range(2):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i], fontsize=17)
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].grid(True)
            if y == 0:
                axes[x,y].set_ylabel("Magnitude", fontsize=16)
            if x==1:
                axes[x,y].set_xlabel("Frequency (Hz)", fontsize=16)
            i += 1
    plt.savefig('ffts.pdf')

#plot eight example filterbank value graphs
def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=True, figsize=(20,5))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 100)))
    for y in range(4):
        for x in range(2):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest', aspect='auto', origin='lower')
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].xaxis.set_major_formatter(formatter)
            i += 1
    plt.savefig('fbanks.pdf')

#plot eight example Mel-Cepstrum Coefficient alue grapsh
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False,
                             sharey=False, figsize=(20,5))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    formatter = ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 100)))
    for y in range(4):
        for x in range(2):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest', aspect='auto', origin='lower')
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(True)
            axes[x,y].xaxis.set_major_formatter(formatter)

            i += 1
    plt.savefig('mfccs.pdf')
    
#plot the dataset period(/class) distribution
def plot_period_distribution():
    #read csv and set audio filename to index
    df.set_index('audio_filename', inplace=True)
    
    #create a dataframe of unique periods, count the amount of pieces per period,
    #and create single list with the count numbers to input in fig()
    title = 'Period Distribution'
    class_dist = df.groupby('period').period.agg('count').to_frame('count')
    piececountlist = class_dist['count'].values.tolist()
    
    #create piechart with distribution of pieces per period
    fig, ax = plt.subplots()
    ax.set_title(title, y=1.08)
    ax.pie(piececountlist, labels=class_dist.index, autopct='%1.1f%%',
           shadow=False, 
           startangle=90, textprops={'fontsize': 16})
    ax.axis('equal')
    
    # reset the index (no longer audio filename)
    df.reset_index(inplace=True)
    plt.savefig('period_distribution_validation.pdf')

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

def calc_chromagram(y, rate):
    chromagram = librosa.feature.chroma_stft(y, sr=rate, hop_length=hop_length)
    return chromagram

def calc_spectral_centroid(y, rate):
    spectral_centroid = librosa.feature.spectral_centroid(y, sr=rate)[0]
    return spectral_centroid

#calculate the values required for the plots
def calculate_values():
    signals = {}
    short_signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    chromagrams = {}  
    spectral_centroids = {}
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        for info, signal, rate, info2, signal2, rate2 in executor.map(load_file_data, classes):
            signals[info] = signal
            short_signals[info] = signal[rate*30:rate*60]
            
            spectral_centroids[info] = calc_spectral_centroid(signal[rate*30:rate*60], rate)
            
            fft[info] = calc_fft(signal, rate)
            
            bank = logfbank(signal[rate*30:rate*60], rate, nfilt=26, nfft=nfft).T
            fbank[info] = bank
            
            mel = mfcc(signal[rate*30:rate*60], rate, numcep=13, nfilt=26, nfft=nfft).T
            mfccs[info] = mel
            
            chromagrams[info] = calc_chromagram(signal[rate*30:rate*60], rate)
            
            signals[info2] = signal2
            short_signals[info2] = signal2[rate2*30:rate2*60]
            
            spectral_centroids[info2] = calc_spectral_centroid(signal2[rate2*30:rate2*60], rate2)
            
            fft[info2] = calc_fft(signal2, rate2)
            
            bank = logfbank(signal2[rate2*30:rate2*60], rate2, nfilt=26, nfft=nfft).T
            fbank[info2] = bank
            
            mel = mfcc(signal2[rate2*30:rate2*60], rate2, numcep=13, nfilt=26, nfft=nfft).T
            mfccs[info2] = mel
            
            chromagrams[info2] = calc_chromagram(signal2[rate2*30:rate2*60], rate2)
            
    return signals, short_signals, fft, fbank, mfccs, chromagrams, spectral_centroids

#return data from two different composers of the given period
def load_file_data(period):
    try:
        wav_file = df[df.period == period].iloc[0,0]
        composer = df[df.period == period].iloc[0,1]
        signal, rate = librosa.load(wav_file, sr=sampling_rate)
        info = period + ', ' + composer
        
        wav_file2 = df[(df.period == period) & (df.canonical_composer != composer)].iloc[0,0]
        composer2 = df[(df.period == period) & (df.canonical_composer != composer)].iloc[0,1]
        info2 = period + ', ' + composer2
        signal2, rate2 = librosa.load(wav_file2, sr=sampling_rate)
        return info, signal, rate, info2, signal2, rate2
    except:
        print('error with item')


#the audio_filename needs to be moved to the front for the code to execute properly
df.set_index('audio_filename', inplace=True)
df.reset_index(inplace=True)

def load_data():
    #try to load data if available to speed up calculating plots
    if os.path.isfile('data.pickle'):
        with open('data.pickle', 'rb') as f:
            signals, short_signals, fft, fbank, mfccs, chromagrams, spectral_centroids= pickle.load(f)
    else:
        signals, short_signals, fft, fbank, mfccs, chromagrams, spectral_centroids = calculate_values()
        
        #save generated data the first time
        with open('data.pickle', 'wb') as f:
            pickle.dump([signals, short_signals, fft, fbank, mfccs, chromagrams, spectral_centroids], f)
    return signals, short_signals, fft, fbank, mfccs, chromagrams, spectral_centroids
    
def main():   
    signals, short_signals, fft, fbank, mfccs, chromagrams, spectral_centroids = load_data()
    alignment = 'horizontal'
    
    plot_period_distribution()
            
    plot_signals(signals)
        
    plot_signals(short_signals)
    
    plot_signals_with_centroids(short_signals, spectral_centroids)
 
    plot_boxplots(signals)
    
    plot_fft(fft)
   
    plot_fbank(fbank)

    plot_mfccs(mfccs)

    plot_chromagrams(chromagrams)
 
    superplot(short_signals, spectral_centroids, fbank, mfccs, chromagrams, alignment)

if __name__ == '__main__':
    main()