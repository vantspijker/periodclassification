# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:10:18 2020

@author: Bas van 't Spijker
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import pickle
import seaborn as sns

from tqdm import tqdm
from scipy.io import wavfile

from concurrent.futures import ProcessPoolExecutor
from keras.layers import LeakyReLU
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from sklearn import svm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, log_loss,  balanced_accuracy_score
from python_speech_features import mfcc

from cfg import Config


#set some keras/gpu settings
configure = tf.ConfigProto()
configure.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=configure)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#read in the csv with the data information
df = pd.read_csv('maestro-v2.0.0.csv')
df['audio_filename'] = df['audio_filename'].str[5:]
df.set_index('audio_filename', inplace=True)

#manually set the classes
classes = ['Baroque', 'Classical', 'Romantic', 'Modern']
classifier = 'period'
class_dist = df.groupby(classifier).period.agg('count').to_frame('countt')

leakyrelu = LeakyReLU(alpha=0.3)
activation_layer = leakyrelu

figsize = 8


#change this for setting machine learning configuration: 'svm', 'time' or 'conv' for mode, and 'chroma' or 'mfcc' for feature
config = Config(mode='svm', feature='chroma')
#for convolutional networks about 6 epochs works best. for LSTMS it is about 9.
epochs = 6

#check if data already exists and open it if yes
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model with {} features'.format(config.mode, config.feature))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

#check if file folder of 30 second samples has been built and if not, build one. this function takes a very long time
def check_files():
    if len(os.listdir('clean')) == 0:
            with ProcessPoolExecutor(max_workers= 8) as executor:
                for filename, signal, rate in executor.map(read_file_operation, df.audio_filename):
                    year = df.loc[df['audio_filename'] == filename]['year']   
                    wavfile.write(filename='clean/' + str(filename).lstrip(str(year)+ '/'), rate=rate, data=signal[rate*30:rate*60])
                    
def read_file_operation(filename):
    signal, rate = librosa.load(filename, sr=config.rate)
    return filename, signal, rate
    
#calculate the average mean and standard deviation of the dataset
def calc_mean_and_stdev():
    baroque_means = []
    classical_means = []
    romantic_means = []
    modern_means = []
    baroque_stdevs = []
    classical_stdevs = []
    romantic_stdevs = []
    modern_stdevs = []
    for filename in tqdm(df.index):
        if df.at[filename, classifier] in classes:
            file = filename
            rate, wav = wavfile.read('clean/'+file)
            signal = np.array(wav)
            mean = np.mean(signal)
            stdev = np.std(signal)
            if df.at[filename, 'period'] == 'Baroque':
                baroque_means.append(mean)
                baroque_stdevs.append(stdev)
            elif df.at[filename, 'period'] == 'Classical':
                classical_means.append(mean)
                classical_stdevs.append(stdev)
            elif df.at[filename, 'period'] == 'Romantic':
                romantic_means.append(mean)
                romantic_stdevs.append(stdev)
            elif df.at[filename, 'period'] == 'Modern':
                modern_means.append(mean)
                modern_stdevs.append(stdev)
                
    baroque_means = np.array(baroque_means)
    classical_means = np.array(classical_means)
    romantic_means = np.array(romantic_means)
    modern_means = np.array(modern_means)
    baroque_stdevs = np.array(baroque_stdevs)
    classical_stdevs = np.array(classical_stdevs)
    romantic_stdevs = np.array(romantic_stdevs)
    modern_stdevs = np.array(modern_stdevs)
    
    baroque_means_average = np.mean(baroque_means)
    classical_means_average = np.mean(classical_means)
    romantic_means_average = np.mean(romantic_means)
    modern_means_average = np.mean(modern_means)
    baroque_stdevs_average = np.mean(baroque_stdevs)
    classical_stdevs_average = np.mean(classical_stdevs)
    romantic_stdevs_average = np.mean(romantic_stdevs)
    modern_stdevs_average = np.mean(modern_stdevs)
    
    print('')
    print("Baroque means_average: " + str(baroque_means_average))
    print("Baroque stdevs average: " + str(baroque_stdevs_average))
    print("Classical means average: " + str(classical_means_average))
    print("Classical stdevs average: " + str(classical_stdevs_average))
    print("Romantic means average: " + str(romantic_means_average))
    print("Romantic stdevs average: " + str(romantic_stdevs_average))
    print("Modern means average: " + str(modern_means_average))
    print("Modern stdevs average: " + str(modern_stdevs_average))
                
#this function builds and formats the data that feeds the neural networks
def build_rand_feat(feature):
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1], tmp.data[2], tmp.data[3], tmp.data[4], tmp.data[5]
    train_data = []
    validate_data = []
    test_data = []
    train_labels = []
    validate_labels = []
    test_labels = []
    _min, _max = float('inf'), -float('inf')
    
    #the flip variable is a hacky way of reducing the Romantic amount of pieces by 2/3, by skipping every two out of 3 pieces
    flip = 0
    
    for filename in tqdm(df.index):
        new_minmax = False
        X_sample = None
        if df.at[filename, classifier] in classes:
            if ((df.at[filename, 'period'] == 'Romantic') & flip==0) | (df.at[filename, 'period'] != 'Romantic'):
                file = filename
                rate, wav = wavfile.read('clean/'+file)
                label = df.at[file, classifier]
                if feature == 'mfcc':
                    X_sample = mfcc(wav, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
                elif feature == 'chroma':
                    X_sample = librosa.feature.chroma_stft(wav, sr=rate, hop_length=512)
                else:
                    break
                if df.at[filename, 'split'] == 'train':
                    train_data.append(X_sample)
                    train_labels.append(classes.index(label))
                    new_minmax = True
                elif df.at[filename, 'split'] == 'validation':
                    validate_data.append(X_sample)
                    validate_labels.append(classes.index(label))
                    new_minmax = True
                elif df.at[filename, 'split'] == 'test':
                    test_data.append(X_sample)
                    test_labels.append(classes.index(label))
                    new_minmax = True
                flip = flip + 1
            else:
                flip = (flip+1) % 3
        if new_minmax:
            _min = min(np.amin(X_sample), _min)
            _max = max(np.amax(X_sample), _max)
            
    train_data = np.array(train_data) 
    validate_data = np.array(validate_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    validate_labels = np.array(validate_labels)
    test_labels = np.array(test_labels)
    train_data = (train_data - _min) / (_max - _min)
    validate_data = (validate_data - _min) / (_max - _min)
    test_data = (test_data - _min) / (_max - _min)
    
    if config.mode == 'conv':
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
        validate_data = validate_data.reshape(validate_data.shape[0], validate_data.shape[1], validate_data.shape[2], 1)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
    elif config.mode == 'time':
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
        validate_data = validate_data.reshape(validate_data.shape[0], validate_data.shape[1], validate_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2])
    train_labels = to_categorical(train_labels, num_classes = len(class_dist.index))
    validate_labels = to_categorical(validate_labels, num_classes = len(class_dist.index))
    test_labels = to_categorical(test_labels, num_classes = len(class_dist.index))

    config.data = (train_data, train_labels, validate_data, validate_labels, test_data, test_labels)
    
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    return train_data, train_labels, validate_data, validate_labels, test_data, test_labels

#Build the convolutional model. This setup is based on a gpu with about 6GB vram. 
def get_conv_model(input_shape):
    model=Sequential()
    model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', input_shape=input_shape))
    model.add(activation_layer)
    model.add(Conv2D(32, (3, 3), strides=(1,1), padding='same'))
    model.add(activation_layer)
    model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same'))
    model.add(activation_layer)
    model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
    model.add(activation_layer)
    if config.feature == 'chroma':
        model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same'))
        model.add(activation_layer)
        model.add(Conv2D(512, (3, 3), strides=(1,1), padding='same'))
        model.add(activation_layer)
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    if config.feature == 'mfcc':
        model.add(Dense(64))
    if config.feature == 'chroma':
        model.add(Dense(128))
    model.add(activation_layer)
    model.add(Dense(len(class_dist.index), activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

#Build the LSTM model. 
def get_recurrent_model(input_shape):
    #shape of data for RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation=None)))
    model.add(activation_layer)
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

#show confusion matrix
def show_conf_matrix(valdata, vallabels_flat, model, name):
    if config.mode == 'svm':
        y_pred = valdata
    else:
        y_pred=model.predict_classes(valdata)
    con_mat = tf.math.confusion_matrix(labels=vallabels_flat, predictions=y_pred).eval(session=sess)
    con_mat_df = pd.DataFrame(con_mat,
                         index = classes, 
                         columns = classes)
    plt.figure(figsize=(figsize, 0.75 * figsize))
    sns.set(font_scale=1.8)
    ax = sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    #fixing layout 
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    plt.ylabel('True Period', fontsize =32)
    plt.xlabel('Predicted Period', labelpad = 10, fontsize =32)
    plt.savefig(name+'.pdf', bbox_inches = "tight")
    plt.show()

#plot the data distribution for the test data
def plot_class_distribution():
    train_data = df[df.split == 'test']
    class_dist = train_data.groupby('period').period.agg('count').to_frame('count')
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution Test Data', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()

#execute the training plan based on the config settings
def execute_machine_learning():
    traindata, trainlabels, valdata, vallabels , testdata, testlabels = build_rand_feat(config.feature)
    trainlabels_flat = np.argmax(trainlabels, axis=1)
    vallabels_flat = np.argmax(vallabels, axis=1)
    testlabels_flat = np.argmax(testlabels, axis=1)
    
    if config.mode == 'conv':
        input_shape = (traindata.shape[1], traindata.shape[2], 1)
        model = get_conv_model(input_shape)
        train_model(model, traindata, trainlabels, trainlabels_flat, valdata, vallabels, vallabels_flat, testdata, testlabels, testlabels_flat)
        
    elif config.mode == 'time':
        input_shape = (traindata.shape[1], traindata.shape[2])
        model = get_recurrent_model(input_shape)
        train_model(model, traindata, trainlabels, trainlabels_flat, valdata, vallabels, vallabels_flat, testdata, testlabels, testlabels_flat)
        
    elif config.mode == 'svm':
        nsamples, nx, ny = traindata.shape
        traindata = traindata.reshape((nsamples,nx*ny))
        nsamples, nx, ny = testdata.shape
        testdata = testdata.reshape((nsamples,nx*ny))
        clf = svm.SVC(kernel='rbf', class_weight='balanced')
        clf.fit(traindata, trainlabels_flat)
        predicted_trainlabels = clf.predict(traindata)
        predicted_testlabels = clf.predict(testdata)
        show_conf_matrix(predicted_testlabels, testlabels_flat, clf, 'svm_confusion_matrix_' + config.feature)
        print("Accuracy of predicting traindata:", accuracy_score(trainlabels_flat, predicted_trainlabels))
        print("Accuracy of predicting testdata:", accuracy_score(testlabels_flat, predicted_testlabels))
        print('')
        print("Balanced accuracy of predicting traindata:", balanced_accuracy_score(trainlabels_flat, predicted_trainlabels))
        print("Balanced accuracy of predicting testdata:", balanced_accuracy_score(testlabels_flat, predicted_testlabels))
    
#this function is for the convolution and LSTM model
def train_model(model, traindata, trainlabels, trainlabels_flat, valdata, vallabels, vallabels_flat, testdata, testlabels, testlabels_flat):
    class_weight = compute_class_weight('balanced',
                                        np.unique(trainlabels_flat),
                                        trainlabels_flat)
    model.fit(traindata, trainlabels, epochs = epochs, batch_size=16, shuffle=True, class_weight=class_weight, validation_data=(valdata, vallabels))
    model.save(config.model_path)
    # model = load_model(config.model_path)
    
    predicted_trainlabels = model.predict_classes(traindata)
    predicted_validationlabels = model.predict_classes(valdata)
    predicted_testlabels = model.predict_classes(testdata)
    
    pred_prob_traindata = model.predict_proba(traindata)
    pred_prob_validationdata = model.predict_proba(valdata)
    pred_prob_testdata = model.predict_proba(testdata)
    
    print('')
    print("Accuracy of train set:", accuracy_score(trainlabels_flat, predicted_trainlabels))
    print("Accuracy of validation set:", accuracy_score(vallabels_flat, predicted_validationlabels))
    print("Accuracy of test set:", accuracy_score(testlabels_flat, predicted_testlabels))
    print('')
    print("Log_loss of train set:", log_loss(trainlabels_flat, pred_prob_traindata))
    print("Log_loss of validation set:", log_loss(vallabels_flat, pred_prob_validationdata))
    print("Log_loss of test set:", log_loss(testlabels_flat, pred_prob_testdata))
    print('')
    print("Balanced accuracy of validation set:", balanced_accuracy_score(vallabels_flat, predicted_validationlabels))
    print("Balanced accuracy of test set:", balanced_accuracy_score(testlabels_flat, predicted_testlabels))
    print('')
    print("Validation set confusion matrix:")
    show_conf_matrix(valdata, vallabels_flat, model, config.mode + '_' + config.feature + '_validation')
    print('')
    print("Test set confusion matrix:")
    show_conf_matrix(testdata, testlabels_flat, model, config.mode + '_' + config.feature + '_test')

#choose the functions to execute
def main():
    calc_mean_and_stdev()
    plot_class_distribution()
    execute_machine_learning()

if __name__ == '__main__':
    main()