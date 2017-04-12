# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
import cPickle

############ --- BEGIN default constants --- ############
FILE_NAME_DEFAULT = '../Data/Travel chain sample data(50000).csv'
MIN_RECORDS_DEFAULT = 2
PLOT_DISTR_DEFAULT = True
TRANSLATE_DICT_DEFAULT = {  '轨道' : 'R',      # subway
                            '公交' : 'B',      # bus
                            '自行车' : 'Z',    # bike
                            '线' : 'Line',
                            '号' : '',
                            '夜' : 'N',
                            '站' : 'Station ',
                            '小区' : 'District ',
                            '机场' : 'Airport ',
                            '公交场' : 'Bus loop ',
                            '北' : 'North ',
                            '南' : 'South ',
                            '东' : 'East ',
                            '西' : 'West '}
############ --- END default constants--- ############

############ --- BEGIN default directories --- ############
PLOT_DIR_DEFAULT = './Plots/'
############ --- END default directories--- ############

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore column 2 'DATA_LINK'
    data = pd.read_csv(fileName, index_col='ID', usecols= range(2)+range(3,23), parse_dates=[0,8,9])
    print(len(data.index), " records loaded")

    return data

def _filter(data, condition, motivation):
    """
    Remove records from data due to motivation accordint to Boolean condition
    """
    recordsBefore = len(data.index)
    data = condition
    recordsLeft = len(data.index)
    recordsRemoved = recordsBefore - recordsLeft
    print(recordsRemoved, " records removed due to ", motivation, ", ", recordsLeft, " records left")

    return data

def _clean(data, min_records):
    """
    Remove rows with faulty data
    """
    # Remove rows containing NaN
    data = _filter(data, data.dropna(), "empty fields")

    # Remove rows with travel detail containing null
    data = _filter(data, data[~data['TRANSFER_DETAIL'].str.contains("null")], "null in transfer description")

    # Remove records with travel time <= 0
    data = _filter(data, data[data['TRAVEL_TIME'] > 0], "travel time <= 0")

    # Remove records with travel distance <= 0
    data = _filter(data, data[data['TRAVEL_DISTANCE'] > 0], "travel distance <= 0")

    # TODO Remove rows with strange transfers

    # Remove cards with less than min_records
    data['NUM_TRIPS'] = data.groupby('CARD_CODE')['TRAVEL_DISTANCE'].transform('count')
    data = _filter(data, data[data['NUM_TRIPS'] >= min_records], "users having insufficient associated records")

    return data

def _parseRoute(data, chineseDict):
    """
    Parse 'TRANSFER_DETAIL' column to get route
    BIKE = (bike.STATION-STATION)
    SUBWAY = (subway.LINE_NAME:STATION-LINE_NAME:STATION)
    BUS = (bus.ROUTE_NAME:)

    GENERAL = (MODE.[LINE_NAME:]? STATION-[LINE_NAME:]? STATION-)
    # LINE_NAME example        5 number line or NAME line
    # ROUTE_NAME example       944 or night 32 (DIRECTION - DIRECTION)
    """

    # TODO Create stops vocabulary

    # TODO Parse with regular expressions
    # for every record in data
        # replace mode : dictionary
        # replace line/route : dictionary
        # replace stops : vocabulary

    # Translate basic keywords
    print("Replacing keywords from Chinese to English")
    for key in reversed(sorted(chineseDict.keys())):
        data['TRANSFER_DETAIL'] = data['TRANSFER_DETAIL'].str.replace(key,chineseDict[key])

    # TODO: Extract stop number (if available) and line route as S3->B56
    #print("Simplifying route")

    return data

def _to_time_bins(data):
    """
    Start and end time stamps into time bins
    """
    print("Extracting start/end hours")
    data['START_HOUR'] = data['START_TIME'].apply(lambda x : x.hour)
    data['END_HOUR'] = data['END_TIME'].apply(lambda x : x.hour)
    return data

def _plotDistribution(sample, plot_dir, variable_name, column_name):
    """
    Plot variables distribution
    """
    #TODO make plots nice with titles, etc

    # Plot card code vs (sorted) variable
    #fig, ax = plt.subplots()
    #sample.sort_values(by=column_name).plot.bar(x='CARD_CODE', y=column_name, ax=ax)
    #plt.savefig(plot_dir+variable_name+'.png', format='png')

    # Plot variable frequency histogram
    fig, ax = plt.subplots()
    sample[column_name].plot.hist(ax=ax, bins=20)
    plt.savefig(plot_dir+variable_name+'_hist.png', format='png')

    # Plot variable box
    fig, ax = plt.subplots()
    sample[column_name].plot.box(ax=ax)
    plt.savefig(plot_dir+variable_name+'_box.png', format='png')

def _standarize(rawData, plot_distr, plot_dir):
    """
    Rescale features to have mean 0 and std 1
    """

    if plot_distr:
        # Sample 1000 random points
        indices = random.sample(rawData.index, 1000)
        sample = rawData.ix[indices]

        print("Plotting original travel time and distance distributions")
        _plotDistribution(sample, plot_dir, 'time', 'TRAVEL_TIME')
        _plotDistribution(sample, plot_dir, 'distance', 'TRAVEL_DISTANCE')
        _plotDistribution(sample, plot_dir, 'transfer_time', 'TRANSFER_TIME_SUM')
        _plotDistribution(sample, plot_dir, 'num_transfers', 'TRANSFER_NUM')
        _plotDistribution(sample, plot_dir, 'start_hour', 'START_HOUR')
        _plotDistribution(sample, plot_dir, 'end_hour', 'END_HOUR')

        # Plot time vs distance
        fig, ax = plt.subplots()
        sample.plot(x='TRAVEL_DISTANCE',y='TRAVEL_TIME', ax=ax, kind='scatter')
        plt.savefig(plot_dir+'distance_vs_time.png', format='png')

    # TODO: only fit and transform to train data, and transform test data
    print("Standarize travel time and distance")
    scaler = StandardScaler()
    rawData[['TRAVEL_TIME', 'TRAVEL_DISTANCE']] = scaler.fit_transform(rawData[['TRAVEL_TIME', 'TRAVEL_DISTANCE']])

    if plot_distr:
        # Sample 1000 random points
        indices = random.sample(rawData.index, 1000)
        sample = rawData.ix[indices]

        print("Plotting standarized travel time and distance distributions")
        _plotDistribution(sample, plot_dir, 'time_standarized', 'TRAVEL_TIME')
        _plotDistribution(sample, plot_dir, 'distance_standarized', 'TRAVEL_DISTANCE')

        # Plot time vs distance
        fig, ax = plt.subplots()
        sample.plot(x='TRAVEL_DISTANCE',y='TRAVEL_TIME', ax=ax, kind='scatter')
        plt.savefig(plot_dir+'distance_vs_time_standarized.png', format='png')

    return rawData

def _store(preprocessedData):
    """
    Store data for use in model
    """
    pass

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    fullData = _loadData(FLAGS.file_name)

    print("---------------------------- Cleaning -----------------------------")
    cleanData = _clean(fullData, FLAGS.min_records)

    ########################### Feature  engineering ###########################

    print("-------------------------- Parse  route ---------------------------")
    withRouteData = _parseRoute(cleanData, TRANSLATE_DICT_DEFAULT)

    print("-------------------- Creating time stamp bins ---------------------")
    withBinsData = _to_time_bins(withRouteData)

    #print("------------------------ Extract weekdays -------------------------")
    # TODO http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%204%20-%20Find%20out%20on%20which%20weekday%20people%20bike%20the%20most%20with%20groupby%20and%20aggregate.ipynb

    print("-------------------------- Standarizing ---------------------------")
    preprocessedData = _standarize(withBinsData, FLAGS.plot_distr, FLAGS.plot_dir)

    #print("--------------------------- Store  data ---------------------------")

def print_flags():
    """
    Print all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    """
    Main function
    """
    print_flags()
    #TODO Make directories if they do not exists yet
    preprocess()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type = str, default = FILE_NAME_DEFAULT,
                        help='Data file to load.')
    parser.add_argument('--min_records', type = int, default = MIN_RECORDS_DEFAULT,
                        help='Traveler is required to have at least this number of records.')
    parser.add_argument('--plot_distr', type = bool, default = PLOT_DISTR_DEFAULT,
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--plot_dir', type = str, default = PLOT_DIR_DEFAULT,
                        help='Directory to which save plots.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)
