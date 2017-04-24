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
import re

############ --- BEGIN default constants --- ############
FILE_NAME_DEFAULT = '../Data/Travel chain sample data(50000).csv'
MIN_RECORDS_DEFAULT = 2
TRANSLATE_DICT_DEFAULT = {  '轨道' : 'R',      # subway
                            '公交' : 'B',      # bus
                            '自行车' : 'Z',    # bike
                            '线' : 'Line',
                            '号' : '',
                            '夜' : 'N'}#,       # Night bus
                            #'站' : 'Station ',
                            #'小区' : 'District ',
                            #'机场' : 'Airport ',
                            #'公交场' : 'Bus loop ',
                            #'北' : 'North ',
                            #'南' : 'South ',
                            #'东' : 'East ',
                            #'西' : 'West '}
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
    Remove records from data due to motivation according to Boolean condition
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

    # Remove cards with less than min_records
    data['NUM_TRIPS'] = data.groupby('CARD_CODE')['TRAVEL_DISTANCE'].transform('count')
    data = _filter(data, data[data['NUM_TRIPS'] >= min_records], "users having insufficient associated records")

    return data

def _gatherStations(data):
    """
    Match STATION pattern and create vocabulary
    """
    pattern = re.compile(r'[:](.+?)[-|)]')
    station = pattern.findall(data)
    print('\nStations:')
    print(station[0])
    print(station[1],'\n')

def _extractTripComponents(trip):
    """
    Get MODE, STATIONS and other mode-specific components
    """
    print('Trip details: ',trip)

    mode = r'(?P<mode>轨道|公交|自行车)'

    pattern = re.compile(mode)
    matcher = pattern.search(trip)

    if matcher.group('mode') == '轨道':
        print('\nParsing a metro trip')

        line_b = r'(?P<line_b>[0-9]+号线|.+?线)'
        station_b = r'(?P<station_b>.+?)'
        line_a = r'(?P<line_a>[0-9]+号线|.+?线)'
        station_a = r'(?P<station_a>.+?)'

        pattern = re.compile(r'\('+mode+r'[.]'+line_b+r'[:]'+station_b+r'[-]'+line_a+r'[:]'+station_a+r'[)]')
        matcher = pattern.search(trip)

        print('Mode:                  ',matcher.group('mode'))
        print('Boarding Line:         ',matcher.group('line_b'))
        print('Boarding Station:      ',matcher.group('station_b'))
        print('Alighting Line:        ',matcher.group('line_a'))
        print('Alighting Station:     ',matcher.group('station_a'))

    elif matcher.group('mode') == '公交':
        print('\nParsing a bus trip')

        route_b = r'(?P<route_b>[0-9]+|.+?[0-9]+)'
        direction_b = r'(?P<direction_b>[(].+?[)])'
        station_b = r'(?P<station_b>.+?)'
        route_a = r'(?P<route_a>[0-9]+|.+?[0-9]+)'
        direction_a = r'(?P<direction_a>[(].+?[)])'
        station_a = r'(?P<station_a>.+?)'

        pattern = re.compile(r'\('+mode+r'[.]'+route_b+direction_b+r'[:]'+station_b+r'[-]'+route_a+direction_a+r'[:]'+station_a+r'[)]')
        matcher = pattern.search(trip)

        print('Mode:                  ',matcher.group('mode'))
        print('Boarding Route:        ',matcher.group('route_b'))
        print('Boarding Direction     ',matcher.group('direction_b'))
        print('Boarding Station:      ',matcher.group('station_b'))
        print('Alighting Route:       ',matcher.group('route_a'))
        print('Alighting Direction    ',matcher.group('direction_a'))
        print('Alighting Station:     ',matcher.group('station_a'))

    elif matcher.group('mode') == '自行车':
        print('Parsing a bike trip')

        station_b = r'(?P<station_b>.+?)'
        station_a = r'(?P<station_a>.+?)'

        pattern = re.compile(r'\('+mode+r'[.]'+station_b+r'[-]'+station_a+r'[)]')
        matcher = pattern.search(trip)

        print('Mode:                  ',matcher.group('mode'))
        print('Boarding Station:      ',matcher.group('station_b'))
        print('Alighting Station:     ',matcher.group('station_a'))

def _parseRoute(data, chineseDict):
    """
    Parse 'TRANSFER_DETAIL' column to get route
    BIKE = (bike.STATION-STATION)
    SUBWAY = (subway.LINE_NAME:STATION-LINE_NAME:STATION)
    BUS = (bus.ROUTE_NAME(DIRECTION-DIRECTION):STATION-ROUTE_NAME(DIRECTION-DIRECTION):STATION)

    GENERAL = (MODE.[X_NAME:]? STATION-[X_NAME:]? STATION)[->stuff]?
    # MODE              轨道|公交|自行车
    # LINE_NAME         5 number line | NAME line
    # ROUTE_NAME        944 | night 32
    """

    # TODO Create stops vocabulary
    _gatherStations(data['TRANSFER_DETAIL'][0])

    # TODO Parse with regular expressions
    # for every record in data
    _extractTripComponents(data['TRANSFER_DETAIL'][0])
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

def _countTransfers(data):
    """
    Re calculate number of transfers
    """
    print("Recalculating transfer number")
    data['TRANSFER_NUM'] = data['TRANSFER_DETAIL'].str.count("->")
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

def _standardize(rawData, plot_distr, plot_dir):
    """
    Rescale features to have mean 0 and std 1
    """

    if plot_distr == 'True':
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

    if plot_distr == 'True':
        # Sample 1000 random points
        indices = random.sample(rawData.index, 1000)
        sample = rawData.ix[indices]

        print("Plotting standarized travel time and distance distributions")
        _plotDistribution(sample, plot_dir, 'time_standardized', 'TRAVEL_TIME')
        _plotDistribution(sample, plot_dir, 'distance_standardized', 'TRAVEL_DISTANCE')

        # Plot time vs distance
        fig, ax = plt.subplots()
        sample.plot(x='TRAVEL_DISTANCE',y='TRAVEL_TIME', ax=ax, kind='scatter')
        plt.savefig(plot_dir+'distance_vs_time_standardized.png', format='png')

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
    data = _loadData(FLAGS.file_name)

    print("---------------------------- Cleaning -----------------------------")
    data = _clean(data, FLAGS.min_records)

    print("-------------------------- Parse  route ---------------------------")
    data = _parseRoute(data, TRANSLATE_DICT_DEFAULT)

    print("---------------------- Count transfer number ----------------------")
    data = _countTransfers(data)

    print("-------------------- Creating time stamp bins ---------------------")
    data = _to_time_bins(data)

    #print("------------------------ Extract weekdays -------------------------")
    # TODO http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%204%20-%20Find%20out%20on%20which%20weekday%20people%20bike%20the%20most%20with%20groupby%20and%20aggregate.ipynb

    print("-------------------------- Standardizing --------------------------")
    preprocessedData = _standardize(data, FLAGS.plot_distr, FLAGS.plot_dir)

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
    parser.add_argument('--plot_distr', type = str, default = True,
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--plot_dir', type = str, default = PLOT_DIR_DEFAULT,
                        help='Directory to which save plots.')

    FLAGS, unparsed = parser.parse_known_args()
    main(None)
