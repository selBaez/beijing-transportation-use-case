# -*- coding: utf-8 -*-

"""
This module cleans and formats all the Yikatong smart card records.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random, cPickle, re, json
from collections import OrderedDict

import paths, shared

############ --- BEGIN default constants --- ############
MIN_RECORDS_DEFAULT = 0
MODE_DICT_DEFAULT = {       '[(]轨道[.]' : '(R.',        # subway
                            '[(]公交[.]' : '(B.',        # bus
                            '[(]自行车[.]' : '(Z.'}      # bike
############ --- END default constants--- ############

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore column 2 'DATA_LINK'
    data = pd.read_csv(fileName, index_col='ID', usecols= range(2)+range(3,23), parse_dates=[0,8,9])
    print(len(data.index), " records loaded")

    return data

def _clean(data, min_records):
    """
    Remove rows with faulty data
    """
    total = data

    # Remove rows containing NaN
    data, numEmpty = shared._filter(data, data.dropna(), "empty fields")

    # Remove rows with travel detail containing null
    data, numNull = shared._filter(data, data[~data['TRANSFER_DETAIL'].str.contains("null")], "null in transfer description")

    # Remove records with travel time <= 0
    data, numTime = shared._filter(data, data[data['TRAVEL_TIME'] > 0], "travel time <= 0")

    # Remove records with travel distance <= 0
    data, numDistance = shared._filter(data, data[data['TRAVEL_DISTANCE'] > 0], "travel distance <= 0")

    # Remove cards with less than min_records
    data['NUM_TRIPS'] = data.groupby('CARD_CODE')['TRAVEL_DISTANCE'].transform('count')
    data, numMin = shared._filter(data, data[data['NUM_TRIPS'] >= min_records], "users having insufficient associated records")

    if FLAGS.plot_distr == 'True':
        shared._plotPie('faulty', [numEmpty, numNull, numDistance, numTime, len(data.index)],\
         ['Empty fields', 'Incomplete \ntransfer details', 'Negative distance', 'Negative travel time', 'Clean'])

    return data

def _createVocabularies(trips):
    """
    Create LINE and STOPS vocabularies based on the data in the given trips
    """
    # Split the trip into rides and gather components
    lines = set()
    stops = set()
    for index, trip in trips.iteritems():
        if FLAGS.verbose == 'True': print('Trip: ',trip)
        rides = trip.split('->')
        for ride in rides:
            _, lineOrigin, stopOrigin, lineDestination, stopDestination = _extractTripFeatures(ride)
            lines.add(lineOrigin)
            lines.add(lineDestination)
            stops.add(stopOrigin)
            stops.add(stopDestination)

    # Report length of vocabularies formed
    print('     Combined lines found:  ',len(lines))
    print('     Combined stops found:  ',len(stops))

    # Turn into dictionaries
    lines =  dict(zip(lines, map(str,range(len(lines)))))
    stops = dict(zip(stops, map(str,range(len(stops)))))

    # Sort them to have longest patterns replaced first
    lines = OrderedDict(sorted(lines.items(), key=lambda t: len(t[0]), reverse=True))
    stops = OrderedDict(sorted(stops.items(), key=lambda t: len(t[0]), reverse=True))

    # Save as JSON for view and pickle for later use
    with open(paths.VOC_DIR_DEFAULT+'_lines.json', 'w') as fp: json.dump(lines, fp, indent=4, ensure_ascii=False)
    with open(paths.VOC_DIR_DEFAULT+'_stops.json', 'w') as fp: json.dump(stops, fp, indent=4, ensure_ascii=False)

    with open(paths.VOC_DIR_DEFAULT+'_lines.pkl', 'w') as fp: cPickle.dump(lines, fp)
    with open(paths.VOC_DIR_DEFAULT+'_stops.pkl', 'w') as fp: cPickle.dump(stops, fp)

    return lines, stops

def _extractOriginAndDestinationFeatures(trip):
    """
    Extract trip origin and destination features
    """
    rides = trip.split('->')

    firstRide = rides[0]
    lastRide = rides[-1]

    modeOrigin, lineOrigin, stopOrigin, _, _ = _extractTripFeatures(firstRide)
    modeDestination, _, _, lineDestination, stopDestination = _extractTripFeatures(lastRide)

    return modeOrigin, modeDestination, lineOrigin, lineDestination, stopOrigin, stopDestination

def _extractTripFeatures(ride):
    """
    Get MODE, LINE and STOPS and other mode-specific components for the given ride

    BIKE = (bike.STOP-STOP)
    SUBWAY = (subway.LINE_NAME:STOP-LINE_NAME:STOP)
    BUS = (bus.LINE_NAME(DIRECTION-DIRECTION):STOP-LINE_NAME(DIRECTION-DIRECTION):STOP)

    GENERAL = (MODE.[LINE_NAME:]?STOP-[LINE_NAME:]?STOP)[->PATTERN]?
    # MODE                       轨道|公交|自行车        ------ equivalent to ------       subway | bus | bike
    # LINE_NAME  (subway)        5 号线 | NAME 线       ------ equivalent to ------       5 number line | NAME line
    # LINE_NAME  (bus)           944 | 夜 32           ------ equivalent to ------       944 | night 32

    """

    if FLAGS.verbose == 'True': print('Ride details: ', ride)

    # Shared fields across modes
    mode = r'(?P<mode>轨道|公交|自行车)'
    stop_b = r'(?P<stop_b>.+?)'
    stop_a = r'(?P<stop_a>.+?)'

    # Match and classify by mode
    pattern = re.compile(mode)
    matcher = pattern.search(ride)

    # Parse metro
    if matcher.group('mode') == '轨道':
        line_b = r'(?P<line_b>.+?)'
        line_a = r'(?P<line_a>.+?)'

        pattern = re.compile(r'\('+mode+r'[.]'+line_b+r'[:]'+stop_b+r'[-]'+line_a+r'[:]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            return 0, matcher.group('line_b'), matcher.group('stop_b'), matcher.group('line_a'), matcher.group('stop_a')
        else:
            print('Failed at parsing metro ride:', ride)
            return 0, None, None, None, None

    # Parse bus
    elif matcher.group('mode') == '公交':
        line_b = r'(?P<line_b>.+?)'
        direction_b = r'(?P<direction_b>[(].+?[)])'
        line_a = r'(?P<line_a>.+?)'
        direction_a = r'(?P<direction_a>[(].+?[)])'

        pattern = re.compile(r'\('+mode+r'[.]'+line_b+direction_b+r'[:]'+stop_b+r'[-]'+line_a+direction_a+r'[:]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            return 1, matcher.group('line_b'), matcher.group('stop_b'), matcher.group('line_a'), matcher.group('stop_a')
        else:
            print('Failed at parsing bus ride:', ride)
            return 1, None, None, None, None

    # Parse bike
    elif matcher.group('mode') == '自行车':
        pattern = re.compile(r'\('+mode+r'[.]'+stop_b+r'[-]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            return 2, '0', matcher.group('stop_b'), '0', matcher.group('stop_a')
        else:
            print('Failed at parsing bike ride:', ride)
            return 2, None, None, None, None

def _parseTrips(data, modes, createVoc):
    """
    Parse 'TRANSFER_DETAIL' column to get ON/OFF mode, line and stop tokenized information
    """

    # Determine which vocabulary to use
    if createVoc == 'True':
        # Create vocabularies
        print('Creating lines and stops vocabularies')
        lines, stops = _createVocabularies(data['TRANSFER_DETAIL'])
    else:
        # Load existing vocabularies
        with open(paths.VOC_DIR_DEFAULT+'_lines.pkl', 'r') as fp: lines = cPickle.load(fp)
        with open(paths.VOC_DIR_DEFAULT+'_stops.pkl', 'r') as fp: stops = cPickle.load(fp)

    # Reduce dataset size since we are just debugging
    if FLAGS.scriptMode == 'short':
        indices = random.sample(data.index, 5)
        data = data.ix[indices]

    # Retrieve on and off trip details
    data['ON_MODE'], data['OFF_MODE'], data['ON_LINE'], data['OFF_LINE'], data['ON_STOP'], data['OFF_STOP'] =  zip(*data['TRANSFER_DETAIL'].apply(lambda x : _extractOriginAndDestinationFeatures(x)))

    # Replace for clean format
    print('Formating trip')

    # Replace line and stops : vocabularies
    data['ON_LINE'].replace(to_replace=lines, inplace=True)
    data['OFF_LINE'].replace(to_replace=lines, inplace=True)

    data['ON_STOP'].replace(to_replace=stops, inplace=True)
    data['OFF_STOP'].replace(to_replace=stops, inplace=True)


    return data

def _countTransfers(data):
    """
    Re calculate number of transfers, and transfer average time (which is dependent on the previous)
    """
    # Plot number of transfers before and after patching
    if FLAGS.plot_distr == 'True':
        original = data.copy()

    print("Recalculating transfer number and transfer average time")
    data['TRANSFER_NUM'] = data['TRANSFER_DETAIL'].str.count("->")
    data['TRANSFER_TIME_AVG'] = np.where(data['TRANSFER_NUM'] > 0, data['TRANSFER_TIME_SUM'] / data['TRANSFER_NUM'], data['TRANSFER_NUM'])

    if FLAGS.plot_distr == 'True':
        shared._plotDistributionCompare(original['TRANSFER_NUM'], data['TRANSFER_NUM'], 'Number of transfers', labels=['Original', 'Recalculation'], bins='Auto')
        shared._plotDistributionCompare(original['TRANSFER_TIME_AVG'], data['TRANSFER_TIME_AVG'], 'Transfer average time', labels=['Original', 'Recalculation'], bins=20)

    return data

def _to_time_bins(data):
    """
    Start and end time stamps into time bins
    """
    print("Extracting start/end hours")
    data['START_HOUR'] = data['START_TIME'].apply(lambda x : x.hour)
    data['END_HOUR'] = data['END_TIME'].apply(lambda x : x.hour)
    return data

def _weekday(data):
    """
    Extract day of observation and determine if it was a weekday
    """
    print("Extracting day")
    data['DAY'] = data['DATADAY'].apply(lambda x: x.day)

    print("Extracting weekday")
    data['WEEKDAY'] = data['DATADAY'].dt.dayofweek

    return data

def _orderFeatures(data):
    """
    Order features by: General, then spatial boarding, then spatial alighting
    """
    order = ['CARD_CODE', 'DAY', 'WEEKDAY', 'PATH_LINK', 'TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_NUM', 'TRANSFER_TIME_AVG', \
            'TRANSFER_TIME_SUM', 'START_TIME', 'END_TIME', 'START_HOUR', 'END_HOUR', 'TRANSFER_DETAIL', 'NUM_TRIPS', \
            'ON_AREA', 'OFF_AREA', \
            'ON_TRAFFIC', 'OFF_TRAFFIC', 'ON_MIDDLEAREA', 'OFF_MIDDLEAREA', 'ON_BIGAREA', 'OFF_BIGAREA', \
            'ON_RINGROAD', 'OFF_RINGROAD',  \
            'ON_MODE', 'OFF_MODE', 'ON_LINE', 'OFF_LINE', 'ON_STOP', 'OFF_STOP']
    data = data[order]

    return data

def _store(data):
    """
    Store clean data
    """
    data.to_pickle(paths.CLEAN_FILE_DEFAULT+'.pkl')
    data.to_csv(paths.CLEAN_FILE_DEFAULT+'.csv')

def prepare():
    """
    Read raw data, clean it, format it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.RAW_FILE_DEFAULT)

    print("---------------------------- Cleaning -----------------------------")
    data = _clean(data, FLAGS.min_records)

    print("-------------------------- Parsing  trip --------------------------")
    data = _parseTrips(data, MODE_DICT_DEFAULT, FLAGS.create_voc)

    print("------------------ Recalculating transfer number ------------------")
    data = _countTransfers(data)

    print("-------------------- Creating time stamp bins ---------------------")
    data = _to_time_bins(data)

    print("--------------------- Extract day and weekday ---------------------")
    data = _weekday(data)

    print("------------------------- Order  features -------------------------")
    data = _orderFeatures(data)

    print("-------------------------- Storing  data --------------------------")
    _store(data)

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
    prepare()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type = str, default = 'False',
                        help='Display parse trip details.')
    parser.add_argument('--min_records', type = int, default = MIN_RECORDS_DEFAULT,
                        help='Traveler is required to have at least this number of records.')
    parser.add_argument('--create_voc', type = str, default = 'False',
                        help='Create lines/stops vocabularies from given data. If False, previously saved vocabularies will be used')
    parser.add_argument('--plot_distr', type = str, default = 'False',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--scriptMode', type = str, default = 'short',
                        help='Run with long  or short dataset.')
    #TODO: overwrite voc

    FLAGS, unparsed = parser.parse_known_args()
    main(None)
