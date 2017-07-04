# -*- coding: utf-8 -*-

"""
This module cleans and formats the Yikatong smart card records.
"""
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random, cPickle, re, json, csv
# from memory_profiler import profile
from datetime import datetime
from multiprocessing import Pool, cpu_count

import paths, shared

############ --- BEGIN default constants --- ############
MODES_DEFAULT = [u'轨道', u'公交', u'自行车']      # subway, bus, bike   # REAL DATA
# MODES_DEFAULT = ['轨道', '公交', '自行车']      # subway, bus, bike      # TEST
############ --- END default constants--- ############

def _loadData(fileName):
    """
    Load csv data on pandas
    """
    # Ignore column 2 'DATA_LINK'
    data = pd.read_csv(fileName, index_col='ID', usecols= range(2)+range(3,23), parse_dates=[0,8,9], encoding='cp936')    # REAL DATA
    # data = pd.read_csv(fileName, index_col='ID', usecols= range(2)+range(3,23), parse_dates=[0,8,9])                        # TEST
    print("{} records loaded".format(len(data.index)))

    return data

def _sampleByCode(data):
    """
    Search for codes in sample codes
    """
    # Load codes sets
    sampleCodes = np.loadtxt(paths.LABELS_DIR_DEFAULT+'sampleCardCodes.txt')

    # Eliminate codes taht are not in sample codes
    data, _ = shared._filter(data, data[data['CARD_CODE'].isin(sampleCodes)], "not desired card codes")

    return data

def _sampleByTrip(data):
    """
    Select 10,000 random records, or as much as possible
    """
    sampleSize = 10000 if len(data.index) > 10000 else len(data.index)
    indices = random.sample(data.index, sampleSize)
    data = data.ix[indices]

    return data

def _clean(data):
    """
    Remove rows with faulty data
    """
    total = len(data.index)

    # Remove rows containing NaN
    data, numEmpty = shared._filter(data, data.dropna(), "empty fields")

    # Remove rows with travel detail containing null
    data, numNull = shared._filter(data, data[~data['TRANSFER_DETAIL'].str.contains("null")], "null in transfer description")

    # Remove records with travel time <= 0
    data, numTime = shared._filter(data, data[data['TRAVEL_TIME'] > 0], "travel time <= 0")

    # Remove records with travel distance <= 0
    data, numDistance = shared._filter(data, data[data['TRAVEL_DISTANCE'] > 0], "travel distance <= 0")

    cleanStat = [numEmpty, numNull, numDistance, numTime, len(data.index)]
    labels = ['Empty fields', 'Incomplete \ntransfer details', 'Negative distance', 'Negative travel time', 'Clean']

    if FLAGS.plot == 'True':
        shared._plotPie(cleanStat, labels, 'faulty', FLAGS.file)

    # Save day statistics
    with open(paths.STAT_DIR_DEFAULT+'cleaning.txt', 'a') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow([data['DATADAY'][0].day, total]+cleanStat)

    return data

def _saveVoc(lines, stops):
    """
    Save vocabularies to json for humans and pickle to load later
    """
    with open(paths.VOC_DIR_DEFAULT+'_lines.json', 'w') as fp: json.dump(lines, fp, indent=4)
    with open(paths.VOC_DIR_DEFAULT+'_stops.json', 'w') as fp: json.dump(stops, fp, indent=4)

    with open(paths.VOC_DIR_DEFAULT+'_lines.pkl', 'w') as fp: cPickle.dump(lines, fp)
    with open(paths.VOC_DIR_DEFAULT+'_stops.pkl', 'w') as fp: cPickle.dump(stops, fp)

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
    mode = ride.split('.')[0].split('(')[-1]

    try:
        # Shared fields across modes
        stop_b = r'(?P<stop_b>.+?)'
        stop_a = r'(?P<stop_a>.+?)'

        # Parse metro
        if mode == MODES_DEFAULT[0]:
            line_b = r'(?P<line_b>.+?)'
            line_a = r'(?P<line_a>.+?)'

            pattern = re.compile(r'\('+mode+r'[.]'+line_b+r'[:]'+stop_b+r'[-]'+line_a+r'[:]'+stop_a+r'[)]$')
            matcher = pattern.search(ride)

            if matcher:
                return 1, matcher.group('line_b'), matcher.group('stop_b'), matcher.group('line_a'), matcher.group('stop_a')
            else:
                if FLAGS.verbose == 'True': print('Failed at parsing metro ride:', ride)
                return 1, None, None, None, None

        # Parse bus
        elif mode == MODES_DEFAULT[1]:
            line_b = r'(?P<line_b>.+?)'
            direction_b = r'(?P<direction_b>[(].+?[)])'
            line_a = r'(?P<line_a>.+?)'
            direction_a = r'(?P<direction_a>[(].+?[)])'

            pattern = re.compile(r'\('+mode+r'[.]'+line_b+direction_b+r'[:]'+stop_b+r'[-]'+line_a+direction_a+r'[:]'+stop_a+r'[)]$')
            matcher = pattern.search(ride)

            if matcher:
                return 2, matcher.group('line_b'), matcher.group('stop_b'), matcher.group('line_a'), matcher.group('stop_a')
            else:
                if FLAGS.verbose == 'True': print('Failed at parsing bus ride:', ride)
                return 2, None, None, None, None

        # Parse bike
        elif mode == MODES_DEFAULT[2]:
            pattern = re.compile(r'\('+mode+r'[.]'+stop_b+r'[-]'+stop_a+r'[)]$')
            matcher = pattern.search(ride)

            if matcher:
                return 3, '0', matcher.group('stop_b'), '0', matcher.group('stop_a')
            else:
                if FLAGS.verbose == 'True': print('Failed at parsing bike ride:', ride)
                return 3, None, None, None, None

    except AttributeError:
        if FLAGS.verbose == 'True': print('Failed at: ', ride)
        return 0, None, None, None, None

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

def _createVocabularies(trips):
    """
    Create LINE and STOPS vocabularies based on the data in the given trips
    """
    # Split the trip into rides and gather components
    lines = set()
    stops = set()
    for index, trip in trips.iteritems():
        rides = trip.split('->')
        for ride in rides:
            _, lineOrigin, stopOrigin, lineDestination, stopDestination = _extractTripFeatures(ride)
            lines.add(lineOrigin)
            lines.add(lineDestination)
            stops.add(stopOrigin)
            stops.add(stopDestination)

    # Report length of vocabularies formed
    print('     Combined lines found:  {}'.format(len(lines)))
    print('     Combined stops found:  {}'.format(len(stops)))

    # Turn into dictionaries
    # IDs start from 1 instead of 0, because 0 means no travel at all
    lines =  dict(zip(lines, map(lambda x: str(x)+'-T',range(1,len(lines)+1))))
    stops = dict(zip(stops, map(lambda x: str(x)+'-T',range(1,len(stops)+1))))

    # Save updated vocabularies
    _saveVoc(lines, stops)

    return lines, stops

def _replaceNewLines(data):
    """
    Replace values from new found lines vocabulary
    """
    data = data.apply(lambda x: newLines.get(x,x))
    return data

def _replaceNewStops(data):
    """
    Replace values from new found stops vocabulary
    """
    data = data.apply(lambda x: newStops.get(x,x))
    return data

def _updateVocabularies(data, lines, stops):
    """
    Find lines and routes that were not tokenized, and include them in vocabularies
    """
    print('\nUpdating vocabularies')
    # Find cases to add
    lines_on = data['ON_LINE'][data['FLAG_ON_LINE'].isnull()]
    lines_off = data['OFF_LINE'][data['FLAG_OFF_LINE'].isnull()]
    stops_on = data['ON_STOP'][data['FLAG_ON_STOP'].isnull()]
    stops_off = data['OFF_STOP'][data['FLAG_OFF_STOP'].isnull()]

    # Create set to avoid duplicates
    global newLines, newStops

    newLines = set(lines_on.values)
    newLines.update(lines_off.values)
    newStops = set(stops_on.values)
    newStops.update(stops_off.values)

    # Report length of vocabularies formed
    print('     New lines found:  {}'.format(len(newLines)))
    print('     New stops found:  {}'.format(len(newStops)))

    if len(newLines) != 0:
        # Create dictionary starting from the last available ID in lines
        newLines =  dict(zip(newLines, map(str,range(len(lines)+1, len(lines)+1+len(newLines)))))

        # Replace cases with new lines vocabulary
        # In parallel
        start = datetime.now()
        num_cores = cpu_count()
        num_partitions = num_cores #8

        pool = Pool(num_cores)
        data_split = np.array_split(lines_on, num_partitions)
        lines_on = pd.concat(pool.map(_replaceNewLines, data_split))

        data_split = np.array_split(lines_off, num_partitions)
        lines_off = pd.concat(pool.map(_replaceNewLines, data_split))
        pool.close()
        pool.join()

        print("Parallel time: {}, chunks: {}".format(datetime.now()-start, num_partitions))

        # Assign new replacements to main dataframe
        data.loc[data['FLAG_ON_LINE'].isnull(), 'ON_LINE'] = lines_on
        data.loc[data['FLAG_OFF_LINE'].isnull(), 'OFF_LINE'] = lines_off

        # Fit new lines to format "-T"
        newLines = {k: str(v)+'-T' for k, v in newLines.items()}

        # Update lines vocabulary
        lines.update(newLines)

    if len(newStops) != 0:
        # Create dictionary starting from the last available ID in stops
        newStops =  dict(zip(newStops, map(str,range(len(stops)+1, len(stops)+1+len(newStops)))))

        # Replace cases with new stops vocabulary
        # In parallel
        start = datetime.now()
        num_cores = cpu_count()
        num_partitions = num_cores #8

        pool = Pool(num_cores)
        data_split = np.array_split(stops_on, num_partitions)
        stops_on = pd.concat(pool.map(_replaceNewStops, data_split))

        data_split = np.array_split(stops_off, num_partitions)
        stops_off = pd.concat(pool.map(_replaceNewLines, data_split))
        pool.close()
        pool.join()

        print("Parallel time: {}, chunks: {}".format(datetime.now()-start, num_partitions))

        # Assign new replacements to main dataframe
        data.loc[data['FLAG_ON_STOP'].isnull(), 'ON_STOP'] = stops_on
        data.loc[data['FLAG_OFF_STOP'].isnull(), 'OFF_STOP'] = stops_off

        # Fit new lines to format "-T"
        newStops = {k: str(v)+'-T' for k, v in newStops.items()}

        # Update lines vocabulary
        stops.update(newStops)

    # Save updated vocabularies
    _saveVoc(lines, stops)

    return data

def _replaceWithVocabularies(data):
    """
    Replace values from vocabulary and flag new ones
    """
    # Replace according to vocabularies
    # data['ON_LINE'].replace(to_replace=lines, inplace=True)
    # data['OFF_LINE'].replace(to_replace=lines, inplace=True)
    #
    # data['ON_STOP'].replace(to_replace=stops, inplace=True)
    # data['OFF_STOP'].replace(to_replace=stops, inplace=True)

    # Apply is less memory consumming
    data['ON_LINE'] = data['ON_LINE'].apply(lambda x: lines.get(x,x))
    data['OFF_LINE'] = data['OFF_LINE'].apply(lambda x: lines.get(x,x))

    data['ON_STOP'] = data['ON_STOP'].apply(lambda x: stops.get(x,x))
    data['OFF_STOP'] = data['OFF_STOP'].apply(lambda x: stops.get(x,x))

    # Flag cases that were not replaced
    data[['ON_LINE', 'FLAG_ON_LINE']] = data['ON_LINE'].str.split('-', expand=True)
    data[['OFF_LINE', 'FLAG_OFF_LINE']] = data['OFF_LINE'].str.split('-', expand=True)
    data[['ON_STOP', 'FLAG_ON_STOP']] = data['ON_STOP'].str.split('-', expand=True)
    data[['OFF_STOP', 'FLAG_OFF_STOP']] = data['OFF_STOP'].str.split('-', expand=True)

    return data

# @profile
def _parseTrips(data, createVoc):
    """
    Parse 'TRANSFER_DETAIL' column to get ON/OFF mode, line and stop tokenized information
    """
    global lines, stops

    # Determine which vocabulary to use
    if createVoc == 'True':
        # Create vocabularies
        print('Creating lines and stops vocabularies')
        lines, stops = _createVocabularies(data['TRANSFER_DETAIL'])
    else:
        # Load existing vocabularies
        with open(paths.VOC_DIR_DEFAULT+'_lines.pkl', 'r') as fp: lines = cPickle.load(fp)
        with open(paths.VOC_DIR_DEFAULT+'_stops.pkl', 'r') as fp: stops = cPickle.load(fp)

    # Retrieve on and off trip details
    data['ON_MODE'], data['OFF_MODE'], data['ON_LINE'], data['OFF_LINE'], data['ON_STOP'], data['OFF_STOP'] = zip(*data['TRANSFER_DETAIL'].apply(lambda x : _extractOriginAndDestinationFeatures(x)))

    # Replace for clean format
    print('Formating trip')

    # Replace line and stops from vocabularies

    # Sequential
    # df = data.copy()
    # start = datetime.now()
    # df = _replaceWithVocabularies(df)
    # print("Sequential time: {}".format(datetime.now()-start))

    # In parallel
    start = datetime.now()
    num_cores = cpu_count()
    num_partitions = num_cores #8

    pool = Pool(num_cores)
    data_split = np.array_split(data, num_partitions)
    data = pd.concat(pool.map(_replaceWithVocabularies, data_split))
    pool.close()
    pool.join()
    print("Parallel time: {}, chunks: {}".format(datetime.now()-start, num_partitions))

    # If the vocabulary was created on this run, we do not need to update
    if createVoc != 'True':
        # Update vocabularies
        data = _updateVocabularies(data, lines, stops)

    return data

def _countTransfers(data):
    """
    Re calculate number of transfers, and transfer average time (which is dependent on the previous)
    """
    # Plot number of transfers before and after patching
    if FLAGS.plot == 'True':
        original = data.copy()

    print("Recalculating transfer number and transfer average time")
    data['TRANSFER_NUM'] = data['TRANSFER_DETAIL'].str.count("->")
    data['TRANSFER_TIME_AVG'] = np.where(data['TRANSFER_NUM'] > 0, data['TRANSFER_TIME_SUM'] / data['TRANSFER_NUM'], data['TRANSFER_NUM'])

    if FLAGS.plot == 'True':
        shared._plotDistributionCompare(original['TRANSFER_NUM'], data['TRANSFER_NUM'], 'Number of transfers', FLAGS.file, \
        labels=['Original', 'Recalculation'], bins='Auto')
        shared._plotDistributionCompare(original['TRANSFER_TIME_AVG'], data['TRANSFER_TIME_AVG'], 'Transfer average time', FLAGS.file, \
        labels=['Original', 'Recalculation'], bins=20)

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
    data['WEEKDAY'] = data['DATADAY'].dt.dayofweek + 1

    return data

def _numTrips(data):
    """
    Count number of trips in this day
    """
    data['NUM_TRIPS'] = data.groupby('CARD_CODE')['TRAVEL_DISTANCE'].transform('count')

    return data

def _orderFeatures(data):
    """
    Order features by: General, then spatial boarding, then spatial alighting
    """
    print("Order by type of feature: general then spatial")
    order = ['CARD_CODE', 'DAY', 'WEEKDAY', 'NUM_TRIPS', 'PATH_LINK', \
            'TRAVEL_TIME', 'TRAVEL_DISTANCE', \
            'TRANSFER_NUM', 'TRANSFER_TIME_AVG', 'TRANSFER_TIME_SUM', \
            'START_TIME', 'END_TIME', 'START_HOUR', 'END_HOUR', \
            'TRANSFER_DETAIL', \
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
    if FLAGS.sampleBy == 'Code':
        fileName = paths.CLEAN_DIR_DEFAULT+FLAGS.file+'- sample codes'
    elif FLAGS.sampleBy == 'Trip':
        fileName = paths.CLEAN_DIR_DEFAULT+FLAGS.file+'- sample trips'
    else:
        fileName = paths.CLEAN_DIR_DEFAULT+FLAGS.file+'-full'

    data.to_csv(fileName+'.csv', encoding='utf-8')

def prepare():
    """
    Read raw data, clean it, format it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.RAW_DIR_DEFAULT+FLAGS.file+'.csv')

    if FLAGS.sampleBy == 'Code':
        print("------------------------- Sample by codes -------------------------")
        data = _sampleByCode(data)

    print("----------------------------  Cleaning ----------------------------")
    data = _clean(data)


    if FLAGS.sampleBy == 'Trip':
        print("------------------------- Sample by trips -------------------------")
        data = _sampleByTrip(data)

    print("------------------ Extract  relevant information ------------------")

    print("               ----------- Parsing  trip ------------              ")
    data = _parseTrips(data, FLAGS.create_voc)

    print("               ----- Creating time stamp bins ------               ")
    data = _to_time_bins(data)

    print("               ------ Extract day  attributes ------               ")
    data = _weekday(data)

    print("               ------ Extract number of trips ------               ")
    data = _numTrips(data)

    print("----------------------------  Patching ----------------------------")
    data = _countTransfers(data)

    print("---------------------------  Formatting ---------------------------")
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
    parser.add_argument('--file', type = str, default = paths.FILE_DEFAULT,
                        help='File to prepare')
    parser.add_argument('--verbose', type = str, default = 'False',
                        help='Display parse trip details.')
    parser.add_argument('--create_voc', type = str, default = 'False',
                        help='Create lines/stops vocabularies from given data. If False, previously saved vocabularies will be used')
    parser.add_argument('--plot', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--sampleBy', type = str, default = 'Code',
                        help='May sample by "Code", or "Trip". False to run full.')


    FLAGS, unparsed = parser.parse_known_args()
    main(None)
