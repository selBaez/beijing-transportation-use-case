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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random, cPickle, re, json
from collections import OrderedDict

import paths, visualization

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

def _createVocabularies(trips):
    """
    Create LINE, ROUTE and STOPS vocabularies based on the data in the given trips
    """
    lines = set()
    routes = set()
    sttaions = set()
    for index, trip in trips.iteritems():
        if FLAGS.verbose == 'True': print('Trip: ',trip)
        rides = trip.split('->')
        for ride in rides:
            lines, routes, stops = _gatherRideComponents(ride, lines, routes)

    # TODO: smarter way to save direction, for now we just ignore it
    print('Subway lines found:       ',len(lines))
    print('Bus routes found:         ',len(routes))
    print('Combined stops found:  ',len(stops))

    # Turn into dictionaries
    lines =  dict(zip(lines, map(lambda x: ' '+str(x)+':',range(len(lines)))))               # TODO: fix . or - cases, for now we replace both with a space
    routes = dict(zip(routes, map(lambda x: ' '+str(x)+':',range(len(routes)))))
    stops = dict(zip(map(lambda x: x.replace('(', '[(]').replace(')', '[)]'), stops), map(str,range(len(stops)))))

    # Sort them to have longest patterns replaced first
    lines = OrderedDict(sorted(lines.items(), key=lambda t: len(t[0]), reverse=True))
    routes = OrderedDict(sorted(routes.items(), key=lambda t: len(t[0]), reverse=True))
    stops = OrderedDict(sorted(stops.items(), key=lambda t: len(t[0]), reverse=True))

    # Save as JSON later use
    with open(paths.VOC_DIR_DEFAULT+'_lines.json', 'w') as fp: json.dump(lines, fp, indent=4, ensure_ascii=False)
    with open(paths.VOC_DIR_DEFAULT+'_routes.json', 'w') as fp: json.dump(routes, fp, indent=4, ensure_ascii=False)
    with open(paths.VOC_DIR_DEFAULT+'_stops.json', 'w') as fp: json.dump(stops, fp, indent=4, ensure_ascii=False)

    return lines, routes, stops

def _gatherRideComponents(ride, lines=set(), routes=set(), stops=set()):
    """
    Get MODE, STOPS and other mode-specific components for the given ride and append to corresponding sets
    Return components cummulative sets

    BIKE = (bike.STOP-STOP)
    SUBWAY = (subway.LINE_NAME:STOP-LINE_NAME:STOP)
    BUS = (bus.ROUTE_NAME(DIRECTION-DIRECTION):STOP-ROUTE_NAME(DIRECTION-DIRECTION):STOP)

    GENERAL = (MODE.[LINE/ROUTE_NAME:]?STOP-[LINE/ROUTE_NAME:]?STOP)[->PATTERN]?
    # MODE              轨道|公交|自行车        ------ equivalent to ------       subway | bus | bike
    # LINE_NAME         5 号线 | NAME 线       ------ equivalent to ------       5 number line | NAME line
    # ROUTE_NAME        944 | 夜 32           ------ equivalent to ------       944 | night 32

    """
    #TODO: consider using split instead of regular expressions at all
    if FLAGS.verbose == 'True': print('Ride details: ',ride)

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
            lines.add('([.]|[-])'+matcher.group('line_b')+'[:]')
            lines.add('([.]|[-])'+matcher.group('line_a')+'[:]')
            stops.add(matcher.group('stop_b'))
            stops.add(matcher.group('stop_a'))
        else:
            print('Failed at parsing metro ride:', ride)

    # Parse bus
    elif matcher.group('mode') == '公交':
        route_b = r'(?P<route_b>.+?)'
        direction_b = r'(?P<direction_b>[(].+?[)])'
        route_a = r'(?P<route_a>.+?)'
        direction_a = r'(?P<direction_a>[(].+?[)])'

        pattern = re.compile(r'\('+mode+r'[.]'+route_b+direction_b+r'[:]'+stop_b+r'[-]'+route_a+direction_a+r'[:]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            routes.add('([.]|[-])'+matcher.group('route_b')+'[:]')
            routes.add('([.]|[-])'+matcher.group('route_a')+'[:]')
            stops.add(matcher.group('stop_b'))
            stops.add(matcher.group('stop_a'))
        else:
            print('Failed at parsing bus ride:', ride)

    # Parse bike
    elif matcher.group('mode') == '自行车':
        pattern = re.compile(r'\('+mode+r'[.]'+stop_b+r'[-]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            stops.add(matcher.group('stop_b'))
            stops.add(matcher.group('stop_a'))
        else:
            print('Failed at parsing bike ride:', ride)

    return lines, routes, stops

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
    Return MODE, LINE/ROUTE and STOP for the given ride
    """
    #TODO refactor with gather ride components

    if FLAGS.verbose == 'True': print('Ride details: ', ride)

    # Shared fields across modes
    mode = r'(?P<mode>R|B|Z)'
    stop_b = r'(?P<stop_b>.+?)'
    stop_a = r'(?P<stop_a>.+?)'

    # Match and classify by mode
    pattern = re.compile(mode)
    matcher = pattern.search(ride)

    # Parse metro or bus
    if matcher.group('mode') == 'R' or matcher.group('mode') == 'B':
        line_b = r'(?P<line_b>.+?)'
        line_a = r'(?P<line_a>.+?)'

        pattern = re.compile(r'\('+mode+r'[ ]'+line_b+r'[:]'+stop_b+r'[ ]'+line_a+r'[:]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            if matcher.group('mode') == 'R': return 0, matcher.group('line_b'), matcher.group('stop_b'), matcher.group('line_a'), matcher.group('stop_a')
            elif matcher.group('mode') == 'B': return 1, matcher.group('line_b'), matcher.group('stop_b'), matcher.group('line_a'), matcher.group('stop_a')
        else:
            print('Failed at parsing metro/bus ride:', ride)
            return matcher.group('mode'), None, None, None, None

    # Parse bike
    elif matcher.group('mode') == 'Z':
        pattern = re.compile(r'\('+mode+r'[.]'+stop_b+r'[-]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            return 2, '0', matcher.group('stop_b'), '0', matcher.group('stop_a')
        else:
            print('Failed at parsing bike ride:', ride)
            return matcher.group('mode'), None, None, None, None

def _parseTrips(data, modes, createVoc):
    """
    Parse 'TRANSFER_DETAIL' column to get route
    """
    # Determine which vocabulary to use
    if createVoc == 'True':
        # Create vocabularies
        print('Creating lines, routes and stops vocabularies')
        lines, routes, stops = _createVocabularies(data['TRANSFER_DETAIL'])
    else:
        # Load existing vocabularies TODO: not working atm
        with open(paths.VOC_DIR_DEFAULT+'_lines.json', 'r') as fp: lines = json.load(fp, encoding="utf-8")
        with open(paths.VOC_DIR_DEFAULT+'_routes.json', 'r') as fp: routes = json.load(fp, encoding="utf-8")
        with open(paths.VOC_DIR_DEFAULT+'_stops.json', 'r') as fp: stops = json.load(fp, encoding="utf-8")

    # Replace for clean format
    print('Formating route')

    # TODO: remove when run over whole dataset
    indices = random.sample(data.index, 1000)
    data = data.ix[indices]
    data = _filter(data, data[data['TRANSFER_NUM'] > 0], "at least one transfer")

    # Replace mode : dictionary
    data['TRANSFER_DETAIL'].replace(to_replace=modes, inplace=True, regex=True)

    # Strip bus directions away
    data['TRANSFER_DETAIL'].replace(to_replace='[(][^.-]+?[-][^.-]+?[)]:', value=':', inplace=True, regex=True)

    # Replace line/route and stops/stops : vocabularies
    data['TRANSFER_DETAIL'].replace(to_replace=routes, inplace=True, regex=True)
    data['TRANSFER_DETAIL'].replace(to_replace=lines, inplace=True, regex=True)
    data['TRANSFER_DETAIL'].replace(to_replace=stops, inplace=True, regex=True)

    # Retrieve on and off trip details
    data['ON_MODE'], data['OFF_MODE'], data['ON_LINE'], data['OFF_LINE'], data['ON_STOP'], data['OFF_STOP'] =  zip(*data['TRANSFER_DETAIL'].apply(lambda x : _extractOriginAndDestinationFeatures(x)))

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
        visualization._plotDistributionCompare(original['TRANSFER_NUM'], data['TRANSFER_NUM'], 'Number of transfers', labels=['Original', 'Recalculation'], bins='Auto')
        visualization._plotDistributionCompare(original['TRANSFER_TIME_AVG'], data['TRANSFER_TIME_AVG'], 'Transfer average time', labels=['Original', 'Recalculation'], bins=20)

    return data

def _to_time_bins(data):
    """
    Start and end time stamps into time bins
    """
    print("Extracting start/end hours")
    data['START_HOUR'] = data['START_TIME'].apply(lambda x : x.hour)
    data['END_HOUR'] = data['END_TIME'].apply(lambda x : x.hour)
    return data

def _orderFeatures(data):
    """
    Order features by: General, then spatial boarding, then spatial alighting
    """
    order = ['DATADAY', 'CARD_CODE', 'PATH_LINK', 'TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_NUM', 'TRANSFER_TIME_AVG', \
            'TRANSFER_TIME_SUM', 'START_TIME', 'END_TIME', 'START_HOUR', 'END_HOUR', 'TRANSFER_DETAIL', 'NUM_TRIPS', \
            'ON_AREA', 'OFF_AREA', \
            'ON_TRAFFIC', 'OFF_TRAFFIC', 'ON_MIDDLEAREA', 'OFF_MIDDLEAREA', 'ON_BIGAREA', 'OFF_BIGAREA', \
            'ON_RINGROAD', 'OFF_RINGROAD',  \
            'ON_MODE', 'OFF_MODE', 'ON_LINE', 'OFF_LINE', 'ON_STOP', 'OFF_STOP']
    data = data[order]

    return data

def _store(data):
    """
    Store data for use in model
    """
    #TODO store test and train separately
    data.to_pickle(paths.CLEAN_FILE_DEFAULT+'.pkl')
    data.to_csv(paths.CLEAN_FILE_DEFAULT+'.csv')

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(paths.ORIGINAL_FILE_DEFAULT)

    print("---------------------------- Cleaning -----------------------------")
    data = _clean(data, FLAGS.min_records)

    print("-------------------------- Parsing route --------------------------")
    data = _parseTrips(data, MODE_DICT_DEFAULT, FLAGS.create_voc)

    print("------------------ Recalculating transfer number ------------------")
    data = _countTransfers(data)

    print("-------------------- Creating time stamp bins ---------------------")
    data = _to_time_bins(data)

    #print("------------------------ Extract weekdays -------------------------")
    # TODO http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%204%20-%20Find%20out%20on%20which%20weekday%20people%20bike%20the%20most%20with%20groupby%20and%20aggregate.ipynb

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
    #TODO Make directories if they do not exists yet
    preprocess()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type = str, default = 'False',
                        help='Display parse route details.')
    parser.add_argument('--min_records', type = int, default = MIN_RECORDS_DEFAULT,
                        help='Traveler is required to have at least this number of records.')
    parser.add_argument('--create_voc', type = str, default = 'True',
                        help='Create lines/routes/stops vocabularies from given data. If False, previously saved vocabularies will be used')
    parser.add_argument('--plot_distr', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')
    #TODO: overwrite voc

    FLAGS, unparsed = parser.parse_known_args()
    main(None)
