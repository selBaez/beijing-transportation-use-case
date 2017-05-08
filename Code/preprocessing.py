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
import json
from collections import OrderedDict

############ --- BEGIN default constants --- ############
MIN_RECORDS_DEFAULT = 0
MODE_DICT_DEFAULT = {       '[(]轨道[.]' : '(R.',        # subway
                            '[(]公交[.]' : '(B.',        # bus
                            '[(]自行车[.]' : '(Z.'}      # bike
############ --- END default constants--- ############

############ --- BEGIN default directories --- ############
LOAD_FILE_DEFAULT = '../Data/sets/Travel chain sample data(50000).csv'
LINES_VOC_FILE_DEFAULT = '../Data/vocabularies/full_lines vocabulary.json'
ROUTES_VOC_FILE_DEFAULT = '../Data/vocabularies/full_routes vocabulary.json'
STOPS_VOC_FILE_DEFAULT = '../Data/vocabularies/full_stops vocabulary.json'
PLOT_DIR_DEFAULT = './Plots/'
SAVE_TO_FILE_DEFAULT = '../Data/sets/preprocessed sample data(50000)'
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

def _createVocabularies(trips, lines_voc=LINES_VOC_FILE_DEFAULT, routes_voc=ROUTES_VOC_FILE_DEFAULT, stops_voc=STOPS_VOC_FILE_DEFAULT):
    """
    Create LINE, ROUTE and STOPS vocabularies based on the data in the given trips
    """
    lines = set()
    routes = set()
    sttaions = set()
    for index, trip in trips.iteritems():
        # if FLAGS.verbose == 'True': print('Trip: ',trip)
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
    with open(lines_voc, 'w') as fp: json.dump(lines, fp, indent=4, ensure_ascii=False)
    with open(routes_voc, 'w') as fp: json.dump(routes, fp, indent=4, ensure_ascii=False)
    with open(stops_voc, 'w') as fp: json.dump(stops, fp, indent=4, ensure_ascii=False)

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
    # if FLAGS.verbose == 'True': print('Ride details: ',ride)

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
            # if FLAGS.verbose == 'True':
                # print('Parsing a metro ride')
                # print('Mode:                  ',matcher.group('mode'))
                # print('Boarding Line:         ',matcher.group('line_b'))
                # print('Boarding Stop:      ',matcher.group('stop_b'))
                # print('Alighting Line:        ',matcher.group('line_a'))
                # print('Alighting Stop:     ',matcher.group('stop_a'))

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
            # if FLAGS.verbose == 'True':
                # print('\nParsing a bus ride')
                # print('Mode:                  ',matcher.group('mode'))
                # print('Boarding Route:        ',matcher.group('route_b'))
                # print('Boarding Stop:      ',matcher.group('stop_b'))
                # print('Alighting Route:       ',matcher.group('route_a'))
                # print('Alighting Stop:     ',matcher.group('stop_a'))

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
            # if FLAGS.verbose == 'True':
                # print('Parsing a bike ride')
                # print('Mode:                  ',matcher.group('mode'))
                # print('Boarding Stop:      ',matcher.group('stop_b'))
                # print('Alighting Stop:     ',matcher.group('stop_a'))

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

    print(modeOrigin, lineOrigin, stopOrigin, modeDestination, lineDestination, stopDestination)

    return modeOrigin, lineOrigin, stopOrigin, modeDestination, lineDestination, stopDestination

def _extractTripFeatures(ride):
    """
    Return MODE, LINE/ROUTE and STOP for the given ride
    """
    #TODO refactor with gather ride components

    # if FLAGS.verbose == 'True': print('Ride details: ', ride)

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
            return matcher.group('mode'), matcher.group('line_b'), matcher.group('stop_b'), matcher.group('line_a'), matcher.group('stop_a')
        else:
            print('Failed at parsing metro ride:', ride)
            return matcher.group('mode'), None, None, None, None

    # Parse bike
    elif matcher.group('mode') == 'Z':
        pattern = re.compile(r'\('+mode+r'[.]'+stop_b+r'[-]'+stop_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher:
            return matcher.group('mode'), '0', matcher.group('stop_b'), '0', matcher.group('stop_a')
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
        with open(LINES_VOC_FILE_DEFAULT, 'r') as fp: lines = json.load(fp, encoding="utf-8")
        with open(ROUTES_VOC_FILE_DEFAULT, 'r') as fp: routes = json.load(fp, encoding="utf-8")
        with open(STOPS_VOC_FILE_DEFAULT, 'r') as fp: stops = json.load(fp, encoding="utf-8")

    # Replace for clean format
    print('Formating route')

    # TODO: remove when run over whole dataset
    indices = random.sample(data.index, 10)
    data = data.ix[indices]
    data = _filter(data, data[data['TRANSFER_NUM'] > 0], "at least one transfer")

    if FLAGS.verbose == 'True': print('\n     Original details', data['TRANSFER_DETAIL'][0])

    # Replace mode : dictionary
    data['TRANSFER_DETAIL'].replace(to_replace=modes, inplace=True, regex=True)

    # Strip bus directions away
    data['TRANSFER_DETAIL'].replace(to_replace='[(][^.-]+?[-][^.-]+?[)]:', value=':', inplace=True, regex=True)

    # Replace line/route and stops/stops : vocabularies
    data['TRANSFER_DETAIL'].replace(to_replace=routes, inplace=True, regex=True)
    data['TRANSFER_DETAIL'].replace(to_replace=lines, inplace=True, regex=True)
    data['TRANSFER_DETAIL'].replace(to_replace=stops, inplace=True, regex=True)

    if FLAGS.verbose == 'True': print('     Parsed details', data['TRANSFER_DETAIL'][0], '\n\n')

    # Retrieve on and off trip details
    data['ON_MODE'], data['ON_LINE'], data['ON_STOP'], data['OFF_MODE'], data['OFF_LINE'], data['OFF_STOP'] =  zip(*data['TRANSFER_DETAIL'].apply(lambda x : _extractOriginAndDestinationFeatures(x)))

    return data

def _countTransfers(data):
    """
    Re calculate number of transfers
    """
    # Plot number of transfers before and after patching
    if FLAGS.plot_distr == 'True':
        original = data['TRANSFER_NUM'].copy()

    print("Recalculating transfer number")
    data['TRANSFER_NUM'] = data['TRANSFER_DETAIL'].str.count("->")

    if FLAGS.plot_distr == 'True':
        new = data['TRANSFER_NUM']
        _plotDistributionCompare(original, new, 'Number of transfers', labels=['Original', 'Recalculation'])

    return data

def _to_time_bins(data):
    """
    Start and end time stamps into time bins
    """
    print("Extracting start/end hours")
    data['START_HOUR'] = data['START_TIME'].apply(lambda x : x.hour)
    data['END_HOUR'] = data['END_TIME'].apply(lambda x : x.hour)
    return data

def _plotDistributionCompare(sample1, sample2, variable_name, labels, xticks=None):
    """
    Plot variables distribution with frequency histogram
    """
    # Plot variable frequency histogram
    fig, ax = plt.subplots()

    # Make plot pretty
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0))

    plt.hist([sample1, sample2], label=labels, color=['#578ac1', '#57c194'])

    plt.legend(loc='upper right')
    plt.xlabel(variable_name)

    # Save
    plt.savefig(FLAGS.plot_dir+variable_name+'_hist.png', format='png')

def _plotDistribution(sample, variable_name, column_name, bins=None, xticks=None):
    """
    Plot variables distribution with frequency histogram
    """
    fig, ax = plt.subplots()

    # Make plot pretty
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    if xticks != None:
        plt.xticks(np.arange(xticks[0], xticks[1], 1.0))

    # Bins
    if bins == None:
        bins = max(sample[column_name]) - min(sample[column_name])

    # Plot
    sample[column_name].plot.hist(ax=ax, bins=bins, color='#578ac1')
    plt.xlabel(variable_name)

    # Save
    plt.savefig(FLAGS.plot_dir+variable_name+'_hist.png', format='png')

def _standardize(data):
    """
    Rescale features to have mean 0 and std 1
    """

    if FLAGS.plot_distr == 'True':
        # Sample 1000 random points
        indices = random.sample(data.index, 2500)
        sample = data.ix[indices]

        # Plot general features
        print("Plotting hour distributions")
        _plotDistributionCompare(sample['START_HOUR'], sample['END_HOUR'], 'Hour of trip', labels=['Start', 'End'], xticks=[0.0, 25.0])
        _plotDistribution(sample, 'Number of trips', 'NUM_TRIPS', xticks=[0.0, 8.0])

        # Plot features to be standardized
        print("Plotting original travel time and distance distributions")
        _plotDistribution(sample, 'Travel time', 'TRAVEL_TIME', bins=20)
        _plotDistribution(sample, 'Travel distance', 'TRAVEL_DISTANCE', bins=20)
        _plotDistribution(sample, 'Total transfer time', 'TRANSFER_TIME_SUM', bins=20)
        _plotDistribution(sample, 'Average transfer time', 'TRANSFER_TIME_AVG', bins=20)

        # Plot correlated features to be standardized: time vs distance
        fig, ax = plt.subplots()
        sample.plot(x='TRAVEL_DISTANCE',y='TRAVEL_TIME', ax=ax, kind='scatter')
        plt.savefig(FLAGS.plot_dir+'distance_vs_time.png', format='png')

    # TODO: only fit and transform to train data, and transform test data
    print("Standarize travel time and distance")
    scaler = StandardScaler()
    data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']] = scaler.fit_transform(data[['TRAVEL_TIME', 'TRAVEL_DISTANCE', 'TRANSFER_TIME_SUM', 'TRANSFER_TIME_AVG']])

    if FLAGS.plot_distr == 'True':
        # Use previous sample
        sample = data.ix[indices]

        # Plot standardized features
        print("Plotting standarized travel time and distance distributions")
        _plotDistribution(sample, 'Travel time standardized', 'TRAVEL_TIME', bins=20)
        _plotDistribution(sample, 'Travel distance standardized', 'TRAVEL_DISTANCE', bins=20)
        _plotDistribution(sample, 'Total transfer time standardized', 'TRANSFER_TIME_SUM', bins=20)
        _plotDistribution(sample, 'Average transfer time standardized', 'TRANSFER_TIME_AVG', bins=20)

        # Box plot of all standardized features

        # Plot standardized correlated features: time vs distance
        fig, ax = plt.subplots()
        sample.plot(x='TRAVEL_DISTANCE',y='TRAVEL_TIME', ax=ax, kind='scatter')
        plt.savefig(FLAGS.plot_dir+'distance_vs_time_standardized.png', format='png')

    return data

def _store(data):
    """
    Store data for use in model
    """
    #TODO store test and train separately
    data.to_pickle(FLAGS.save_to_file+'.pkl')
    data.to_csv(FLAGS.save_to_file+'.csv')

def preprocess():
    """
    Read raw data, clean it and store preprocessed data
    """
    print("---------------------------- Load data ----------------------------")
    data = _loadData(FLAGS.load_file)

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

    #print("----------------------- Finding smart codes -----------------------")
    #TODO find records related to given smart codes

    #print("------------------- Create train  and test sets -------------------")
    #TODO divide and add labels?

    print("-------------------------- Standardizing --------------------------")
    data = _standardize(data)

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
    parser.add_argument('--verbose', type = str, default = 'True',
                        help='Display parse route details.')
    parser.add_argument('--load_file', type = str, default = LOAD_FILE_DEFAULT,
                        help='Data file to load.')
    parser.add_argument('--min_records', type = int, default = MIN_RECORDS_DEFAULT,
                        help='Traveler is required to have at least this number of records.')
    parser.add_argument('--create_voc', type = str, default = 'True',
                        help='Create lines/routes/stops vocabularies from given data. If False, previously saved vocabularies will be used')
    parser.add_argument('--plot_distr', type = str, default = 'True',
                        help='Boolean to decide if we plot distributions.')
    parser.add_argument('--plot_dir', type = str, default = PLOT_DIR_DEFAULT,
                        help='Directory to which save plots.')
    parser.add_argument('--save_to_file', type = str, default = SAVE_TO_FILE_DEFAULT,
                        help='Data file to save data in.')
    #TODO: overwrite voc
    #TODO: labeled or unlabeled? (labeled includes searching for codes)

    FLAGS, unparsed = parser.parse_known_args()
    main(None)
