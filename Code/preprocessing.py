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

############ --- BEGIN default constants --- ############
MIN_RECORDS_DEFAULT = 2
MODE_DICT_DEFAULT = {       '[(]轨道.' : '(R.',        # subway
                            '[(]公交.' : '(B.',        # bus
                            '[(]自行车.' : '(Z.'}#,    # bike
                            #'线' : 'Line',
                            #'号' : '',
                            #'夜' : 'N',       # Night bus
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
LOAD_FILE_DEFAULT = '../Data/Travel chain sample data(50000).csv'
LINES_VOC_FILE_DEFAULT = '../Data/lines vocabulary.json'
ROUTES_VOC_FILE_DEFAULT = '../Data/routes vocabulary.json'
STATIONS_VOC_FILE_DEFAULT = '../Data/stations vocabulary.json'
PLOT_DIR_DEFAULT = './Plots/'
SAVE_TO_FILE_DEFAULT = '../Data/preprocessed sample data(50000)'
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

def _createVocabularies(trips, lines_voc=LINES_VOC_FILE_DEFAULT, routes_voc=ROUTES_VOC_FILE_DEFAULT, stations_voc=STATIONS_VOC_FILE_DEFAULT):
    """
    Create LINE, ROUTE and STATIONS vocabularies
    """
    lines = set()
    routes = set()
    sttaions = set()
    for index, trip in trips.iteritems():
        # if FLAGS.verbose == 'True': print('Trip: ',trip)
        rides = re.split('->', trip)
        for ride in rides:
            lines, routes, stations = _extractRideComponents(ride, lines, routes)

    # TODO: smarter way to save direction, for now we just ignore it
    print('Subway lines found:       ',len(lines))
    print('Bus routes found:         ',len(routes))
    print('Combined stations found:  ',len(stations))

    # Deal with parenthesis
    # for station in stations:
    #     print(route)
    #     route = route.replace('(', '\(')
    #     route = route.replace(')', '\)')
    #     print(route)

    # Turn into dictionaries
    # TODO: fix . or - cases
    lines =  dict(zip(lines, map(lambda x: '-'+str(x)+':',range(len(lines)))))
    routes = dict(zip(routes, map(lambda x: '-'+str(x)+':',range(len(routes)))))
    stations = dict(zip(stations, map(str,range(len(stations)))))

    # Save as JSON later use
    with open(lines_voc, 'w') as fp: json.dump(lines, fp, indent=4, ensure_ascii=False)
    with open(routes_voc, 'w') as fp: json.dump(routes, fp, indent=4, ensure_ascii=False)
    with open(stations_voc, 'w') as fp: json.dump(stations, fp, indent=4, ensure_ascii=False)

    return lines, routes, stations

def _extractRideComponents(ride, lines=set(), routes=set(), stations=set()):
    """
    Get MODE, STATIONS and other mode-specific components

    BIKE = (bike.STATION-STATION)
    SUBWAY = (subway.LINE_NAME:STATION-LINE_NAME:STATION)
    BUS = (bus.ROUTE_NAME(DIRECTION-DIRECTION):STATION-ROUTE_NAME(DIRECTION-DIRECTION):STATION)

    GENERAL = (MODE.[LINE/ROUTE_NAME:]?STATION-[LINE/ROUTE_NAME:]?STATION)[->PATTERN]?
    # MODE              轨道|公交|自行车        ------ equivalent to ------       subway | bus | bike
    # LINE_NAME         5 号线 | NAME 线       ------ equivalent to ------       5 number line | NAME line
    # ROUTE_NAME        944 | 夜 32           ------ equivalent to ------       944 | night 32

    """
    # if FLAGS.verbose == 'True': print('Ride details: ',ride)

    # Shared fields across modes
    mode = r'(?P<mode>轨道|公交|自行车)'
    station_b = r'(?P<station_b>.+?)'
    station_a = r'(?P<station_a>.+?)'

    # Match and classify by mode
    pattern = re.compile(mode)
    matcher = pattern.search(ride)

    # Parse metro
    if matcher.group('mode') == '轨道':
        line_b = r'(?P<line_b>.+?)'
        line_a = r'(?P<line_a>.+?)'

        pattern = re.compile(r'\('+mode+r'[.]'+line_b+r'[:]'+station_b+r'[-]'+line_a+r'[:]'+station_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher and FLAGS.verbose == 'True':
            # print('Parsing a metro ride')
            # print('Mode:                  ',matcher.group('mode'))
            # print('Boarding Line:         ',matcher.group('line_b'))
            # print('Boarding Station:      ',matcher.group('station_b'))
            # print('Alighting Line:        ',matcher.group('line_a'))
            # print('Alighting Station:     ',matcher.group('station_a'))

            lines.add('([.]|[-])'+matcher.group('line_b')+'[:]')
            lines.add('([.]|[-])'+matcher.group('line_a')+'[:]')
            stations.add(matcher.group('station_b'))
            stations.add(matcher.group('station_a'))
        else:
            print('Failed at parsing metro ride:', ride)

    # Parse bus
    elif matcher.group('mode') == '公交':
        route_b = r'(?P<route_b>.+?)'
        direction_b = r'(?P<direction_b>[(].+?[)])'
        route_a = r'(?P<route_a>.+?)'
        direction_a = r'(?P<direction_a>[(].+?[)])'

        pattern = re.compile(r'\('+mode+r'[.]'+route_b+direction_b+r'[:]'+station_b+r'[-]'+route_a+direction_a+r'[:]'+station_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher and FLAGS.verbose == 'True':
            # print('\nParsing a bus ride')
            # print('Mode:                  ',matcher.group('mode'))
            # print('Boarding Route:        ',matcher.group('route_b'))
            # print('Boarding Station:      ',matcher.group('station_b'))
            # print('Alighting Route:       ',matcher.group('route_a'))
            # print('Alighting Station:     ',matcher.group('station_a'))

            routes.add('([.]|[-])'+matcher.group('route_b')+'[:]')
            routes.add('([.]|[-])'+matcher.group('route_a')+'[:]')
            stations.add(matcher.group('station_b'))
            stations.add(matcher.group('station_a'))
        else:
            print('Failed at parsing bus ride:', ride)

    # Parse bike
    elif matcher.group('mode') == '自行车':
        pattern = re.compile(r'\('+mode+r'[.]'+station_b+r'[-]'+station_a+r'[)]$')
        matcher = pattern.search(ride)

        if matcher and FLAGS.verbose == 'True':
            # print('Parsing a bike ride')
            # print('Mode:                  ',matcher.group('mode'))
            # print('Boarding Station:      ',matcher.group('station_b'))
            # print('Alighting Station:     ',matcher.group('station_a'))

            stations.add(matcher.group('station_b'))
            stations.add(matcher.group('station_a'))
        else:
            print('Failed at parsing bike ride:', ride)

    return lines, routes, stations

def _parseRoute(data, modes, createVoc):
    """
    Parse 'TRANSFER_DETAIL' column to get route
    """
    indices = random.sample(data.index, 1000)
    data = data.ix[indices]

    if createVoc == 'True':
        # Create vocabularies
        print('Creating lines, routes and stations vocabularies')
        lines, routes, stations = _createVocabularies(data['TRANSFER_DETAIL'])
    else:
        # Load existing vocabularies
        with open(LINES_VOC_FILE_DEFAULT, 'r') as fp: lines = json.load(fp, encoding="utf-8")
        with open(ROUTES_VOC_FILE_DEFAULT, 'r') as fp: routes = json.load(fp, encoding="utf-8")
        with open(STATIONS_VOC_FILE_DEFAULT, 'r') as fp: stations = json.load(fp, encoding="utf-8")

    # Replace for clean format
    print('Formating route')
    # Replace mode : dictionary
    data['TRANSFER_DETAIL'].replace(to_replace=modes, inplace=True, regex=True)

    #print(data['TRANSFER_DETAIL'][:5])
    data['TRANSFER_DETAIL'].replace(to_replace='[(][^.]+?[)]:', value=':', inplace=True, regex=True)              # Strip bus directions away

    # Replace line/route and stops/stations : vocabularies
    print('\nOriginal without direction\n',data['TRANSFER_DETAIL'][:5])
    data['TRANSFER_DETAIL'].replace(to_replace=routes, inplace=True, regex=True, limit=1)

    print('\nRoutes replaced\n',data['TRANSFER_DETAIL'][:5])
    data['TRANSFER_DETAIL'].replace(to_replace=lines, inplace=True, regex=True, limit=1)

    #print(data['TRANSFER_DETAIL'][:5])
    data['TRANSFER_DETAIL'].replace(to_replace=stations, inplace=True, regex=True) #TODO: fix replacement of best fit

    print(data['TRANSFER_DETAIL'][:5])

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
    #TODO: check why columns are not created
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

def _standardize(data, plot_distr, plot_dir):
    """
    Rescale features to have mean 0 and std 1
    """

    if plot_distr == 'True':
        # Sample 1000 random points
        indices = random.sample(data.index, 1000)
        sample = data.ix[indices]

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
    # TODO: scale transfer time too?
    print("Standarize travel time and distance")
    scaler = StandardScaler()
    data[['TRAVEL_TIME', 'TRAVEL_DISTANCE']] = scaler.fit_transform(data[['TRAVEL_TIME', 'TRAVEL_DISTANCE']])

    if plot_distr == 'True':
        # Sample 1000 random points
        indices = random.sample(data.index, 1000)
        sample = data.ix[indices]

        print("Plotting standarized travel time and distance distributions")
        _plotDistribution(sample, plot_dir, 'time_standardized', 'TRAVEL_TIME')
        _plotDistribution(sample, plot_dir, 'distance_standardized', 'TRAVEL_DISTANCE')

        # Plot time vs distance
        fig, ax = plt.subplots()
        sample.plot(x='TRAVEL_DISTANCE',y='TRAVEL_TIME', ax=ax, kind='scatter')
        plt.savefig(plot_dir+'distance_vs_time_standardized.png', format='png')

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
    data = _parseRoute(data, MODE_DICT_DEFAULT, FLAGS.create_voc)

    print("------------------ Recalculating transfer number ------------------")
    #data = _countTransfers(data)

    print("-------------------- Creating time stamp bins ---------------------")
    #data = _to_time_bins(data)

    #print("------------------------ Extract weekdays -------------------------")
    # TODO http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%204%20-%20Find%20out%20on%20which%20weekday%20people%20bike%20the%20most%20with%20groupby%20and%20aggregate.ipynb

    #print("------------------- Create train  and test sets -------------------")
    #TODO divide and add labels?

    print("-------------------------- Standardizing --------------------------")
    #data = _standardize(data, FLAGS.plot_distr, FLAGS.plot_dir)

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
                        help='Create lines/routes/stations vocabularies from given data. If False, previously saved vocabularies will be used')
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
