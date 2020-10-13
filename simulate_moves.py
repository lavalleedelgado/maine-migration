from typing import Tuple, List, Dict, Set, Callable, Iterator, Any
import argparse
import sqlite3
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# Identify default command line arguments.
DEFAULT_DB_PATH = 'acs/acs.sqlite3'
DEFAULT_SEED = 1

# Identify default columns on which to assess similarity.
CHARACTERISTICS = ['white', 'black', 'hispanic', 'asian', 'bachelors']

# Identify column labels for simulation result.
SIMULATION_DATA_LABLES = ['year', 'origin', 'destination']


def simulate_moves(
    db_path: str,
    counties_0: List[int],
    counties_1: List[int],
    year: int
) -> Iterator[Tuple[int, int]]:
    '''
    Simulate tract-to-tract moves from county-to-county quantities. Optimize the
    case where there is one destination county in the inner loop. NB: command
    line interface accepts one county for origin or destination, but many are
    possible and the default behavior loads all available counties.

    db_path (str): location of the database.
    counties_0 (list): collections of FIPS codes for origin counties.
    counties_1 (list): collections of FIPS codes for destination counties.
    year (int): time at which to simulate moves.

    Return origin-destination census tract pairs (list of tuples).
    '''
    # Set an indicator for whether there is just one destination county.
    one_destination = len(counties_1) == 1
    # Load destination tracts now if optimal.
    if one_destination:
        tracts_1 = __get_destination_tracts(db_path, counties_1[0], year)
    # Consider each origin county.
    for county_0 in counties_0:
        # Get the origin tracts and departure characteristics.
        tracts_0 = __get_origin_tracts(db_path, county_0, year)
        # Consider each destination county.
        for county_1 in counties_1:
            # Get the move type and number of moves for this county pair.
            move_t = __get_move_type(county_0, county_1)
            move_n = __get_move_count(db_path, county_0, county_1, year, move_t)
            # Skip this destination if there are no moves to simulate.
            if not move_n:
                continue
            # Get the destination tracts and arrival characteristics.
            if not one_destination:
                tracts_1 = __get_destination_tracts(db_path, county_1, year)
            # Simulate and report these moves.
            moves = __simulate_moves(tracts_0, tracts_1, move_t, move_n)
            # Report these moves to the end of the simulation.
            for tract_0, tract_1 in moves:
                yield tract_0, tract_1


def __get_counties(
    db_path: str
) -> List[int]:
    '''
    Request FIPS codes for all counties in the data, assuming that they are
    permanent entities across years.

    db_path (str): location of the database.

    Return county FIPS codes (list of ints).
    '''
    # Initialize a container for county FIPS codes.
    counties = []
    # Initialize a representation of the database.
    db = sqlite3.connect(db_path).cursor()
    # Write the select statement.
    select = '''
    SELECT DISTINCT county_last_year
    FROM flow
    WHERE county_last_year / 100000 = 0;
    '''
    # Execute the statement.
    db.execute(select)
    # Yield each resultant county.
    while True:
        county = db.fetchone()
        if not county:
            break
        counties.append(county[0])
    # Close the connection to the database.
    db.connection.close()
    # Return the list of county FIPS codes.
    return counties


def __validate_county(
    county: int
) -> bool:
    '''
    Check whether the FIPS code of a county is valid, whether it is an integer
    with either 4 or 5 digits.

    county (int): FIPS codes of a county.

    Return truth value (bool).
    '''
    assert isinstance(county, (int, np.integer))
    return 4 <= len(str(county)) <= 5


def __get_origin_tracts(
    db_path: str,
    county_0: int,
    year: int,
    columns: List[str] = CHARACTERISTICS
) -> pd.DataFrame:
    '''
    Request the census tracts in the origin county in a year with the count and
    characteristics of departures. This assumes departures are representative of
    change in the origin tract in the previous year.
    
    db_path (str): location of the database.
    county_0 (int): FIPS code of the origin county.
    year (int): time at which to simulate moves.
    columns (list): labels for profile characteristics.

    Return origin tracts with departures and characteristics (pd.DataFrame).
    '''
    # Ensure column labels are available.
    assert columns
    # Initialize a representation of the database.
    db = sqlite3.connect(db_path)
    # Write the predicate for departure characteristics.
    p = '(last.%s * population - this.%s * stayed) / (population - stayed) AS %s'
    # Write the select statement.
    select = f'''
    WITH
    last AS (
        SELECT tract, population, {', '.join(columns)}
        FROM profile
        WHERE tract / 1000000 = {county_0}
        AND year = {year} - 1
    ),
    this AS (
        SELECT tract, stayed, {', '.join(columns)}
        FROM profile
        WHERE tract / 1000000 = {county_0}
        AND year = {year}
    )
    SELECT
        tract, population - stayed AS departures,
        {', '.join(p % (c, c, c) for c in columns)}
    FROM last JOIN this USING (tract);
    '''
    # Request the data.
    data = pd.read_sql(select, db).set_index('tract')
    # Fill empty and negative values with zero.
    data[data < 0] = 0
    data[data.isna()] = 0
    # Close the connection to the database.
    db.close()
    # Return the data.
    return data


def __get_destination_tracts(
    db_path: str,
    county_1: int,
    year: int,
    columns: List[str] = CHARACTERISTICS
) -> pd.DataFrame:
    '''
    Request the census tracts in the destination county in a year with the count
    and characteristics of arrivals. This assumes arrivals are representative of
    the destination tract in the current year.

    db_path (str): location of the database.
    county_1 (int): FIPS code of the destination county.
    year (int): time at which to simulate moves.
    columns (list): labels for profile characteristics.

    Return destination tracts with arrivals and characteristics (pd.DataFrame).
    '''
    # Ensure column labels are available.
    assert columns
    # Initialize a representation of the database.
    db = sqlite3.connect(db_path)
    # Write the select statement.
    select = f'''
    SELECT
        tract, moved_in_county, moved_cross_county, moved_cross_state,
        {', '.join(columns)}
    FROM profile
    WHERE tract / 1000000 = {county_1} AND year = {year};
    '''
    # Request the data.
    data = pd.read_sql(select, db).set_index('tract')
    # Fill empty and negative values with zero.
    data[data < 0] = 0
    data[data.isna()] = 0
    # Close the connection to the database.
    db.close()
    # Return the data.
    return data


def __get_move_type(
    county_0: int,
    county_1: int
) -> str:
    '''
    Compare FIPS codes of two counties to determine the move type, whether
    'moved_in_county', 'moved_cross_county', or 'moved_cross_state'. These
    labels correspond to the count of arrivals in the profile characteristics.

    county_0, county_1 (int): FIPS codes of two counties.

    Return the move type (str).
    '''
    # Check whether the counties are the same.
    if county_0 == county_1:
        return 'moved_in_county'
    # Check whether the counties are in the same state.
    if county_0 // 1000 == county_1 // 1000:
        return 'moved_cross_county'
    # The counties are in different states otherwise.
    return 'moved_cross_state'


def __get_move_count(
    db_path: str,
    county_0: int,
    county_1: int,
    year: int,
    move_t: str
) -> int:
    '''
    Request number of moves from the origin county to the destination county in
    a year. This implementation weights the number of arrivals of the
    corresponding move type in the destination county per the profile data by
    the number of moves from the origin county to the destination county per the
    flow data.

    db_path (str): location of the database.
    county_0 (int): FIPS code of the origin county.
    county_1 (int): FIPS code of the destination county.
    year (int): time at which to simulate moves.

    Return non-negative number of moves (int).
    '''
    # Get the count from the origin to the desintation county per flow data.
    n = __get_county_to_county_moves(db_path, county_0, county_1, year)
    if n <= 0:
        return 0
    # Get the count of arrivals per flow data.
    f = __get_flow_arrivals(db_path, county_1, year, move_t)
    if f <= 0:
        return 0
    # Get the count of arrivals per profile data.
    p = __get_profile_arrivals(db_path, county_1, year, move_t)
    if p <= 0:
        return 0
    # Return the weighted number of moves.
    return int(n / f * p)


def __get_county_to_county_moves(
    db_path: str,
    county_0: int,
    county_1: int,
    year: int
) -> int:
    '''
    Request the number of moves from an origin county to a destination county in
    a year per the flow data.

    db_path (str): location of the database.
    county_0 (int): FIPS code of the origin county.
    county_1 (int): FIPS code of the destination county.
    year (int): time at which to find observations.

    Return number of moves (int).
    '''
    # Write the select statement.
    select = f'''
    SELECT flow
    FROM flow
    WHERE county = {county_1}
    AND county_last_year = {county_0}
    AND year = {year};
    '''
    # Execute and collect the result of the query.
    return __get_one(db_path, select, int)


def __get_flow_arrivals(
    db_path: str,
    county_1: int,
    year: int,
    arrivals: str
) -> int:
    '''
    Request the number of arrivals to a county in a year per the flow data.
    NB: in-county migration does not exist in the flow data.

    db_path (str): location of the database.
    county_1 (int): FIPS code of the destination county.
    year (int): time at which to find observations.
    arrivals (str): label that identifies the arrival population.

    Return number of arrivals (int).
    '''
    # Write the predicate per the move type.
    if arrivals == 'moved_in_county':
        return 0
    elif arrivals == 'moved_cross_county':
        p = '/ 1000 = %d' % (county_1 / 1000)
    elif arrivals == 'moved_cross_state':
        p = '/ 1000 != %d' % (county_1 / 1000)
    # Write the select statement.
    select = f'''
    SELECT SUM(flow)
    FROM flow
    WHERE county = {county_1}
    AND county_last_year {p}
    AND year = {year};
    '''
    # Execute and collect the result of the query.
    return __get_one(db_path, select, int)


def __get_profile_arrivals(
    db_path: str,
    county_1: int,
    year: int,
    arrivals: str
) -> int:
    '''
    Request the number of arrivals to a county in a year per the flow data.

    db_path (str): location of the database.
    county_1 (int): FIPS code of the destination county.
    year (int): time at which to find observations.
    arrivals (str): label that identifies the arrival population.

    Return number of arrivals (int).
    '''
    # Write the select statement.
    select = f'''
    SELECT SUM({arrivals})
    FROM profile
    WHERE tract / 1000000 = {county_1}
    AND year = {year};
    '''
    # Execute and collect the result of the query.
    return __get_one(db_path, select, int)


def __get_one(
    db_path: str,
    query: str,
    dtype: Any
) -> Any:
    '''
    Query a database for one result and cast as the datatype.

    db_path (str): location of the database.
    query (str): select statement to execute against the databse.
    dtype (type): datatype expected for the result.

    Return the result of the query.
    '''
    # Initialize a representation of the database.
    db = sqlite3.connect(db_path).cursor()
    # Execute and collect the result of the query.
    r = db.execute(query).fetchone()
    # Close the connection to the database.
    db.connection.close()
    # Return the default datatype value if no result exists.
    if not r:
        return dtype()
    # Return the result cast as the datatype otherwise.
    return dtype(r[0])


def __estimate_move_count(
    db_path: str,
    county_0: int,
    county_1: int,
    year: int,
    order: Tuple[int]
) -> int:
    '''
    Estimate the number of moves from the origin county to the destination 
    county for a year.

    db_path (str): location of the database.
    county_0 (int): FIPS code of the origin county.
    county_1 (int): FIPS code of the destination county.
    year (int): time at which to simulate moves.
    order (tuple): paramterization of the ARIMA model.

    Return estimated number of moves (int).
    '''
    # Initialize a representation of the database.
    db = sqlite3.connect(db_path).cursor()
    # Write the select statement.
    select = f'''
    SELECT year, flow AS moves
    FROM flow
    WHERE county = {county_1}
    AND county_last_year = {county_0}
    AND year < {year}
    ORDER BY YEAR ASC;
    '''
    # Execute and collect the result of the query.
    data = db.execute(select).fetchall()
    # Close the connection to the database.
    db.connection.close()
    # Return zero if there are no moves to model.
    if not data:
        return 0
    # Cast the data as an array.
    data = np.array(data)
    # Fit the moves to an ARIMA process per parameterization.
    model = ARIMA(data[:, 1], order=order).fit()
    # Forecast the number of moves in the year requested.
    i = year - data[-1, 0]
    n = model.forecast(i)[-1]
    # Return the number of moves
    return int(n)


def __simulate_moves(
    tracts_0: pd.DataFrame,
    tracts_1: pd.DataFrame,
    arrivals: str,
    n: int,
    columns: List[str] = CHARACTERISTICS
) -> Iterator[Tuple[int, int]]:
    '''
    Simulate moves between census tracts given the count observed between their
    respective counties as origin-destination tract pairs.
    
    tracts_0, tracts_1 (pd.DataFrame): collections of origins and destinations.
    arrivals (str): label that identifies the arrival population.
    n (int): number of moves to simulate.
    columns (list): labels for profile characteristics to calculate weights.

    Return origin-destination census tract pairs (list of tuples).
    '''
    # Ensure there are moves to simulate.
    assert n
    # Calculate move probabilities.
    p_depart_0 = tracts_0['departures'] / tracts_0['departures'].sum()
    p_arrive_1 = tracts_1[arrivals] / tracts_1[arrivals].sum()
    # Calculate similarities between origin and destination tracts.
    p_weights = __calculate_weights(tracts_0[columns], tracts_1[columns])
    # Generate and draw from the pmf of moves.
    moves_pmf = __make_moves_pmf(p_depart_0, p_arrive_1, p_weights)
    moves_sim = __draw_moves_pmf(moves_pmf, n)
    # Yield these moves.
    for move in moves_sim:
        yield tuple(move)


def __calculate_weights(
    tracts_0: pd.DataFrame,
    tracts_1: pd.DataFrame
) -> pd.DataFrame:
    '''
    Calculate weights for each origin with respect to each destination census
    tract as the distance between their characteristics.

    tracts_0, tracts_1 (pd.DataFrame): collections of origins and destinations.

    Return weights (pd.DataFrame).
    '''
    # Ensure equivalence of covariates in collections of tracts.
    assert set(tracts_0.columns) == set(tracts_1.columns)
    # Initialize a container for the weights.
    weights = np.zeros(shape=(tracts_0.shape[0], tracts_1.shape[0]))
    weights = pd.DataFrame(weights, index=tracts_0.index, columns=tracts_1.index)
    # Consider each origin, destination tract pair.
    for i in tracts_0.index:
        for j in tracts_1.index:
            # Calculate the distance between their characteristics.
            w = np.linalg.norm(tracts_1.loc[j] - tracts_0.loc[i])
            # Register this weight with the matrix of weights.
            weights.loc[i, j] = w
    # Normalize such that the shortest distance has the strongest weight.
    weights -= weights.max().max()
    weights /= weights.sum().sum()
    # Return the normalized weights.
    return weights


def __make_moves_pmf(
    p_depart_0: pd.DataFrame,
    p_arrive_1: pd.DataFrame,
    p_weights: pd.DataFrame
) -> np.ndarray:
    '''
    Generate a probability mass function of moves between each origin and each
    destination census tract per probability of departure and arrival as well as
    some weight. Order the pmf by probability.

    p_depart_0 (pd.DataFrame): probability of departure from each origin tract.
    p_arrive_1 (pd.DataFrame): probability of arrival to each destination tract.
    p_weights (pd.DataFrame): weights for each origin, destination tract pair.

    Return pmf (np.ndarray).
    '''
    # Initialize a container for the pmf.
    pmf = np.zeros(shape=(p_depart_0.shape[0] * p_arrive_1.shape[0], 3))
    # Initialize a counter to access each index in the pmf.
    k = 0
    # Consider each origin, destination tract pair.
    for i in p_depart_0.index:
        for j in p_arrive_1.index:
            # Calculate the draft probability of this pair.
            p = p_depart_0.loc[i] * p_arrive_1.loc[j] * p_weights.loc[i, j]
            # Register this probability in the function.
            pmf[k, :] = np.array([p, i, j])
            # Increment the counter.
            k += 1
    # Ensure the counter reached the end of the pmf.
    assert k == pmf.shape[0]
    # Sort the probabilities, careful to maintain correspondence with tract pairs.
    pmf = pmf[pmf[:, 0].argsort()]
    # Calculate the cumulative sum of normalized probabilities.
    pmf[:, 0] = np.cumsum(pmf[:, 0] / pmf[:, 0].sum())
    # Return the pmf.
    return pmf


def __draw_moves_pmf(
    moves_pmf: np.ndarray,
    n: int
) -> np.ndarray:
    '''
    Draw from a probability mass function of moves.

    moves_pmf (np.ndarray): p x 3 matrix of cumulative probability, tract pairs.
    n (int): number of draws to make of the target distribution.

    Return n origin, destination tract pairs.
    '''
    # Initialize container for draws.
    draws = np.zeros(shape=(n, 2), dtype=int)
    # Make n draws of the distribution.
    for i in range(n):
        # Make a draw of the uniform distribution.
        u = np.random.random()
        # Identify this location in the target distribution.
        for j in range(moves_pmf.shape[0]):
            # Check whether this probability exceeds the draw.
            if moves_pmf[j, 0] > u:
                # Draw this tract pair.
                draw = moves_pmf[j, 1:]
                # Register this draw with the matrix of draws and stop.
                draws[i, :] = draw
                break
    # Return the draws.
    return draws


def run(args):
    # Set the seed.
    np.random.seed(args.seed)
    # Ensure a year exists.
    assert args.year
    # Load all counties where the origin or destination does not exist.
    if not args.counties_0:
        args.counties_0 = __get_counties(args.db_path)
        if not args.counties_1:
            args.counties_1 = args.counties_0
    elif not args.counties_1:
        args.counties_1 = __get_counties(args.db_path)
    # Ensure valid FIPS codes.
    assert all(__validate_county(c) for c in args.counties_0)
    assert all(__validate_county(c) for c in args.counties_1)
    # Run the simulation.
    moves = simulate_moves(
        args.db_path, args.counties_0, args.counties_1, args.year)
    # Print the header row of the output.
    print(*SIMULATION_DATA_LABLES, sep='\t')
    # Report each result.
    for move in moves:
        print(args.year, *move, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate tract-to-tract moves from county-to-county counts.'
    )
    parser.add_argument(
        '--db_path',
        default=DEFAULT_DB_PATH,
        type=str,
        help='Location of the database with the ACS flow and profile data.',
        dest='db_path'
    )
    parser.add_argument(
        '--from',
        type=int,
        nargs=1,
        help='FIPS code of origin county.',
        dest='counties_0'
    )
    parser.add_argument(
        '--to',
        type=int,
        nargs=1,
        help='FIPS code of desintation county.',
        dest='counties_1'
    )
    parser.add_argument(
        '--in',
        type=int,
        help='Year at which to simulate moves.',
        dest='year'
    )
    parser.add_argument(
        '--seed',
        default=DEFAULT_SEED,
        type=int,
        help='Starting position for the random number generator.',
        dest='seed'
    )
    run(parser.parse_args())
