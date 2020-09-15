# Maine migration
Patrick Lavallee Delgado \
September 2020

## Overview

This project attempts to simulate tract-to-tract move counts from estimated county-to-county quantities in the American Community Survey, five-year migration flows, by the US Census Bureau. The analysis considers migration in an out of southern Maine, but the approach is applicable to any geography.

## Southern Maine

The underlying assumption of this simulation is that individuals move to sort themselves among people more like themselves and less like their neighbors. This is precisely why census tracts, which are drawn for homogeneity, are more desirable than counties, which have arbitrary bounds, as the unit of analysis.

1. Why is tract-level knowledge important?
2. From where to people arrive?
3. To where do they depart?
4. Who is from away? Are we changing?

## Implementation

I developed this project using Python 3.7.4 and it requires the standard library, `numpy` 1.16.4, `pandas` 1.0.5, and any dependencies.

The simulation runs on the command line and writes origin-destination tract pairs to standard output. It accepts arguments for origin county FIPS code `--from`, destination county FIPS code `--to`, and year `--in` in the data. Neither county FIPS code is required.

The `--db_path` argument gives the location of the database of ACS data, which by default is at `acs/acs.sqlite3`, and the `--seed` arguments sets the starting position of the random number generator.

The snippet below runs the simulation for tract-to-tract moves from all counties to Cumberland County, Maine, observed in 2015 and redirects the output:

```
$ python3 run_simulation.py --to 23005 --in 2015 > out
```

### Data

The data is from the American Community Survey, five-year estimates, from the US Census Bureau. There are two datasets: county-to-county migration flows and tract-level profile characteristics. I use the migration flows data to set the number of moves to simulate between counties, and the profile characteristics data to assign origin-destination tract pairs to each move. It is important to emphasize that these data are period estimates collected over five years, which, while not exact to the year, do offer complete geographic coverage.

I download these data from their API and store in a SQLite database. Please see my [get-census](https://github.com/lavalleedelgado/get-census) repository for the complete implementation. The YAML files in the `acs/` directory prescribe the data to collect.

### Simulation

The program `simulate_moves.py` (1) constructs a probability mass function that describes the probability of a move from each tract in the origin county to each tract in the destination county and (2) makes as many draws from the distribution as moves observed in the data. The probability of an origin-destination tract pair is the product of the corresponding departure-arrival events and some weight that expresses the similarity of those tracts.

A tract with more departures/arrivals receives a higher probability of departure/arrival. I estimate departures from a tract as the difference between (1) the number of people in the current year who lived in the same residence one year previous and (2) the number of people over one year old in the previous year. The data identify arrivals to a tract from the same county, a different county in the same state, or a different state. I choose the population of arrivals that matches the origin county. The data account for people who lived abroad one year previous but I do not simulate those moves.

Tract pairs for which the characteristics of departures are more similar to those of arrivals receive higher weights. I calculate similarity as the Euclidean distance between vectors that describe departures/arrivals by percent white, black, Hispanic, Asian, bachelor's degree attainment. I do not use median income because, unlike other measures, it is not estimable without a reference distribution. The equations below show how to decompose a tract characteristic `c` at time step `t` to describe subpopulations of departures, arrivals, remainers:

```
c_{t-1}(departures + remainers) = c_{d}departures + c_{r}remainers
c_{t}(remainers + arrivals) = c_{r}remainers + c_{a}arrivals

c_{t}(departures + remainers) - c_{d}departures = c_{t+1}(remainers + arrivals) - c_{a}arrivals
```

This gives one equation with two unknowns, each for the coefficient on departures `c_{d}` and arrivals `c_{a}`. To keep the estimation tractable, I assume arrivals share the characteristics of remainers and ignore that the remainers changes between time steps. Departures are representative of change in the origin tract from the previous year. Arrivals are representative of the destination tract in the current year.

There are cases when the coefficient on departures is negative and reveals information on the coefficient on arrivals. I update the corresponding characteristic for departures to zero, but I do not update that for arrivals because it would seem to misguide the vector of characteristics and favor selection of tract pairs on a lumpy mixture of assumptions.

I want to emphasize the two critical assumptions: (1) a tract is a more likely destination when it is more similar to the change in the tract from which the mover originates and (2) departures/arrivals are representative of their origins/destinations. Also, and because departures from and arrivals to tracts are known, the simulation only considers neighborhood choice and does not model the move decision. That is beyond the scope of intention for this project, but I do share my thoughts in the next section.

This approach constructs the entire probability mass function to describe a move as one decision instead of as two disjoint events. I think this is a more natural interpretation. But also, selecting tract pairs in two draws would oversample origins/destinations with more departures/arrivals, whichever draws first. Instead, the pmf has many discrete events that the probability of any tract pair is very small. I sort the probabilities such that a random draw of the distribution is from a region of similarly likely possibilities.

## Future work

This project neither models the move decision nor forecasts move counts in unobserved years. This is challenging because the time series is zero-inflated count data with just nine observations. A defendable data generating process would also include other predictors such as birth, death, and workforce indicators that tend to have better coverage at the county level. So, I would model tract-level population and county-to-county move counts and repeat the present work to simulate the tract-to-tract quantities for future years. It would also be interesting to simulate a joint distribution of mover occupation and demographic profile.


<!-- 
This portfolio explores whether and why people migrate into, out from, and around Maine. I use data from the American Community Survey, 5-year estimate, (ACS) by the US Census Bureau and from the Quarterly Census of Employment & Wages (QCEW) by the Bureau of Labor Statistics to map and analyze these trends. These data reveal to where Mainers move and identify the local characteristics that might continue to encourage migration.

## Research questions
- Is it safe to compare ACS 5-year estimates of adjacent years?
- How many Mainers by county relocate out of state?
- How many new Mainers by county relocate to the state?
- Between which counties do Mainers move?
- Where does the population consolidate?
    - cluster tracts into population centers
    - identify a stopping condition: population and physical size?
    - identify "disenfranchised" or isolated tracts
    - allow membership in multiple clusters
- How do these cluster characteristics change with migration?
    - demographic composition
    - educational attainment
    - median household income
    - median age
    - population density
    - access to broadband
    - access to healthcare
- How do average tradable and local wages change?
- How do health and education change?
- How does voting behavior change? (voting, op-ed sentiment, social media?)
- Does migration in one year predict these outcomes in another?
- How do tracts compare by cluster membership?
    - northern Maine versus southern Maine
    - clustered versus not clustered

## Policy questions
- What are the top line consequences of migration?
- How can policy avert worse outcomes in areas that migration affects?

## Next steps
1. Download data for each year from 2000 through 2018.
2. Join data in a database.
3. Draw visualizations:
    - map of county-level migration
    - map of clusters
    - map of cluster-level population change
    - correlations between migrations and cluster characteristics
4. Find other interesting data:
    - road maps to measure time and distance to populations
    - migration between clusters instead of just counties
    - birth and death statistics to subtract from cluster population change
    - health outcomes, insurance claims, or spending generally
    - voter registration and election outcomes

## Get more resources
Maine DOL
John Doerr
Maine & Co.
Live and Work in Maine
Food and aquaculture industry growth
Northern Maine Development Commission
Economic Development Administration
- Denise Garland

Camoine, Jim Demesis (sp?)
Peter Steele
The Beacon Group

** Start with John Doerr
** Maine Department of Community Development

Get foot in the door with economic development company!
Maine Technology Institute for jobs.
 -->
