import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from itertools import combinations
import statistics


cg_data = pd.read_csv('groups.tsv', delimiter='\t')
isolated_galaxies = pd.read_csv('isolated_galaxies.tsv', delimiter='\t')
galaxies_morphology = pd.read_csv('galaxies_morphology.tsv', delimiter='\t')
cg_data.dropna(inplace=True)
galaxy_coordinates = pd.read_csv('galaxies_coordinates.tsv', delimiter='\t')


# initializing ACMD cosmology model wit hthe given parameters
my_cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
# merging the two datasets that need to be used
df = cg_data.merge(galaxy_coordinates)
# dropping unnecessary columns
df.drop(columns=['features'], inplace=True)
# converting it to table to make it easier to use astropy units
table = Table.from_pandas(df)
# calculating angular diameter in kiloparsecs using median redshifts using the cosmology model
table['angular diameter'] = my_cosmo.angular_diameter_distance(table['z']).to(u.kpc)
# getting the proper coordinates of each galaxy
table['skycoord'] = SkyCoord(ra=table['RA'] * u.degree, dec=table['DEC'] * u.degree, frame='fk5')
# creating a list of groups to iterate through without repeating when there are multiple galaxies in a group
groups = []
[groups.append(x) for x in (df['Group'].to_list()) if x not in groups]
# empty table that will be filled in with the necessary data
table2 = Table(names=('Group', 'angular diameter', 'angular distance', 'physical distance'))
# iterating through each group
for i in groups:
    # creates a table with the galaxies that are within the specified group
    holder = table[table['Group'] == i]
    # getting all possible combinations of the distance between galaxies in order to find the median value for each
    # group
    sep = combinations(holder['skycoord'], 2)
    find_median = []
    for x in sep:
        # getting the angular distance
        find_median.append(x[0].separation(x[1]))
    # adding the group number, median angular diameter, median angular distance, and physical distance
    # median physical distance = median angular diameter * (median angular distance in radians)
    median_diameter = statistics.median(holder['angular diameter'])
    median_distance = statistics.median(find_median)
    table2.add_row((int(i.split()[1]), median_diameter, median_distance,
                    median_diameter * median_distance.to(u.rad)))
# creating a scatter plot for the two values
plt.scatter(table2['angular diameter'], table2['physical distance'])
# plt.show()
# finding p-values
shapiro = stats.shapiro(cg_data['mean_mu'])
shapiro2 = stats.shapiro(table2['physical distance'])
pearson = stats.pearsonr(cg_data['mean_mu'], table2['physical distance'])
print('76.98939987756634', shapiro2[1], shapiro[1], pearson[1])
