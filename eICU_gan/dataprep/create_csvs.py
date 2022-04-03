"""
# Create dataframes script

# you will need to alter the SAVE_LOCATION if you want to save somewhere other than ./../data directory
"""

import pandas as pd
import numpy as np
import os

# origin workbooks
ORIG_LOCATION = './../data/orig'

# csvs will be saved to SAVE_LOCATION + '/clean/' and SAVE_LOCATION + '/cleanpivots/'
SAVE_LOCATION = './../data'

# initializing the Dict of dataframe descriptions and dataframes, will use for looping through dataframes for saving to files
dataframes_dict = {}

# creating the % of Hampton Roads and % of Virginia values for each year
def AddHamptonAndVirginiaPercentages(df, year_lb, year_ub, row_lb, row_ub):
    """
    creating columns % Hampton Roads, % of Virginia values for each year - the input and output dataframes are NOT pivoted

    :param df: Required, unpivoted dataframe of jurisdictions (rows) and years (columns)
    
    :param year_lb: Required, lower bound of the year column
    
    :param year_ub: Required, upper bound (not inclusive) of the year column
    
    :param row_lb: Required, row lower bound index for individual hampton roads jurisdictions (not including aggregate jurisdictions: e.g. Hampton Roads, Virginia, US)
    
    :param row_ub: Required, row upper bound (not inclusive) of the individual jurisdiction rows
    
    :return: dataframe with the perc_Virginia and perc_Hampton columns added
    """

    i_insert = 2

    for y in range(year_lb,year_ub):
        # for each year add a column of the Jurisdiction # / Hampton #
        lst_year_Hampton = []
        # for each year add a column of the Jurisdiction # / Virginia #
        lst_year_Virginia = []
        for r in range(row_lb,row_ub):
            j_value = df.loc[r][y]
            #print('j_value: ',j_value)
            h_value = df.loc[row_ub][y]
            #print('h_value: ',h_value)
            v_value = df.loc[row_ub+1][y]
            #print('v_value: ',v_value)

            lst_year_Hampton.append(j_value / h_value)
            lst_year_Virginia.append(j_value / v_value)

        # add values for the Hampton and Virginia rows, so we don't get an error on the insert
        lst_year_Hampton.extend([1.,1.])
        lst_year_Virginia.extend([1.,1.])

        df.insert(loc=(i_insert), column=str(y) + '_Perc_Virginia', value=lst_year_Virginia)
        df.insert(loc=(i_insert), column=str(y) + '_Perc_Hampton', value=lst_year_Hampton)

        # index to insert increases by three each time, because we are moving over 1 year, but have also added two new columns for %Hampton Roads and %VirginiaBeach
        i_insert += 3
    
    # remove the last two rows: Hampton and Virginia Beach
    df = df.iloc[:-2]
    
    return df


def AddHamptonAndVirginiaAndUSCompare(df, year_lb, year_ub, row_lb, row_ub):
    """
    creating columns Ratio of Hampton Roads PDC, Hampton Roads MSA, Virginia, US Metro, and US values for each year - the input and output dataframes are NOT pivoted

    :param df: Required, unpivoted dataframe of jurisdictions (rows) and years (columns)
    
    :param year_lb: Required, lower bound of the year column
    
    :param year_ub: Required, upper bound (not inclusive) of the year column
    
    :param row_lb: Required, row lower bound index for individual hampton roads jurisdictions (not including aggregate jurisdictions: e.g. Hampton Roads, Virginia, US)
    
    :param row_ub: Required, row upper bound (not inclusive) of the individual jurisdiction rows
    
    :return: dataframe with the _Of_Hampton_PDC, _Of_Hampton_MSA, _Of_Virginia, _Of_US_Metrom and _Of_US columns added
    """

    i_insert = 2

    for y in range(year_lb,year_ub):
        # initialize the ratio lists
        lst_year_Hampton_PDC = []
        lst_year_Hampton_MSA = []
        lst_year_Virginia = []
        lst_year_US_Metro = []
        lst_year_US = []
        
        for r in range(row_lb,row_ub):
            j_value = df.loc[r][y]
            h_pdc_value = df.loc[row_ub][y]
            h_msa_value = df.loc[row_ub + 1][y]
            v_value = df.loc[row_ub+2][y]
            us_metro_value = df.loc[row_ub+3][y]
            us_value = df.loc[row_ub+4][y]

            lst_year_Hampton_PDC.append(j_value / h_pdc_value)
            lst_year_Hampton_MSA.append(j_value / h_msa_value)
            lst_year_Virginia.append(j_value / v_value)
            lst_year_US_Metro.append(j_value / us_metro_value)
            lst_year_US.append(j_value / us_value)

        # add values for the aggregate rows, so we don't get an error on the insert
        lst_year_Hampton_PDC.extend([1.,1.,1.,1.,1.])
        lst_year_Hampton_MSA.extend([1.,1.,1.,1.,1.])
        lst_year_Virginia.extend([1.,1.,1.,1.,1.])
        lst_year_US_Metro.extend([1.,1.,1.,1.,1.])
        lst_year_US.extend([1.,1.,1.,1.,1.])

        df.insert(loc=(i_insert), column=str(y) + '_Of_US', value=lst_year_US)
        df.insert(loc=(i_insert), column=str(y) + '_Of_US_Metro', value=lst_year_US_Metro)
        df.insert(loc=(i_insert), column=str(y) + '_Of_Virginia', value=lst_year_Virginia)
        df.insert(loc=(i_insert), column=str(y) + '_Of_Hampton_MSA', value=lst_year_Hampton_MSA)
        df.insert(loc=(i_insert), column=str(y) + '_Of_Hampton_PDC', value=lst_year_Hampton_PDC)

        # index to insert increases by three each time, because we are moving over 1 year, but have also added two new columns for %Hampton Roads and %VirginiaBeach
        i_insert += 6
    
    # remove the last five rows: teh aggregate rows: Hampton PDC, Hampton MSA, Virginia, US Metro, US
    #df = df.iloc[:-5]
    
    return df

# creating the % of PDC Hampton Roads, % of MSA Hampton Roads,% of Virginia, and % US values for each year
def AddHamptonAndVirginiaAndUSCompareLaborForce(df, year_lb, year_ub, row_lb, row_ub):
    """
    creating columns Ratio of Hampton Roads PDC, Hampton Roads MSA, Virginia, and US values for each year - the input and output dataframes are NOT pivoted

    :param df: Required, unpivoted dataframe of jurisdictions (rows) and years (columns)
    
    :param year_lb: Required, lower bound of the year column
    
    :param year_ub: Required, upper bound (not inclusive) of the year column
    
    :param row_lb: Required, row lower bound index for individual hampton roads jurisdictions (not including aggregate jurisdictions: e.g. Hampton Roads, Virginia, US)
    
    :param row_ub: Required, row upper bound (not inclusive) of the individual jurisdiction rows
    
    :return: dataframe with the _Of_Hampton_PDC, _Of_Hampton_MSA, _Of_Virginia, _Of_US_Metrom and _Of_US columns added
    """
    
    i_insert = 2

    for y in range(year_lb,year_ub):
        # initialize the ratio lists
        lst_year_Hampton_PDC = []
        lst_year_Hampton_MSA = []
        lst_year_Virginia = []
        lst_year_US = []
        
        for r in range(row_lb,row_ub):
            j_value = df.loc[r][y]
            h_pdc_value = df.loc[row_ub][y]
            h_msa_value = df.loc[row_ub + 1][y]
            v_value = df.loc[row_ub+2][y]
            us_value = df.loc[row_ub+3][y]

            lst_year_Hampton_PDC.append(j_value / h_pdc_value)
            lst_year_Hampton_MSA.append(j_value / h_msa_value)
            lst_year_Virginia.append(j_value / v_value)
            lst_year_US.append(j_value / us_value)

        # add values for the aggregate rows, so we don't get an error on the insert
        lst_year_Hampton_PDC.extend([1.,1.,1.,1.])
        lst_year_Hampton_MSA.extend([1.,1.,1.,1.])
        lst_year_Virginia.extend([1.,1.,1.,1.])
        lst_year_US.extend([1.,1.,1.,1.])

        df.insert(loc=(i_insert), column=str(y) + '_Of_US', value=lst_year_US)
        df.insert(loc=(i_insert), column=str(y) + '_Of_Virginia', value=lst_year_Virginia)
        df.insert(loc=(i_insert), column=str(y) + '_Of_Hampton_MSA', value=lst_year_Hampton_MSA)
        df.insert(loc=(i_insert), column=str(y) + '_Of_Hampton_PDC', value=lst_year_Hampton_PDC)

        # index to insert increases by three each time, because we are moving over 1 year, but have also added two new columns for %Hampton Roads and %VirginiaBeach
        i_insert += 5
    
    # remove the last five rows: teh aggregate rows: Hampton PDC, Hampton MSA, Virginia, US Metro, US
    #df = df.iloc[:-4]
    
    return df

def createPivotVersion(df, year_lb, year_ub, val_list, piv_col_list):
    """
    manually pivoting the dataframe on year (could have used pandas, but this works)

    :param df: Required, unpivoted dataframe of jurisdictions (rows) and years (columns)
    
    :param year_lb: Required, lower bound of the year column
    
    :param year_ub: Required, upper bound (not inclusive) of the year column
    
    :param piv_col_list: list of columns to pivot
    
    :return: dataframe with the _Of_Hampton_PDC, _Of_Hampton_MSA, _Of_Virginia, _Of_US_Metrom and _Of_US columns added
    """
        
    df_piv = pd.DataFrame()
    cart_jur_year = [[b,a] for b in range(year_lb,year_ub) for a in df['Jurisdiction']]

    df_piv = pd.DataFrame(cart_jur_year, columns=['year', 'jurisdiction'])
    
    for val, col in zip(val_list,piv_col_list):
        lst_vals = []
        for year in range(year_lb,year_ub):
            lst_vals = np.append(lst_vals,df[str(year) + str(val)].to_numpy())
        df_piv[col] = lst_vals
    
    return df_piv


# for each dataframe change the year column names to have a suffix to denote which data table the data came from
# Jurisdiction needs to remain the same between dataframes for joining
def getNewColumnNames(df,suffix):
    """
    All the starting xlsx's had column names of [YEAR], in the new version we need columns that describe the value, e.g. 'pop' for population. This function adds _[suffix] to the column name

    :param df: Required, unpivoted dataframe of jurisdictions (rows) and years (columns)
    
    :param suffix: Suffix to add to column name
    
    :return: dataframe with suffix added to NOT JURISDICTION columns
    """
    
    cols = df.columns
    cols = [str(c) + '_' + suffix for c in cols[1:]]
    lst_cols = ['Jurisdiction']
    lst_cols.extend(cols)
    return lst_cols

def getStartingColumnNames(yr_lb = 1970, yr_ub = 2016):
    """
    Silly helper function to get the current list of columns that start with 'Jurisdiction' years within the yr_lb - yr_ub (not inclusive range)
    
    :return: array of column names starting with ['Jurisdiction']
    """
    
    # Jurisdiction is the standard column for all the workbooks
    lst_cols = ['Jurisdiction']
    # this is the current list of years where there is data
    lst_years =  list(range(yr_lb,yr_ub))
    # this is the total list of columns
    lst_cols.extend(lst_years)
    
    return lst_cols


def combineJurisdictions(df_16j, 
                        jurisdictions_to_remove= ['Franklin','Southampton County','James City County','Williamsburg','Poquoson','York County'],
                        j_merge_lst = [['Franklin','Southampton County'], ['James City County','Williamsburg'], ['Poquoson','York County']],
                        j_new_name = ['Franklin(City) & Southampton','James City & Williamsburg','Poquoson & York'],
                        sum_or_avg = 'sum'):

    """
    combineJurisdictions()

    Some worksheets have 13 Jurisdictions, others have 16. Luckily the same jurisdictions are combined to make the 13 jurisdictions.

    'Franklin' and 'Southampton County' into 'Franklin(City) & Southampton'
    'James City County' and 'Williamsburg' into 'James City & Williamsburg'
    'Poquoson' and 'York County' into 'Poquoson & York'

    0: 1_1_Population (16, 139)
    1: 1_5_Annual_Pop_Change (16, 47)
    2: 1_7_Birth_Rate (16, 47)
    3: 1_9_Death_Rate (16, 47)
    4: 2_1_Income (13, 277)
    5: 2_4_Total_Income (13, 277)
    6: 3_2_Employment (13, 277)
    7: 3_3_Employment_Military (13, 277)
    8: 4_2_LaborForce (16, 141)
    9: 4_4_Unemployed (16, 141)
    10: 4_5_Unemployment_Rate (16, 29)
    11: 4_6_Employment_Pop_Ratio (16, 29)
    """
    
    # remove the rows we don't want from the 16 jurisdiction dataframes
    df_13j = df_16j[~df_16j['Jurisdiction'].isin(jurisdictions_to_remove)]

    # merge the values by summing each cell
    for merges, new_name in zip(j_merge_lst, j_new_name):
        merge1_name = merges[0]
        merge2_name = merges[1]

        # get the two rows we want to merge
        vals1 = np.asarray(df_16j.query("Jurisdiction == '" + merge1_name + "'").iloc[0,1:])
        vals2 = np.asarray(df_16j.query("Jurisdiction == '" + merge2_name + "'").iloc[0,1:])

        # set the jurisdiction name
        new_row = []
        new_row.append(new_name)
        
        if sum_or_avg == 'sum':
            # sum the values
            new_row.extend(vals1 + vals2)
        else:
            # avg the values
            new_row.extend((vals1 + vals2) / 2)

        # append to the df_13j dataframe
        df_new = pd.DataFrame([new_row], columns = df_13j.columns)
        df_13j = pd.concat([df_13j,df_new], ignore_index=True)
   
    df_13j.reset_index(drop=True, inplace=True)   
    return df_13j

"""
## 1) Population Data Source: <https://www.hrpdcva.gov/page/data-book/>

### Worksheets in Population Workbook
Bold worksheets will be prepared.

1. **Population (July Estimates)**
    1. by jurisdiction (16 jurisdictions)
    1. 1960 - 2017
    1. includes rows that can be used for percentages of population of Hampton Roads and Virginia

1. **January to December Population Change**
    1. by jurisdiction (16 jurisdictions)
    1. 1961 - 2016

1. **Birth Rate (Per 1,000 Persons)**
    1. by jurisdiction (16 jurisdictions)
    1. 1970 - 2015

1. **Deaths Rate (Per 1,000 Persons)**
    1. by jurisdiction (16 jurisdictions)
    1. 1975 - 2015

"""
# get the starting column names
lst_cols = getStartingColumnNames()

# read from the starting xlsx workbooks
df_pop = pd.read_excel(ORIG_LOCATION + '/01_population.xlsx', sheet_name="1.1", header=3)[lst_cols]
df_annual_pop_change = pd.read_excel(ORIG_LOCATION + '/01_population.xlsx', sheet_name="1.5", header=3)[lst_cols]
df_br = pd.read_excel(ORIG_LOCATION + '/01_population.xlsx', sheet_name="1.7", header=3)[lst_cols]
df_dr = pd.read_excel(ORIG_LOCATION + '/01_population.xlsx', sheet_name="1.9", header=3)[lst_cols]

## POPULATION
# population: add percentages
df_pop = AddHamptonAndVirginiaPercentages(df_pop, 1970, 2016, 0, 16)
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_pop,'pop')
df_pop.columns = lst_cols
# combine jurisdictions to get same jurisdictions across data sets
df_pop = combineJurisdictions(df_pop)

# populations: create the pivot version
df_pop_piv = createPivotVersion(df_pop, 1970, 2016, ['_pop','_Perc_Hampton_pop','_Perc_Virginia_pop'], ['pop','pop_perc_Hampton','pop_perc_Virginia'])

## Annual Population Change
# annual population change: add percentages, rename columns, and merge jurisdictions
# remove the aggregate rows for Hampton Roads and Virginia, we aren't using for this dataset
df_annual_pop_change = df_annual_pop_change.iloc[:-2]
lst_cols = getNewColumnNames(df_annual_pop_change,'popchange')
df_annual_pop_change.columns = lst_cols
df_annual_pop_change = combineJurisdictions(df_annual_pop_change)

#annual population change: create the pivot version
df_annual_pop_change_piv = createPivotVersion(df_annual_pop_change, 1970, 2016, ['_popchange'], ['popchange'])

## BIRTH RATE
# birth rate: add percentages, rename columns, and merge jurisdictions
# remove the aggregate rows for Hampton Roads and Virginia, we aren't using for this dataset
df_br = df_br.iloc[:-2]
lst_cols = getNewColumnNames(df_br,'birthrate')
df_br.columns = lst_cols
df_br = combineJurisdictions(df_br, 
                        ['Franklin','Southampton County','James City County','Williamsburg','Poquoson','York County'],
                        [['Franklin','Southampton County'], ['James City County','Williamsburg'], ['Poquoson','York County']],
                        ['Franklin(City) & Southampton','James City & Williamsburg','Poquoson & York'],
                        'avg')

#birth rate: create the pivot version
df_br_piv = createPivotVersion(df_br, 1970, 2016, ['_birthrate'], ['birthrate'])

## DEATH RATE
# death rate: add percentages, rename columns, and merge jurisdictions
# remove the aggregate rows for Hampton Roads and Virginia, we aren't using for this dataset
df_dr = df_dr.iloc[:-2]
lst_cols = getNewColumnNames(df_dr,'deathrate')
df_dr.columns = lst_cols
df_dr = combineJurisdictions(df_dr, 
                        ['Franklin','Southampton County','James City County','Williamsburg','Poquoson','York County'],
                        [['Franklin','Southampton County'], ['James City County','Williamsburg'], ['Poquoson','York County']],
                        ['Franklin(City) & Southampton','James City & Williamsburg','Poquoson & York'],
                        'avg')

#death rate: create the pivot version
df_dr_piv = createPivotVersion(df_dr, 1970, 2016, ['_deathrate'], ['deathrate'])

"""
## 2) Income Earning <https://www.hrpdcva.gov/page/data-book/>

### Worksheets in Income Workbook

1. **Per Capita Income**
    1. by jurisdiction (13 jurisdictions)
    1. 1969 - 2016
    1. includes rows that can be used for ratios of jurisidiction value compared to Hampton Roads PDC (paper data capture), Hampton Roads MSA (metro statistical area), Virginia, US (Metro Portion), and United States
    
1. **Local Total Personal Income**
    1. by jurisdiction (13 jurisdictions)
    1. 1969 - 2016
    1. in Thousands of dollars
    1. includes rows that can be used for ratios of jurisidiction value compared to Hampton Roads PDC (paper data capture), Hampton Roads MSA (metro statistical area), Virginia, US (Metro Portion), and United States

"""

# get the starting column names
lst_cols = getStartingColumnNames()

df_income = pd.read_excel(ORIG_LOCATION + '/02_income_earnings.xlsx', sheet_name="2.1", header=3)[lst_cols]
df_total_income = pd.read_excel(ORIG_LOCATION + '/02_income_earnings.xlsx', sheet_name="2.4", header=3)[lst_cols]

## Income
# income: add percentages, rename columns
df_income = AddHamptonAndVirginiaAndUSCompare(df_income, 1970, 2016, 0, 13)
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_income,'income')
df_income.columns = lst_cols
# remove the aggregate rows after we have used them
df_income = df_income.iloc[:-5]

#income: create the pivot version
cols_orig = ['_income','_Of_Hampton_PDC_income','_Of_Hampton_MSA_income','_Of_Virginia_income','_Of_US_Metro_income','_Of_US_income']
cols_piv = ['income','income_ratio_Hampton_PDC','income_ratio_Hampton_MSA','income_ratio_Virginia','income_ratio_US_Metro','income_ratio_US']
df_income_piv = createPivotVersion(df_income, 1970, 2016, cols_orig, cols_piv)

## Total Income
# total income: add percentages, rename columns
df_total_income = AddHamptonAndVirginiaAndUSCompare(df_total_income, 1970, 2016, 0, 13)
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_total_income,'totalincome')
df_total_income.columns = lst_cols
# remove the aggregate rows after we have used them
df_total_income = df_total_income.iloc[:-5]

#total income: create the pivot version
cols_orig = ['_totalincome','_Of_Hampton_PDC_totalincome','_Of_Hampton_MSA_totalincome','_Of_Virginia_totalincome','_Of_US_Metro_totalincome','_Of_US_totalincome']
cols_piv = ['totalincome','totalincome_ratio_Hampton_PDC','totalincome_ratio_Hampton_MSA','totalincome_ratio_Virginia','totalincome_ratio_US_Metro','totalincome_ratio_US']
df_total_income_piv = createPivotVersion(df_total_income, 1970, 2016, cols_orig, cols_piv)

"""
## 3) Employment <https://www.hrpdcva.gov/page/data-book/>

### Worksheets in Employment Workbook

1. **Jurisdiction Total Employment**
    1. by jurisdiction (13 jurisdictions)
    1. 1969 - 2016
    1. includes rows that can be used for jurisidiction value comparison to Hampton Roads PDC (paper data capture), Hampton Roads MSA (metro statistical area), Virginia, US (Metro Portion), and United States
1. **Jurisdiction Military Employment**
    1. by jurisdiction (13 jurisdictions)
    1. 1969 - 2016
    1. includes rows that can be used for jurisidiction value comparison to Hampton Roads PDC (paper data capture), Hampton Roads MSA (metro statistical area), Virginia, US (Metro Portion), and United States
"""

# get the starting column names
lst_cols = getStartingColumnNames()

df_employment = pd.read_excel(ORIG_LOCATION + '/03_employment.xlsx', sheet_name="3.2", header=3)[lst_cols]
df_military_employment = pd.read_excel(ORIG_LOCATION + '/03_employment.xlsx', sheet_name="3.3", header=3)[lst_cols]

## Employment
# employment: add percentages, rename columns
df_employment = AddHamptonAndVirginiaAndUSCompare(df_employment, 1970, 2016, 0, 13)
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_employment,'emp')
df_employment.columns = lst_cols
# remove the aggregate rows after we have used them
df_employment = df_employment.iloc[:-5]

#employment: create the pivot version
cols_orig = ['_emp','_Of_Hampton_PDC_emp','_Of_Hampton_MSA_emp','_Of_Virginia_emp','_Of_US_Metro_emp','_Of_US_emp']
cols_piv = ['emp','emp_perc_Hampton_PDC','emp_perc_Hampton_MSA','emp_perc_Virginia','emp_perc_US_Metro','emp_perc_US']
df_employment_piv = createPivotVersion(df_employment, 1970, 2016, cols_orig, cols_piv)

## Military Employment
# military employment: add percentages, rename columns
df_military_employment = AddHamptonAndVirginiaAndUSCompare(df_military_employment, 1970, 2016, 0, 13)
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_military_employment,'emp_mil')
df_military_employment.columns = lst_cols
# remove the aggregate rows after we have used them
df_military_employment = df_military_employment.iloc[:-5]

#military employment: create the pivot version
cols_orig = ['_emp_mil','_Of_Hampton_PDC_emp_mil','_Of_Hampton_MSA_emp_mil','_Of_Virginia_emp_mil','_Of_US_Metro_emp_mil','_Of_US_emp_mil']
cols_piv = ['emp_mil','emp_mil_perc_Hampton_PDC','emp_mil_perc_Hampton_MSA','emp_mil_perc_Virginia','emp_mil_perc_US_Metro','emp_mil_perc_US']
df_military_employment_piv = createPivotVersion(df_military_employment, 1970, 2016, cols_orig, cols_piv)

"""
## 4) Unemployment <https://www.hrpdcva.gov/page/data-book/>

### Worksheets in Labor Force Unemployment Workbook

1. **Labor Force by Jurisdiction**
    1. by jurisdiction (16 jurisdictions)
    1. 1990 - 2017
    1. includes rows that can be used for jurisidiction value comparison to Hampton Roads PDC (paper data capture), Hampton Roads MSA (metro statistical area), and United States
1. **Number of Unemployed by Jurisdiction**
    1. by jurisdiction (16 jurisdictions)
    1. 1990 - 2017
    1. includes rows that can be used for jurisidiction value comparison to Hampton Roads PDC (paper data capture), Hampton Roads MSA (metro statistical area), Virginia, and United States
1. **Unemployment Rate by Jurisdiction**
    1. by jurisdiction (16 jurisdictions)
    1. 1990 - 2017
    1. includes rows that can be used for jurisidiction value comparison to Hampton Roads PDC (paper data capture), Hampton Roads MSA (metro statistical area), Virginia, and United States
1. **Employment to Population Ratio (Total Population)**
    1. by jurisdiction (16 jurisdictions)
    1. 1990 - 2017
    1. includes rows that can be used for jurisidiction value comparison to Hampton Roads PDC (paper data capture), Virginia, and United States

"""
# get the starting column names
lst_cols = getStartingColumnNames(1990, 2018)

df_laborforce = pd.read_excel(ORIG_LOCATION + '/04_laborforce_unemployment.xlsx', sheet_name="4.2", header=3)[lst_cols]
df_unemployed = pd.read_excel(ORIG_LOCATION + '/04_laborforce_unemployment.xlsx', sheet_name="4.4", header=3)[lst_cols]
df_unemploy_rate= pd.read_excel(ORIG_LOCATION + '/04_laborforce_unemployment.xlsx', sheet_name="4.5", header=3)[lst_cols]
df_employ_pop_ratio= pd.read_excel(ORIG_LOCATION + '/04_laborforce_unemployment.xlsx', sheet_name="4.6", header=3)[lst_cols]

## Laborforce
# laborforce: add percentages, rename columns, and combine jurisdictions
df_laborforce = AddHamptonAndVirginiaAndUSCompareLaborForce(df_laborforce, 1990, 2018, 0, 16)
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_laborforce,'laborforce')
df_laborforce.columns = lst_cols
# remove the aggregate rows after we have used them
df_laborforce = df_laborforce.iloc[:-4]
# combine jurisdictions
df_laborforce = combineJurisdictions(df_laborforce, ['Franklin','Southampton','James City','Williamsburg','Poquoson','York'], [['Franklin','Southampton'], ['James City','Williamsburg'], ['Poquoson','York']], ['Franklin(City) & Southampton','James City & Williamsburg','Poquoson & York'])

# laborforce: create the pivot version
cols_orig = ['_laborforce','_Of_Hampton_PDC_laborforce','_Of_Hampton_MSA_laborforce','_Of_Virginia_laborforce','_Of_US_laborforce']
cols_piv = ['laborforce','laborforce_perc_Hampton_PDC','laborforce_perc_Hampton_MSA','laborforce_perc_Virginia','laborforce_perc_US_Metro']
df_laborforce_piv = createPivotVersion(df_laborforce, 1990, 2018, cols_orig, cols_piv)


## Unemployed
# unemployed: add percentages, rename columns, and combine jurisdictions
df_unemployed = AddHamptonAndVirginiaAndUSCompareLaborForce(df_unemployed, 1990, 2018, 0, 16)
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_unemployed,'unemployed')
df_unemployed.columns = lst_cols
# remove the aggregate rows after we have used them
df_unemployed = df_unemployed.iloc[:-4]
# combine jurisdictions
df_unemployed = combineJurisdictions(df_unemployed, ['Franklin','Southampton','James City','Williamsburg','Poquoson','York'], [['Franklin','Southampton'], ['James City','Williamsburg'], ['Poquoson','York']], ['Franklin(City) & Southampton','James City & Williamsburg','Poquoson & York'])

# unemployed: create the pivot version
cols_orig = ['_unemployed','_Of_Hampton_PDC_unemployed','_Of_Hampton_MSA_unemployed','_Of_Virginia_unemployed','_Of_US_unemployed']
cols_piv = ['unemployed','unemployed_perc_Hampton_PDC','unemployed_perc_Hampton_MSA','unemployed_perc_Virginia','unemployed_perc_US_Metro']
df_unemployed_piv = createPivotVersion(df_unemployed, 1990, 2018, cols_orig, cols_piv)

## Unemployment Rate
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_unemploy_rate,'unemployed_rate')
df_unemploy_rate.columns = lst_cols
# remove the aggregate rows after we have used them
df_unemploy_rate = df_unemploy_rate.iloc[:-4]
# combine jurisdictions
df_unemploy_rate = combineJurisdictions(df_unemploy_rate, ['Franklin','Southampton','James City','Williamsburg','Poquoson','York'], [['Franklin','Southampton'], ['James City','Williamsburg'], ['Poquoson','York']], ['Franklin(City) & Southampton','James City & Williamsburg','Poquoson & York'], sum_or_avg='avg')

# unemployment Rate: create the pivot version
cols_orig = ['_unemployed_rate']
cols_piv = ['unemployed_rate']

df_unemploy_rate_piv = createPivotVersion(df_unemploy_rate, 1990, 2018, cols_orig, cols_piv)

## Employment Population Ratio
# set the column names, except Jurisidiction to [COLUMN_NAME]_pop
lst_cols = getNewColumnNames(df_employ_pop_ratio,'employ_pop_ratio')
df_employ_pop_ratio.columns = lst_cols
# remove the aggregate rows after we have used them
df_employ_pop_ratio = df_employ_pop_ratio.iloc[:-3]
# combine jurisdictions
df_employ_pop_ratio = combineJurisdictions(df_employ_pop_ratio, ['Franklin','Southampton','James City','Williamsburg','Poquoson','York'], [['Franklin','Southampton'], ['James City','Williamsburg'], ['Poquoson','York']], ['Franklin(City) & Southampton','James City & Williamsburg','Poquoson & York'], sum_or_avg='avg')

# employment population Ratio: create the pivot version
cols_orig = ['_employ_pop_ratio']
cols_piv = ['employ_pop_ratio']
df_employ_pop_ratio_piv = createPivotVersion(df_employ_pop_ratio, 1990, 2018, cols_orig, cols_piv)

"""
Save all the dataframes to csv's
"""
dataframes_dict['1_1_Population'] = df_pop
dataframes_dict['1_5_Annual_Pop_Change'] = df_annual_pop_change
dataframes_dict['1_7_Birth_Rate'] = df_br
dataframes_dict['1_9_Death_Rate'] = df_dr
dataframes_dict['2_1_Income'] = df_income
dataframes_dict['2_4_Total_Income'] = df_total_income
dataframes_dict['3_2_Employment'] = df_employment
dataframes_dict['3_3_Employment_Military'] = df_military_employment
dataframes_dict['4_2_LaborForce'] = df_laborforce
dataframes_dict['4_4_Unemployed'] = df_unemployed
dataframes_dict['4_5_Unemployment_Rate'] = df_unemploy_rate
dataframes_dict['4_6_Employment_Pop_Ratio'] = df_employ_pop_ratio

dataframes_piv_dict = {}
dataframes_piv_dict['1_1_Population_pivot'] = df_pop_piv
dataframes_piv_dict['1_5_Annual_Pop_Change_pivot'] = df_annual_pop_change_piv
dataframes_piv_dict['1_7_Birth_Rate_pivot'] = df_br_piv
dataframes_piv_dict['1_9_Death_Rate_pivot'] = df_dr_piv
dataframes_piv_dict['2_1_Income_pivot'] = df_income_piv
dataframes_piv_dict['2_4_Total_Income_pivot'] = df_total_income_piv
dataframes_piv_dict['3_2_Employment_pivot'] = df_employment_piv
dataframes_piv_dict['3_3_Employment_Military_pivot'] = df_military_employment_piv
dataframes_piv_dict['4_2_LaborForce_pivot'] = df_laborforce_piv
dataframes_piv_dict['4_4_Unemployed_pivot'] = df_unemployed_piv
dataframes_piv_dict['4_5_Unemployment_Rate_pivot'] = df_unemploy_rate_piv
dataframes_piv_dict['4_6_Employment_Pop_Ratio_pivot'] = df_employ_pop_ratio_piv

# need to save each dataframe individually
for i, key in enumerate(dataframes_dict):
    dataframes_dict[key].to_csv(SAVE_LOCATION + '/clean/' + key + '.csv',index=False)
    print('Saved ' + key + ' - shape:' + str(dataframes_dict[key].shape))
    

# need to save each pivoted dataframe individually
for i, key in enumerate(dataframes_piv_dict):
    dataframes_piv_dict[key].to_csv(SAVE_LOCATION + '/cleanpivots/' + key + '.csv',index=False)
    print('Saved ' +  key + ' - shape:' + str(dataframes_piv_dict[key].shape))
    
"""
Create csv with all the dataframes in the 1970 - 2015 date range
"""
# get the data from the clean pivots
filenames_lst = ['1_1_Population_pivot.csv',
                 '1_5_Annual_Pop_Change_pivot.csv',
                 '1_7_Birth_Rate_pivot.csv',
                 '1_9_Death_Rate_pivot.csv',
                 '2_1_Income_pivot.csv',
                 '2_4_Total_Income_pivot.csv',
                 '3_2_Employment_pivot.csv',
                 '3_3_Employment_Military_pivot.csv',
                 '4_2_LaborForce_pivot.csv',
                 '4_4_Unemployed_pivot.csv',
                 '4_5_Unemployment_Rate_pivot.csv',
                 '4_6_Employment_Pop_Ratio_pivot.csv']

# open the saved files (THIS ALSO CONFIRMS THE FILES WERE SAVED CORRECTLY) and set the indexes so .join can be used
df_pop = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[0]))
df_pop.set_index(['year', 'jurisdiction'], inplace=True)

df_pop_change = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[1]))
df_pop_change.set_index(['year', 'jurisdiction'], inplace=True)

df_birth_rate = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[2]))
df_birth_rate.set_index(['year', 'jurisdiction'], inplace=True)

df_death_rate = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[3]))
df_death_rate.set_index(['year', 'jurisdiction'], inplace=True)

df_income = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[4]))
df_income.set_index(['year', 'jurisdiction'], inplace=True)

df_total_income = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[5]))
df_total_income.set_index(['year', 'jurisdiction'], inplace=True)

df_employment = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[6]))
df_employment.set_index(['year', 'jurisdiction'], inplace=True)

df_employment_mil = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[7]))
df_employment_mil.set_index(['year', 'jurisdiction'], inplace=True)


df_all = df_pop.join(df_pop_change).join(df_birth_rate).join(df_death_rate).join(df_income).join(df_total_income).join(df_employment).join(df_employment_mil)
df_all = df_all.reset_index()
df_all.to_csv(SAVE_LOCATION + '/cleanpivots/1970_2015_data.csv',index=False)

"""
Create csv with all the dataframes in the 1990 - 2017 data range
"""
# open the saved files (THIS ALSO CONFIRMS THE FILES WERE SAVED CORRECTLY) and set the indexes so .join can be used
df_laborforce = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[8]))
df_laborforce.set_index(['year', 'jurisdiction'], inplace=True)

df_unemploy = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[9]))
df_unemploy.set_index(['year', 'jurisdiction'], inplace=True)

df_unemploy_rate = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[10]))
df_unemploy_rate.set_index(['year', 'jurisdiction'], inplace=True)

df_employ_pop_ratio = pd.read_csv(os.path.join(SAVE_LOCATION + '/cleanpivots', filenames_lst[11]))
df_employ_pop_ratio.set_index(['year', 'jurisdiction'], inplace=True)


df_all_1990_set = df_laborforce.join(df_unemploy).join(df_unemploy_rate).join(df_employ_pop_ratio)
df_all_1990_set = df_all_1990_set.reset_index()
df_all_1990_set.to_csv(SAVE_LOCATION + '/cleanpivots/1990_2017_data.csv',index=False)

"""
Outer join the 1970 - 2015 to the 1990 - 2017 datasets
"""
# set the indices
df_all.set_index(['year', 'jurisdiction'], inplace=True)
df_all_1990_set.set_index(['year', 'jurisdiction'], inplace=True)
# outer join
df_all_all = df_all.join(df_all_1990_set, how='outer')
# reset indices
df_all_all.reset_index(inplace=True)
# save
df_all_all.to_csv(SAVE_LOCATION + '/cleanpivots/all_data.csv',index=False)