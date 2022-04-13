from ast import Not
import numpy as np
import pandas as pd

"""
Create One and Two Dimensional numpy arrays for simple GAN generation

1. age
1. age and unit type
    * only for:
        * CTICU, 
        * Cardiac ICU 
        * CSICU)
1. age and ethnicity
    * only for:
        * Caucasian
        * African American
        * Native American
"""

def create_eICU_Age_npy(eICU_df_filename = '../data/eICU/patients_pasthist_forgan.csv', save_filename = '../data/eICU/eICU_age.npy'):
    """
    Retrieve eICU dataframe from file, select the 'age' column and save to a .npy file. 
    Age will be converted to int8 np dtype.

    Parameters:
    :param eICU_df_filename: filename to retrieve the eICU data
    :param save_filename: filename to save the .npy file. If save_filename = 'None', then the file will not be saved.

    Returns:
    numpy array of age and unittype    
    """

    # retrieve the eICU data
    df_eICUforGAN = pd.read_csv(eICU_df_filename)

    # set the datatype to int, to get rid of default float
    dt = np.dtype([('age',np.int8)])
    np_age = np.array(df_eICUforGAN['age'], dtype=dt)

    if save_filename != 'None':
        # save npy file
        np.save(save_filename, np_age)

    return np_age


def create_eICU_Age_Unittype_npy(
    unittypes: np.array = ['ALL'], 
    eICU_df_filename: str = '../data/eICU/patients_pasthist_forgan.csv', 
    save_filename: str = '../data/eICU/eICU_age_unittype.npy'
    ): #-> np.np.array
    """
    Retrieve eICU dataframe from file, select the 'age' and 'unittype' columns and save to a .npy file.
    Unittypes included will only be the subset of unittypes in the unittypes parameter. 
    Age will be converted to int8 np dtype.

    Parameters:
    :param unittypes: array of unittype strings to include in the age_unittype npy file. Example: ['CTICU','Cardiac ICU','CSICU']. If 'ALL' is in the list, then all unittypes will be included
    :param eICU_df_filename: filename to retrieve the eICU data
    :param save_filename: filename to save the .npy file. If save_filename = 'None', then the file will not be saved.

    Returns:
    numpy array of age and unittype 
    """
    # retrieve the eICU data
    df_eICUforGAN = pd.read_csv(eICU_df_filename)

    # create the dataframes with only the unittypes and ethnicities we are interested in
    df_units_of_interest = df_eICUforGAN
    if not ('ALL' in unittypes):
        df_units_of_interest = df_eICUforGAN[df_eICUforGAN['unittype'].isin(unittypes)]

    dt = np.dtype([('age',np.int8)])
    np_age = np.array(df_units_of_interest['age'], dtype=dt)

    dt = np.dtype([('unittype', np.unicode_, 20)])
    np_unittype = np.array(df_units_of_interest['unittype'], dtype=dt)

    np_age_unittype = np.array(np.array((np_age,np_unittype)).T)

    if save_filename != 'None':
        # save npy file
        np.save(save_filename, np_age_unittype)
    
    return np_age_unittype

def create_eICU_Age_Ethnicity_npy(
    ethnicities: np.array = ['ALL'], 
    eICU_df_filename: str = '../data/eICU/patients_pasthist_forgan.csv', 
    save_filename: str = '../data/eICU/eICU_age_unittype.npy'
    ): #-> np.np.array
    """
    Retrieve eICU dataframe from file, select the 'age' and 'unittype' columns and save to a .npy file.
    Unittypes included will only be the subset of unittypes in the unittypes parameter. 
    Age will be converted to int8 np dtype.

    Parameters:
    :param unittypes: array of unittype strings to include in the age_unittype npy file. Example: ['CTICU','Cardiac ICU','CSICU']. If 'ALL' is in the list, then all unittypes will be included
    :param eICU_df_filename: filename to retrieve the eICU data
    :param save_filename: filename to save the .npy file. If save_filename = 'None', then the file will not be saved.

    Returns:
    numpy array of age and unittype 
    """
    # retrieve the eICU data
    df_eICUforGAN = pd.read_csv(eICU_df_filename)

    # create the dataframes with only the unittypes and ethnicities we are interested in
    df_ethnicity_of_interest = df_eICUforGAN
    if not ('ALL' in ethnicities):
        df_ethnicity_of_interest = df_eICUforGAN[df_eICUforGAN['ethnicity'].isin(ethnicities)]

    # age array with only ethnicities of interest
    dt = np.dtype([('age',np.int8)])
    np_age = np.array(df_ethnicity_of_interest['age'], dtype=dt)

    # ethnicity array with only ethnicities of interest
    dt = np.dtype([('ethnicity', np.unicode_, 20)])
    np_ethnicity = np.array(df_ethnicity_of_interest['ethnicity'], dtype=dt)

    # merge the two arrays
    np_age_ethnicity = np.array(np.array((np_age,np_ethnicity)).T)

    # save this file if file name not 'None'
    if save_filename != 'None':
        # save npy file
        np.save(save_filename, np_age_ethnicity)
    
    return np_age_ethnicity

def create_eICU_Age_PastHistory_UnitStayLength_npy(
    pasthistories: np.array = ['ALL'], 
    eICU_df_filename: str = '../data/eICU/patients_pasthist_forgan.csv', 
    save_filename: str = '../data/eICU/eICU_age_unitstaylength_pasthistory.npy'
    ): #-> np.np.array
    """
    Retrieve eICU dataframe from file, select the 'age' and 'unittype' columns and save to a .npy file.
    Unittypes included will only be the subset of unittypes in the unittypes parameter. 
    Age will be converted to int8 np dtype.

    Parameters:
    :param pasthistories: array of pasthistories strings to include in the npy file. Example: ['medication dependent','hypertension requiring treatment','COPD - moderate','stroke - date unknown','asthma']. 
    If 'ALL' is in the list, then all pasthistories will be included
    :param eICU_df_filename: filename to retrieve the eICU data
    :param save_filename: filename to save the .npy file. If save_filename = 'None', then the file will not be saved.

    Returns:
    numpy array of age and unittype 
    """
    # retrieve the eICU data
    df_eICUforGAN = pd.read_csv(eICU_df_filename)

    # create the dataframes with only the pasthistories we are interested in
    df_ethnicity_of_interest = df_eICUforGAN
    if not ('ALL' in pasthistories):
        df_ph_of_interest = df_eICUforGAN[df_eICUforGAN['pasthistoryvalue'].isin(pasthistories)]

    # age array with only past histories of interest
    dt = np.dtype([('age',np.int8)])
    np_age = np.array(df_ph_of_interest['age'], dtype=dt)

    # unitdischargeoffset array with only past histories of interest
    dt = np.dtype([('unitdischargeoffset',np.int8)])
    np_unitdischargeoffset = np.array(df_ph_of_interest['unitdischargeoffset'], dtype=dt)

    # ethnicity array with only past histories of interest
    dt = np.dtype([('pasthistoryvalue', np.unicode_, 20)])
    np_ph = np.array(df_ph_of_interest['pasthistoryvalue'], dtype=dt)

    # merge the two arrays
    np_age_unitdischargeoffset_ph = np.array(np.array((np_age,np_unitdischargeoffset,np_ph)).T)

    # save this file if file name not 'None'
    if save_filename != 'None':
        # save npy file
        np.save(save_filename, np_age_unitdischargeoffset_ph)
    
    return np_age_unitdischargeoffset_ph