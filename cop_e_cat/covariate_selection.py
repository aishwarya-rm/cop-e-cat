# Adapted from code written by Niranjani Prasad: https://github.com/bee-hive/covidEHR/tree/master/mimic/np6/utils
import sys

sys.path.append("../utils/")
import os
import json
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from pandas.core.frame import DataFrame
import pickle
from cop_e_cat.utils import pipeline_config, print_per_verbose

PopulationFeaturesDict = Dict[str, Tuple[float, float]]


def gather_cohort(
    adults: bool = True,
    patient_weight: bool = True,
    icd_diagnoses: List[
        str
    ] = [],  # assuming these are represented as strings not floats
    min_los: int = 1,
    max_los: int = 8,
    regenerate: bool = True,
) -> Tuple[DataFrame, List[int], PopulationFeaturesDict]:
    """Collect the target cohort (patient population).

    Args:
        adults (bool, optional): [description]. Defaults to True.
        patient_weight (bool, optional): [description]. Defaults to True.
        icd_diagnoses (List[str], optional): If set, results will be filtered to only those patient records
            with one of the included diagnosis codes. Unset by default.
        min_los (int, optional): Minimum length of stay for a record to be included. Defaults to 1.
        max_los (int, optional): Maximum length of stay for a record to be included. Defaults to 8.
        regenerate (bool, optional): Regenerate the dataframe, disregarding a saved version

   Returns:
        Tuple[DataFrame, List[str], PopulationFeaturesDict]: Three items: A dataframe containing all admissions, a list of hadm_ids,
            and a data structure recording the mean and standard deviation for the weight and age of the selected patients.
    """
    # Filter for adults with icustay length between 1 and 8 days
    if regenerate:
        query = (
            "select subject_id, stay_id, hadm_id, intime, outtime, los from icustays where los > "
            + str(min_los)
            + " and los <= "
            + str(max_los)
            + ";"
        )
        adm = pipeline_config.q(query, "mimic_icu")

        if adults:
            print_per_verbose(1, "Filtering for age >= 18 years")
            # Filter for patients greater than 18 years old
            query = "select subject_id, gender, anchor_age from patients where anchor_age >= 18;"
            ages = pipeline_config.q(query, "mimic_core")
        else:
            query = "select subject_id, gender, anchor_age from patients;"
            ages = pipeline_config.q(query, "mimic_core")
        # Admissions + Age information
        admissions = pd.merge(adm, ages, on=["subject_id"])

        if patient_weight:
            print_per_verbose(1, "Adding information about patientweight")
            # Add patient weight information
            query = "select stay_id, hadm_id, patientweight from procedureevents;"
            weights = pipeline_config.q(query, "mimic_icu")
            admissions = pd.merge(admissions, weights, on=["hadm_id"]).drop_duplicates()

        if len(icd_diagnoses) > 0:
            print_per_verbose(1, "Filtering for ICD diagnoses")
            query = (
                "select hadm_id, icd_code, icd_version from diagnoses_icd where icd_code in "
                + str(tuple(icd_diagnoses))
            )
            diagnoses = pipeline_config.q(query, "mimic_hosp")
            admissions = pd.merge(
                admissions, diagnoses, on=["hadm_id"]
            ).drop_duplicates()

        stats = {}
        features = ["patientweight", "anchor_age"]
        for f in features:
            average = admissions[f].mean()
            sd = admissions[f].std()
            stats[f] = (average, sd)

        hadm_ids = admissions.hadm_id.unique()

        print_per_verbose(1, "Saving data")
        admissions.to_pickle(pipeline_config.admissions_pkl)
        with open(pipeline_config.hadm_ids, "w") as f:
            for id in hadm_ids:
                f.write("%s," % id)

        with open(
            os.path.join(pipeline_config.stats_dir, "cohort_stats.json"), "w"
        ) as f:
            json.dump(stats, f)
    else:
        with open(pipeline_config.admissions_pkl, "rb") as f:
            admissions = pickle.load(f)
        f = open(pipeline_config.hadm_ids, 'r')
        hadm_ids = f.read().split(',')

        f = open(os.path.join(pipeline_config.stats_dir, "cohort_stats.json"))
        stats = json.load(f)
    return admissions, hadm_ids, stats


def gather_labs(
    labs_dict: Dict[str, List[int]], regenerate: bool = True
) -> Tuple[DataFrame, PopulationFeaturesDict]:
    """Gather information about ICU level labs.

    Args:
            labs_dict (Dict[str, List[int]]): Dict keyed by lab descriptions, whose values are a string
                    representation of corresponding item ids.
            regenerate (bool, optional): Regenerate the dataframe, disregarding a saved version

    Returns:
            Tuple[DataFrame, PopulationFeaturesDict]: Dataframe of filtered labs, and statistics of
                    population mean and sd for each lab description (combining relevant itemids).
    """
    if regenerate:
        labs_itemids = []
        for key in labs_dict:
            itemids = labs_dict[key]
            labs_itemids.extend(itemids)

        print_per_verbose(1, "Gathering labs")
        query = (
            "select stay_id, hadm_id, charttime, itemid, valuenum, valueuom from chartevents where itemid in "
            + str(tuple(labs_itemids))
            + " and valuenum >= 0 and valuenum < 99999"
        )
        filtered_labs_df = pipeline_config.q(query, "mimic_icu")

        stats = {}
        for key in labs_dict:
            itemids = labs_dict[key]
            mean = filtered_labs_df[filtered_labs_df["itemid"].isin(itemids)]['valuenum'].mean()
            sd = filtered_labs_df[filtered_labs_df["itemid"].isin(itemids)]['valuenum'].std()
            stats[key] = (mean, sd)

        print_per_verbose(1, "Saving lab stats")
        with open(os.path.join(pipeline_config.stats_dir, "labs_stats.json"), "w") as f:
            json.dump(stats, f)

        print_per_verbose(1, "Saving lab dataframes")
        filtered_labs_df.to_pickle(os.path.join(pipeline_config.output_dir, "labs"))
    else:
        with open(os.path.join(pipeline_config.output_dir, "labs"), "rb") as f:
            filtered_labs_df = pickle.load(f)
        f = open(pipeline_config.hadm_ids, 'r')
        hadm_ids = f.read().split(',')

        f = open(os.path.join(pipeline_config.stats_dir, "labs_stats.json"))
        stats = json.load(f)
    return filtered_labs_df, stats


def gather_meds(meds_dict: Dict[str, List[int]], regenerate: bool = True) -> DataFrame:
    """Gather all ICU level medications.

    Args:
            meds_dict (Dict[str, str]): Dictionary of medication descriptions mapped to list of names? (Underlying data is unclear)
            regenerate (bool, optional): Regenerate the dataframe, disregarding a saved version

    Returns:
            DataFrame: Deduplicated data frame of medication records.
    """
    if regenerate:
        meds_ids = []
        for key in meds_dict:
            names = meds_dict[key]
            meds_ids += names

        print_per_verbose(1, "Gathering meds")
        query = (
            "SELECT subject_id, hadm_id, stay_id, starttime, endtime, storetime, itemid, amount, amountuom, rate, rateuom FROM inputevents WHERE itemid in "
            + str(tuple(meds_ids))
            + " and amount > 0 and amount < 9999"
        )
        filtered_meds_df = pipeline_config.q(query, "mimic_icu")

        print_per_verbose(1, "Saving meds dataframes")
        filtered_meds_df.to_pickle(os.path.join(pipeline_config.output_dir, "meds_icu"))
    else:
        with open(os.path.join(pipeline_config.output_dir, "meds_icu"), "rb") as f:
            filtered_meds_df = pickle.load(f) 
    return filtered_meds_df.drop_duplicates()


def gather_vitals(
    vitals_dict: Dict[str, List[int]], regenerate: bool = True
) -> Tuple[DataFrame, PopulationFeaturesDict]:
    """Gather all vitals records.

    Args:
            vitals_dict (Dict[str, List[int]]): Dictionary of vitals descriptions mapped to list of item ids (as text)
            regenerate (bool, optional): Regenerate the dataframe, disregarding a saved version

    Returns:
            Tuple[DataFrame, PopulationFeaturesDict]: Data frame of vitals, and statistics on their means/sds.
    """
    if regenerate:
        vitals_itemids = []
        for key in vitals_dict:
            itemid = vitals_dict[key]
            vitals_itemids.extend(itemid)

        query = (
            "SELECT subject_id, hadm_id, stay_id, charttime, storetime, itemid, value, valuenum, valueuom, warning FROM chartevents"
            + " WHERE itemid in "
            + str(tuple(vitals_itemids))
            + " and valuenum >= 0 and valuenum < 9999"
        )
        print_per_verbose(1, "Gathering vitals")
        filtered_vitals_df = pipeline_config.q(query, "mimic_icu")
        filtered_vitals_df = filtered_vitals_df.drop_duplicates()

        stats: PopulationFeaturesDict = {}
        for key in vitals_dict:
            itemids = vitals_dict[key]
            mean = filtered_vitals_df[filtered_vitals_df["itemid"].isin(itemids)][
                "valuenum"
            ].mean()
            sd = filtered_vitals_df[filtered_vitals_df["itemid"].isin(itemids)][
                "valuenum"
            ].std()
            stats[key] = (mean, sd)

        print_per_verbose(1, "Saving vitals dataframes")
        filtered_vitals_df.to_pickle(os.path.join(pipeline_config.output_dir, "vitals"))

        print_per_verbose(1, "Saving vitals stats")
        with open(
            os.path.join(pipeline_config.stats_dir, "vitals_stats.json"), "w"
        ) as f:
            json.dump(stats, f)
    else:
        with open(os.path.join(pipeline_config.output_dir, "vitals.pkl"), "rb") as f:
            filtered_vitals_df = pickle.load(f)
        f = open(os.path.join(pipeline_config.stats_dir, "vitals_stats.json"))
        stats = json.load(f)
    return filtered_vitals_df, stats


def gather_outputs(output_dict: Dict[str, List[int]], regenerate: bool = True) -> DataFrame:
    """Gather urine output.

    Args:
            output_dict (Dict[str, List[int]]): Dictionary of output-item descriptions, mapped to list of item ids (as text)
            regenerate (bool, optional): Regenerate the dataframe, disregarding a saved version

    Returns:
            DataFrame: Dataframe of output events.
    """
    if regenerate:
        output_itemids = []
        for key in output_dict:
            output_itemids.extend(output_dict[key])

        print_per_verbose(1, "Gathering outputs")
        query = (
            "select hadm_id, charttime, itemid, value, valueuom from outputevents where itemid in "
            + str(tuple(output_itemids))
            + " and value >= 0 and value < 99999"
        )

        filtered_output_df = pipeline_config.q(query, "mimic_icu")

        print_per_verbose(1, "Saving dataframes")
        filtered_output_df.to_pickle(os.path.join(pipeline_config.output_dir, "output"))
    else:
        with open(os.path.join(pipeline_config.output_dir, "output"), "rb") as f:
            filtered_output_df = pickle.load(f)
    return filtered_output_df


def gather_procedures(
    vent_dict: Dict[Any, Any], procs_dict: Dict[Any, Any]
) -> Tuple[DataFrame, DataFrame]:
    """Gather information about ventilation and procedures.

    Args:
            vent_dict (Dict[Any, Any]): Dictionary of MIMIC-IV itemids related to ventilation.
            procs_dict (Dict[Any, Any]): (Unused.) Dictionary of MIMIV-IV itemids related to procedures.

    Returns:
            Tuple[DataFrame, DataFrame]: Tuple of deduplicated dataframes (Procedures, Ventilations)
    """

    def vent_icu(vent_itemids):
        query = (
            "SELECT subject_id, hadm_id, stay_id, starttime, endtime, itemid, amount, amountuom, rate, rateuom, totalamount FROM inputevents WHERE itemid in "
            + str(tuple(vent_itemids))
            + " and amount > 0 and amount < 9999"
        )
        filtered_vent_df = pipeline_config.q(query, "mimic_icu")
        return filtered_vent_df.drop_duplicates()
    
    def procs_icu(procs_itemids):
        query = (
            "SELECT subject_id, stay_id, hadm_id, itemid, starttime, value, valueuom FROM procedureevents WHERE itemid="
            + str(procs_itemids)
            + " and value >=0"
        )
        filtered_procs_df = pipeline_config.q(query, "mimic_icu")
        return filtered_procs_df.drop_duplicates()
    
    vent_itemids = []
    for key in vent_dict:
        itemids = vent_dict[key]
        vent_itemids += itemids

    print_per_verbose(1, "Gathering ventilation information")
    if len(vent_itemids) > 0:
        vent = vent_icu(vent_itemids)
    else:
        vent = pd.DataFrame()
    
    procs_itemids = 225441
    print_per_verbose(1, "Gathering procedure information")
    procs = procs_icu(
        procs_itemids
    )  # parameter is ignored--not present in function signature

    print_per_verbose(1, "Saving dataframes")
    procs.to_pickle(os.path.join(pipeline_config.output_dir, "procs"))
    vent.to_pickle(os.path.join(pipeline_config.output_dir, "ventilation"))

    return procs, vent


"""
Gather information about comorbidities. 
"""


def gather_comorbidities(comorbidities_dict):
    comorb_strings = []
    for key in comorbidities_dict:
        names = comorbidities_dict[key]
        comorb_strings += names

    query = (
        "select icd_code, icd_version, long_title from d_icd_diagnoses where long_title in "
        + str(tuple(comorb_strings))
    )
    comorbs = pipeline_config.q(query, "mimic_hosp")

    query = "select hadm_id, icd_code, icd_version from diagnoses_icd"
    diags = pipeline_config.q(query, "mimic_hosp")

    all_diags = pd.merge(comorbs, diags, on=["icd_code", "icd_version"], how="inner")
    return all_diags
