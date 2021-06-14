import datetime as dt
import json
import os
import numpy as np
import pandas as pd
import tqdm
from typing import Any, Dict, List, Tuple, Optional

from cop_e_cat import covariate_selection as cs
import cop_e_cat.utils.params as pm
from cop_e_cat.utils import combine_stats, pipeline_config, init_configuration, local_verbosity, print_per_verbose


class CopECatParams():
    def __init__(self, params_file):
        f = open(params_file)
        self.params = json.load(f)
        self.imputation:         bool                 = self.params['imputation']
        self.patientweight:      bool                 = self.params['patientweight']
        self.delta:              int                  = self.params['delta']
        self.max_los:            int                  = self.params['max_los']
        self.min_los:            int                  = self.params['min_los']
        self.adults:             bool                 = self.params['adults_only']
        self.icd_codes:          List[Any]            = self.params['icd_codes']
        self.vitals_dict:        Dict[str, List[int]] = self.params['vitals_dict']
        self.labs_dict:        Dict[str, List[int]] = self.params['labs_dict']
        self.vent_dict:          Dict[str, List[int]] = self.params['vent_dict']
        self.output_dict:        Dict[str, List[int]] = self.params['output_dict']
        self.meds_dict:          Dict[str, List[int]] = self.params['meds_dict']
        self.proc_dict:          Dict[str, List[str]] = self.params['procs_dict']    ## NOTE TYPO CHANGE
        self.comorbidities_dict: Dict[str, List[str]] = self.params['comorbidities_dict']
        self.checkpoint_dir:     str                  = self.params['checkpoint_dir'] ## NOTE TYPO CHANGE
        self.output_dir:         Optional[str]        = self.params['output_dir']

        # Initialize the output directories
        self.init_dirs()

    def init_dirs(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)


def _timestamp_in_window(index: Any, reference_time: Any, series: pd.Series, hours_delta: int) -> bool:
    """Determine whether a given timestamp (as 'charttime' field of a dataframe Series entry) is within a given delta
    of a reference time.
    Notionally, this is intended to be used inside a loop that's iterating over a dataframe (or slice thereof). Each
    entry in the corresponding dataframe is expected to have a 'charttime' field with timestamp data.
    (It would probably be more elegant to vectorize this.)

    Args:
        index (Any): Index iterator over dataframe.
        reference_time (Any): the 'current' timestamp for a chart.
        series (pd.Series): Relevant slice of a dataframe.
        hours_delta (int): Size of the window (in hours).

    Returns:
        bool: True if the entry of series at index 'index' is between reference_time and reference_time + hours_delta,
        otherwise False.
    """        
    delta = dt.timedelta(hours=hours_delta)
    try:
        entry_chart_time = pd.to_datetime(series.loc[index, 'charttime'])
        return (entry_chart_time >= reference_time and entry_chart_time < reference_time + delta)
    except:
        try:
            entry_chart_time = pd.to_datetime(series.loc[index, 'charttime'])
            return (entry_chart_time >= reference_time).any() and (entry_chart_time < reference_time + delta).any()
        except:
            return False
    return False

def _update_frame_with_value(chart: pd.DataFrame, key: str, timestamp: Any, index: Any, series: pd.Series) -> pd.DataFrame:
    """Update dataframe `chart` to average in the new value identified by the 'value' key of data series `series` for the index `index`.

    Args:
        chart (pd.DataFrame): The state space dataframe being built
        key (str): Key identifying the covariate being iterated over
        timestamp (Any): Timestamp of the current window/bucket.
        index (Any): index of iteration over the rows recording covariate observations.
        series (pd.Series): Relevant slice of the source data frame: the covariate observations, subsetted to a relevant
            group of covariates.

    Returns:
        pd.DataFrame: The chartframe being built, as modified in-place.
    """    
    key_lc = key.lower()
    import ipdb; ipdb.set_trace()
    chart.loc[timestamp, key_lc] = np.nanmean([chart.loc[timestamp, key_lc] , series.loc[index, 'value']])
    return chart

def _update_frame_with_valuenum(chart: pd.DataFrame, key: str, timestamp: Any, index: Any, series: pd.Series) -> pd.DataFrame:
    """Update dataframe `chart` to average in the new value identified by the 'valuenum' key of data series `series` for the index `index`.

    Args:
        chart (pd.DataFrame): The state space dataframe being built
        key (str): Key identifying the covariate being iterated over
        timestamp (Any): Timestamp of the current window/bucket.
        index (Any): index of iteration over the rows recording covariate observations.
        series (pd.Series): Relevant slice of the source data frame: the covariate observations, subsetted to a relevant
            group of covariates.

    Returns:
        pd.DataFrame: The chartframe being built, as modified in-place.
    """    
    key_lc = key.lower()
    ## As with the function above, I'm concerned about iterative averaging here
    chart.loc[timestamp, key_lc] = np.nanmean([chart.loc[timestamp, key_lc], series.loc[index, 'valuenum']])
    return chart

def _buildTimeFrame(df_adm: pd.DataFrame, delta: int=6) -> pd.DataFrame:
        """Generate dataframe with one entry per desired final state space. This is the scaffold
        to which we will subsequently add requested features.

        Args:
            df_adm (pd.DataFrame): Dataframe containing admissions data.
            delta (int, optional): Hours per state space. Defaults to 6.

        Returns:
            pd.DataFrame: Dataframe with timestamps to align features to.
        """
        # Get admit and discharge time in numeric form, round down/up respectively to the nearest hour
        start = pd.to_datetime(df_adm.intime.unique()).tolist()[0]
        start -= dt.timedelta(minutes=start.minute, seconds=start.second, microseconds=start.microsecond)

        # Set end time to the end of the hour in which the last patient was dismissed
        end = pd.to_datetime(df_adm.outtime.unique()).tolist()[0]
        end -= dt.timedelta(minutes=end.minute, seconds=end.second, microseconds=end.microsecond)
        end += dt.timedelta(hours=1)

        times = []
        curr = start
        while curr < end:
            times.append(curr)
            curr += dt.timedelta(hours=delta)
        timeFrame = pd.DataFrame(data={'timestamp': times}, index=times)
        return timeFrame

class CopECat():
    """Class which generates state spaces for MIMIC-IV data, using specified data values, interpolation, and cohort settings.
    """

    def __init__(self, params: CopECatParams):
        """Constructor for CopECat instance.

        Args:
            params (CopECatParams): Configuration parameters (table names and loaded parameters file).
        """		
        self.params = params
        self.ONE_HOT_LABS = ['Ca', 'Glucose', 'CPK', 'PTH', 'LDH', 'AST', 'ALT']
        self.UNIT_SCALES = { 'grams': 1000, 'mcg': 0.001, 'pg': 1e-9, 'ml': 1 }

    def _load_data(self) -> Tuple[List[int], List[Any], Dict[str, float]]:
        """Return the cohort, dataframes containing various features, and the mean of each feature over the cohort population.

        Returns:
            Tuple[List[int], List[Any], Dict[str, float]]: 3-tuple. First element is a deduped list of unique hadm_ids for every
                member of the cohort. Second element is a list of dataframes and/or Tuple[DataFrame, PopulationFeaturesDict]
                with the table-level features. Final element is a dictionary mapping feature name (from params.py) to the
                mean of that feature over the retrieved cohort population.
        """		
        admissions, cohort, _ = cs.gather_cohort(adults=self.params.adults, patient_weight=self.params.patientweight,
                                                  icd_diagnoses=self.params.icd_codes, min_los=self.params.min_los,
                                                  max_los=self.params.max_los, regenerate=True)
        vitals, stats = cs.gather_vitals(self.params.vitals_dict, regenerate=True)
        labs, stats = cs.gather_labs(self.params.labs_dict, regenerate=True)
        outputs = cs.gather_outputs(self.params.output_dict, regenerate=True)
        procs, vent = cs.gather_procedures(self.params.vent_dict, self.params.proc_dict)
        meds = cs.gather_meds(self.params.meds_dict, regenerate=True)
        comorbs = cs.gather_comorbidities(self.params.comorbidities_dict)
        
        popmeans = self._generate_popmeans()
        cohort = list(set(set(cohort) & set(vitals['hadm_id']) & set(labs['hadm_id']) & set(meds['hadm_id'])))
        print("Length of cohort: ", len(cohort))
        return cohort, [admissions, vitals, labs, outputs, procs, vent, meds, comorbs], popmeans

    def _generate_popmeans(self) -> Dict[str, float]:
        """Write to file the population means of the requested features.

        Returns:
            Dict[str, float]: Dictionary mapping feature name to population mean.
        """
        # Hardcoded vars--these should be the paths to the feature files extracted by pipeline_config
        f = open(pipeline_config.vitals_stats)
        vitals_stats = json.load(f)
        f = open(pipeline_config.labs_stats)
        labs_icu_stats = json.load(f)
        f = open(pipeline_config.cohort_stats)
        cohort_stats = json.load(f)

        popmeans = combine_stats([vitals_stats, labs_icu_stats, cohort_stats])
        with open(pipeline_config.popmeans_stats, 'w') as f:
            json.dump(popmeans, f)
        return popmeans
    
    # Modify this line to adjust verbosity level in code, overriding what's set by command line argument
    #@local_verbosity(new_level=3)
    def _chartFrames(self, hadm_id: str, tables: List[pd.DataFrame], popmeans: Dict[str, float]) -> pd.DataFrame:
        """Build time-indexed dataframe of each patient admission, with resampled values of all variables

        Args:
            hadm_id (int): Hadm_id for one patient
            tables (List[Any]): List of the dataframes (often with PopulationFeaturesDict) for the extracted
                table-level data.
            popmeans (Dict[str, float]): Mapping of feature name to its mean among the cohort.

        Returns:
            pd.DataFrame: State space representation for one patient.
        """
        use_admissions    = True
        use_comorbidities = True
        use_vitals        = True
        use_labs          = True
        use_meds          = True
        use_procedures    = True
        use_ventilation   = True
        admissions, vitals, labs, outputs, procs, vent, meds, comorbidities = tables

        delta: int = self.params.delta
            
        def _add_admissions(chart: pd.DataFrame, admission_frame: pd.Series) -> pd.DataFrame:
            print_per_verbose(1, 'Admission Data')
            for var in ['hadm_id', 'anchor_age', 'patientweight', 'los', 'gender']:
                chart[var.lower()] = admission_frame[var].head(1).item()
            chart['gender'] = (chart['gender'] == 'F').astype(int)
            return chart
        
        def _add_comorbidities(chart: pd.DataFrame, comorbidities: pd.DataFrame, hadm_id: int, comorbidities_dict: Dict[str, List[str]]) -> pd.DataFrame:
            print_per_verbose(1, 'Morbidities')
            df_comorbs = comorbidities[comorbidities.hadm_id == hadm_id]
            for subpop in comorbidities_dict:
                subpop_df = df_comorbs[df_comorbs.long_title.isin(comorbidities_dict[subpop])]
                if subpop_df.empty:
                    chart[subpop] = 0
                else:
                    chart[subpop] = 1
            chart['expired'] = 0
            return chart
            
        def _add_vitals(chart: pd.DataFrame, vitals: pd.DataFrame, hadm_id: int, delta: int, vitals_dict: Dict[str, List[int]], popmeans: Dict[str, float]) -> pd.DataFrame:
            print_per_verbose(1, 'Vitals')
            df_vits = vitals[vitals.hadm_id == hadm_id].drop_duplicates()
            for k in sorted(list(vitals_dict.keys())):
                chart[k.lower()] = np.nan
                for t in chart.timestamp:
                    subset = df_vits[df_vits.itemid.isin(vitals_dict[k])]
                    for i in subset.index:
                        if _timestamp_in_window(i, t, subset, delta):
                            chart = _update_frame_with_valuenum(chart, k, t, i, subset)
                chart[k.lower()] = chart[k.lower()].fillna(method='ffill').fillna(value=popmeans[k])
            return chart
        
        def _add_labs(chart: pd.DataFrame, labs: pd.DataFrame, hadm_id:int, delta:int, labs_dict: Dict[str, List[int]], popmeans: Dict[str, float]) -> pd.DataFrame:
            print_per_verbose(1, 'Labs')
            df_labs = labs[labs.hadm_id == hadm_id].drop_duplicates()
            for k in sorted(list(labs_dict.keys())):
                chart[k.lower()] = np.nan
                for t in chart.timestamp:
                    subset = df_labs[df_labs['itemid'].isin(self.params.labs_dict[k])]
                    for i in subset.index:
                        if _timestamp_in_window(i, t, subset, delta):
                            if k not in self.ONE_HOT_LABS:
                                chart = _update_frame_with_valuenum(chart, k, t, i, subset)
                            else:
                                chart.loc[t, k.lower()] = 1
                if k not in self.ONE_HOT_LABS:
                    chart[k.lower()] = chart[k.lower()].fillna(method='ffill').fillna(value=popmeans[k])
                else:
                    chart[k.lower()] = chart[k.lower()].fillna(method='ffill', limit=24//delta).fillna(value=0)
            return chart
        
        def _add_vent(chart: pd.DataFrame, vent: pd.DataFrame, hadm_id:int, delta:int) -> pd.DataFrame:
            print_per_verbose(1, 'Ventilation')
            ## Again, guessing about the table values. I don't understand the selector code.
            df_vent = vent[vent.hadm_id == hadm_id].drop_duplicates()
            if df_vent.empty:
                chart['vent'] = 0
            else:
                chart['vent'] = np.nan
                for t in chart.timestamp:
                    for i in df_adm.index:
                        if _timestamp_in_window(i, t, df_vent, delta):
                            chart = _update_frame_with_value(chart, 'vent', t, i, df_vent)
                        else:
                            chart.loc[t, 'vent'] = 0
            return chart
        def _add_meds(chart: pd.DataFrame, meds: pd.DataFrame, hadm_id:int, delta:int, meds_dict: Dict[str, List[int]]) -> pd.DataFrame:
            print_per_verbose(1, 'Medication')
            df_meds = meds[meds.hadm_id == hadm_id].drop_duplicates()
            for k in sorted(list(meds_dict.keys())):
                chart[k.lower()] = 0
                if k in ['K-IV', 'K-nonIV', 'Mg-IV', 'Mg-nonIV', 'P-IV', 'P-nonIV']:
                    chart['hours-' + k.lower()] = 0
                subset = df_meds[df_meds.itemid.isin(self.params.meds_dict[k])]
                for t in chart.timestamp:
                    for i, row in subset.iterrows():
                        if row.amountuom == 'dose':
                            continue
                        scaler = self.UNIT_SCALES.get(row.amountuom, 1)
                        if row.endtime is np.nan:
                            chart.loc[t, 'hours-' + k.lower()] = float('nan')
                            continue
                        td = pd.to_datetime(row.endtime) - pd.to_datetime(row.starttime)
                        hours = td.days * 24 + td.seconds // 3600
                        ## QUERY: So we have two tests:
                        # Case 1: med entry starts in the current time bucket.
                        # --> Add the amount (scaled to mg), & add the time-delta (in hours) to hours-[key].
                        # Case 2: If the med entry starts before this bucket and ends after the current bucket:
                        # --> Add the amount (scaled to mg) & add the time-delta (in hours) to hours-[key].
                        # Case 3: If the med entry starts after the start of this time bucket or does not end after this time bucket:
                        # --> Add 1 to the amount.
                        # Cases 1 & 3 are not mutually exclusive--in fact case 1 is a subset of case 3 (assuming endtime is always
                        # greater than starttime)--and I'm not sure why Case 3 adds a static 1 instead of the amount.
                        # What am I missing?
                        if ((pd.to_datetime(row.starttime) >= t) and (pd.to_datetime(row.starttime) < t + dt.timedelta(hours=delta))):
                            chart.loc[t, k.lower()] += scaler * float(row.amount)
                            chart.loc[t, 'hours-' + k.lower()] = hours
                        if ((pd.to_datetime(row.starttime) <= t) and (pd.to_datetime(row.endtime) > t)):
                            chart.loc[t, k.lower()] += scaler * float(row.amount)
                            chart.loc[t, 'hours-' + k.lower()] = hours
                        else:
                            chart.loc[t, k.lower()] += 1
            return chart
        
        def _add_procs(chart: pd.DataFrame, procs: pd.DataFrame, hadm_id:int, delta:int, proc_dict: Dict[str, List[int]]) -> pd.DataFrame:
            df_procs = procs[procs.hadm_id == hadm_id]
            keys = sorted(list(proc_dict.keys()))
            import ipdb; ipdb.set_trace()
            for k in keys:
                chart[k] = 0
                subset = df_procs[df_procs.inputkey == k]
                for t in chart.timestamp:
                    for i, row in subset.iterrows():
                        if ((row.input_start >= t) and (row.input_start < t + dt.timedelta(hours=delta))):
                            chart.loc[t, k] = 1
                        if ((row.input_start <= t) and (row.input_end > t)):
                            chart.loc[t, k] = 1
            return chart
                        

        df_adm = admissions[admissions.hadm_id == hadm_id].drop_duplicates()
        chart = _buildTimeFrame(df_adm)
        if use_admissions:
            chart = _add_admissions(chart, df_adm)
        
        if use_comorbidities:
            chart = _add_comorbidities(chart, comorbidities, hadm_id, self.params.comorbidities_dict)

        if use_vitals:
            chart = _add_vitals(chart, vitals, hadm_id, delta, self.params.vitals_dict, popmeans)

        if use_labs:
            chart = _add_labs(chart, labs, hadm_id, delta, self.params.labs_dict, popmeans)

        if use_ventilation:
            chart = _add_vent(chart, vent, hadm_id, delta)

        if use_meds:
            chart = _add_meds(chart, meds, hadm_id, delta, self.params.meds_dict)

        if use_procedures:
            chart = _add_procs(chart, procs, hadm_id, delta, self.params.proc_dict)

        chart = chart[~np.isnat(chart.timestamp)]
        # Delete any rows/columns with missing values?
        chart = chart.dropna()
        print_per_verbose(1, 'Done!')

        return chart

    def _gridBatch(self, batch: List[int], tables: List[Any], popmeans: Dict[str, float]) -> None:
        """Persist state space to csv.

        Args:
            batch (List[str]): Unique list of hadm_ids for every member of cohort.
            tables (List[Any]): Tables for the various feature sources, see _load_data.
            popmeans (Dict[str, float]): Dictionary mapping each feature name to its mean value in the cohort.
        """
        batchCharts = pd.DataFrame()

        for i, hadm_id in tqdm.tqdm(enumerate(batch)):
            #try:
            chart = self._chartFrames(hadm_id, tables, popmeans)
            batchCharts = batchCharts.append(chart, ignore_index=True)
            if i % 5 == 0:
                batchCharts.to_csv(self.params.checkpoint_dir + str(i) + 'checkpoint.csv', index=False)
#             except:
#                 print('Error in', hadm_id)
        print("Batch done!")
        batchCharts.to_csv(self.params.output_dir + 'allFrames.csv', index=False)

    def generate_state_spaces(self):
        cohort, tables, popmeans = self._load_data()
        self._gridBatch(cohort[:10], tables, popmeans)



