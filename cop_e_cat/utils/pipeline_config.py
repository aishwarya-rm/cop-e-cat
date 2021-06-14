from pandas.core.frame import DataFrame
import psycopg2
import pandas as pd
import pickle
from typing import Any, Optional

sqluser = ''
dbname = ''
sqlpwd = ''
output_dir = "../../processed_data/pkl_files/"
stats_dir = "../../processed_data/stats_files/"

# File names
admissions_pkl = output_dir + 'admissions.pkl'
hadm_ids = output_dir + 'hadm_ids.txt'
cohort_stats = stats_dir + 'cohort_stats.json'

labs_pkl = output_dir + 'labs.pkl'
labs_stats = stats_dir + 'labs_stats.json'

vitals_pkl = output_dir + 'vitals.pkl'
vitals_stats = stats_dir + 'vitals_stats.json'

procedures_pkl = output_dir + 'procedures.pkl'
output_pkl = output_dir + 'outputevents.pkl'
meds_icu_pkl = output_dir + 'meds_icu.pkl'
meds_hosp_pkl = output_dir + 'meds_hosp.pkl'

popmeans_stats = output_dir + 'popmeans.json' 


# Query function
def q(query: str, schema_name: str, save_name: Any=None) -> Optional[DataFrame]:
    """Connect to database, execute a query, and save results to file.

    Args:
        query (str): Query to execute, as interpolated string. Note that this runs the risk of SQL injection.
        schema_name (str): Schema to connect to.
        save_name (Any, optional): Value is ignored,but if set, query reslt will be pickled to file.

    Returns:
        Optional[DataFrame]: If save_name is unset, the result of pandas exectuting the query will
            be returned as a DataFrame. Otherwise, value is pickled to file.
    """
    con = psycopg2.connect(dbname=dbname, user=sqluser, password=sqlpwd, host='db2-sn17')
    cur = con.cursor()
    cur.execute('SET search_path to ' + schema_name)
    if save_name is not None:
        queried = pd.read_sql_query(query,con)
        # 'queried' should be a data frame--do we really want to use that as a file name?
        pickle.dump(queried, open(queried, 'wb'))
    else:
        return pd.read_sql_query(query,con)
