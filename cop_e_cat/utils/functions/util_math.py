import statistics
from typing import Dict, List

def combine_stats(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    overall_stats: Dict[str, float] = {}
    for d in dicts:
        for k in d:
            if k in overall_stats:
                overall_stats[k].append(d[k])
            else:
                overall_stats[k] = d[k]
    for k in overall_stats.keys():
        overall_stats[k] = statistics.mean(overall_stats[k])

    return overall_stats
