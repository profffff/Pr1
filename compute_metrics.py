from PAPR_analyzer import PAPR_analyzer

def compute_metrics(Xm, scenario):
    metrics = {}
    metrics['CCDF'] = PAPR_analyzer(Xm, scenario)
    return metrics