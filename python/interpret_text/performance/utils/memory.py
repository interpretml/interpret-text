import os
import psutil

def get_peak_memory():
    if os.name == 'nt':
        # For windows, get peak working set memory
        process = psutil.Process(os.getpid())
        peak_memory = process.memory_info().peak_wset
    else:
        # For linux, get max resident set size
        import resource
        peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_memory