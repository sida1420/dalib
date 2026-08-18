"""Read-only process memory observations for diagnostic reporting."""

import ctypes
import os


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory_bytes() -> tuple[int | None, int | None]:
    """Return current and peak RSS without mutating simulation state."""

    if os.name != "nt":
        return None, None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong,
    ]
    get_memory_info.restype = ctypes.c_int
    handle = get_current_process()
    success = get_memory_info(
        handle, ctypes.byref(counters), counters.cb,
    )
    if not success:
        return None, None
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
