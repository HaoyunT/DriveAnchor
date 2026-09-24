from pathlib import Path
import ctypes
import numpy as np
lib=ctypes.CDLL(str(Path(__file__).with_name('dp_geometry_kernel.so')))
fn=lib.curvature_cost
fn.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_double]
fn.restype=ctypes.c_int

def evaluate(path,max_curvature):
    path=np.ascontiguousarray(path,dtype=np.float64)
    if path.ndim!=2 or path.shape[1]!=2:raise ValueError('Nx2 path required')
    value=ctypes.c_double()
    status=fn(path.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(path),ctypes.byref(value),max_curvature)
    if status<0:raise ValueError('Invalid native kernel input')
    return status==1,value.value
