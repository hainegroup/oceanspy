import numpy as _np

# ================
# USEFUL FUNCTIONS
# ================
def _check_instance(objs, classinfos):
    for key, value in objs.items():
        if isinstance(classinfos, str): classinfo=classinfos
        else                          : classinfo=classinfos[key]

        if not eval('isinstance(value, {})'.format(classinfo)):
            if classinfo[0]=='_': classinfo = classinfo[1:]
            raise TypeError("`{}` must be {}".format(key, classinfo))

def _check_list_of_string(obj, objName):
    obj = _np.asarray(obj, dtype='str')
    if obj.ndim == 0: 
        obj = obj.reshape(1)
    elif obj.ndim >1: 
        raise TypeError('Invalid `{}`'.format(objName))
    return obj

def _check_range(od, obj, objName):
    if   'Y'    in objName:
        pref    = 'Y'
        valchek = od._ds['YG']
    elif 'X'    in objName:
        pref    = 'X'
        valchek = od._ds['XG']
    elif 'Z'    in objName:
        pref    = 'Z'
        valchek = od._ds['Zp1']
    elif 'time' in objName:
        pref    = 'time'
        valchek = od._ds['time']
    
    obj  = _np.asarray(obj, dtype=valchek.dtype)
    if obj.ndim == 0: obj = obj.reshape(1)
    elif obj.ndim >1: 
        raise TypeError('Invalid `{}`'.format(objName))
    maxcheck = valchek.max().values
    mincheck = valchek.min().values
    if any(obj<mincheck) or any(obj>maxcheck):
        _warnings.warn("\n{}Range of the oceandataset is: {}"
                       "\nRequested {} has values outside this range.".format(pref, [mincheck, maxcheck], objName), stacklevel=2)
    return obj

def _check_native_grid(od, func_name):
    wrong_dims = ['mooring', 'station', 'particle']
    for wrong_dim in wrong_dims:
        if wrong_dim in od._ds.dims:
            raise ValueError('`{}` cannot subsample {} oceandatasets'.format(func_name, wrong_dims))
            
def _check_part_position(od, InputDict):
    for InputName, InputField in InputDict.items():
        if 'time' in InputName:
            InputField = _np.asarray(InputField, dtype = od._ds['time'].dtype)
            if InputField.ndim == 0: 
                InputField = InputField.reshape(1)
            ndim = 1
        else:
            InputField  = _np.asarray(InputField)
            if InputField.ndim <2 and InputField.size==1: 
                InputField = InputField.reshape((1, InputField.size))
            ndim = 2
        if InputField.ndim > ndim: 
            raise TypeError('Invalid `{}`'.format(InputName))
        else: 
            InputDict[InputName] = InputField
    return InputDict

def _handle_aliased(od, aliased, varNameList):
    if aliased:
        varNameListIN = _rename_aliased(od, varNameList)
    else:
        varNameListIN = varNameList
    varNameListOUT = varNameList
    return varNameListIN, varNameListOUT

def _check_ijk_components(od, iName=None, jName=None, kName=None):
    ds = od._ds
    for _, (Name, dim) in enumerate(zip([iName, jName, kName], ['Xp1', 'Yp1', 'Zl'])):
        if Name is not None and dim not in ds[Name].dims:
            raise ValueError('[{}] must have dimension [{}]'.format(Name, dim))