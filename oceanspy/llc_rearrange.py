import copy as _copy
import reprlib

import dask
import numpy as _np
import xarray as _xr

# metric variables defined at vector points, defined as global within this file
metrics = ["dxC", "dyC", "dxG", "dyG", "hFacW", "hFacS", "rAs", "rAw", "maskS", "maskW"]


_datype = _xr.core.dataarray.DataArray
_dstype = _xr.core.dataset.Dataset


class LLCtransformation:
    """A class containing the transformation of LLCgrids"""

    def __init__(
        self,
        ds,
        varlist,
        transformation,
        centered="Atlantic",
        faces="all",
        chunks=None,
        drop=False,
    ):
        self._ds = ds  # xarray.DataSet
        self._varlist = varlist  # variables names to be transformed
        self._transformation = transformation  # str - type of transf
        self._centered = centered  # str - where to be centered
        self._chunks = (
            chunks  # dict - determining the relevant chunking of the dataset.
        )
        self._faces = faces  # faces involved in transformation

    @classmethod
    def arctic_crown(
        self,
        ds,
        varlist,
        centered,
        faces="all",
        chunks=None,
        drop=False,
    ):
        """Transforms the dataset in which `face` appears as a dimension into
        one without faces, with grids and variables sharing a common grid
        orientation.
        """

        if centered not in ["Atlantic", "Pacific"]:
            raise ValueError(
                "Centering option not recognized. Options are" "Atlantic or Pacific"
            )

        if isinstance(faces, str):
            faces = _np.arange(13)

        ds = _copy.deepcopy(mates(ds.reset_coords()))

        DIMS_c = [
            dim for dim in ds["XC"].dims if dim not in ["face"]
        ]  # horizontal dimensions on tracer points.
        DIMS_g = [
            dim for dim in ds["XG"].dims if dim not in ["face"]
        ]  # horizontal dimensions on corner points
        dims_c = Dims(DIMS_c[::-1])  # j, i format
        dims_g = Dims(DIMS_g[::-1])

        Nx = len(ds[dims_c.X])

        if isinstance(varlist, str):
            if varlist == "all":
                varlist = ds.data_vars
            else:
                varlist = [varlist]
        elif len(varlist) > 0:
            varlist = list(varlist)
        elif len(varlist) == 0:
            raise ValueError("Empty list of variables")

        dsa2 = []
        dsa5 = []
        dsa7 = []
        dsa10 = []
        ARCT = [dsa2, dsa5, dsa7, dsa10]

        for var_name in varlist:
            if "face" in ds[var_name].dims:
                arc_faces, *nnn, DS = arct_connect(ds, var_name, faces=faces)
                ARCT[0].append(DS[0])
                ARCT[1].append(DS[1])
                ARCT[2].append(DS[2])
                ARCT[3].append(DS[3])
            else:
                # print('here, '+ var_name)
                ARCT[0].append(ds[var_name])
                ARCT[1].append(ds[var_name])
                ARCT[2].append(ds[var_name])
                ARCT[3].append(ds[var_name])

        for i in range(len(ARCT)):
            if type(ARCT[i][0]) == _datype:
                ARCT[i] = _xr.merge(ARCT[i])

        DSa2, DSa5, DSa7, DSa10 = ARCT
        if type(DSa2) != _dstype:
            DSa2 = 0
        if type(DSa5) != _dstype:
            DSa5 = 0
        if type(DSa7) != _dstype:
            DSa7 = 0
        if type(DSa10) != _dstype:
            DSa10 = 0

        DSa7 = shift_dataset(DSa7, dims_c.X, dims_g.X)

        DSa10 = shift_dataset(DSa10, dims_c.Y, dims_g.Y)
        DSa10 = rotate_dataset(DSa10, dims_c, dims_g, rev_x=False, rev_y=True)
        DSa10 = rotate_vars(DSa10)

        DSa2 = rotate_dataset(
            DSa2, dims_c, dims_g, rev_x=True, rev_y=False, transpose=True
        )
        DSa2 = rotate_vars(DSa2)

        # =====
        # Determine the facets involved in the cutout
        _facet1 = [k for k in range(7, 10)]
        _facet2 = [k for k in range(10, 13)]
        _facet3 = [k for k in range(3)]
        _facet4 = [k for k in range(3, 6)]

        faces1 = []
        faces2 = []
        faces3 = []
        faces4 = []

        for k in _np.arange(13):
            if k in faces:
                if k in _facet1:
                    faces1.append(ds.isel(face=k))
                elif k in _facet2:
                    faces2.append(ds.isel(face=k))
                elif k in _facet3:
                    faces3.append(ds.isel(face=k))
                elif k in _facet4:
                    faces4.append(ds.isel(face=k))
            else:
                if k in _facet1:
                    faces1.append(0)
                elif k in _facet2:
                    faces2.append(0)
                elif k in _facet3:
                    faces3.append(0)
                elif k in _facet4:
                    faces4.append(0)

        # =====
        # Below are list for each facets containin either zero of a surviving face.

        faces1 = [DSa7] + faces1
        faces2 = [DSa10] + faces2
        faces3.append(DSa2)
        faces4.append(DSa5)

        # =====
        # Facet 1

        Facet1 = shift_list_ds(faces1, dims_c.X, dims_g.X, Nx)

        DSFacet1 = combine_list_ds(Facet1)
        DSFacet1 = rotate_vars(DSFacet1)
        DSFacet1 = rotate_dataset(
            DSFacet1,
            dims_c,
            dims_g,
            rev_x=False,
            rev_y=True,
            transpose=True,
            nface=int(3.5 * Nx),
        )

        if type(DSFacet1) == _dstype:
            for _var in DSFacet1.variables:
                if len(DSFacet1[_var].dims) > 2:
                    DIMS = [dim for dim in DSFacet1[_var].dims]
                    _dims = Dims(DIMS[::-1])
                    dtr = list(_dims)
                    dtr[-2], dtr[-1] = dtr[-1], dtr[-2]
                    DSFacet1[_var] = DSFacet1[_var].transpose(*dtr)
        DSFacet1 = flip_v(DSFacet1)

        # =====
        # Facet 2

        Facet2 = shift_list_ds(faces2, dims_c.X, dims_g.X, Nx)
        DSFacet2 = combine_list_ds(Facet2)
        DSFacet2 = rotate_dataset(
            DSFacet2,
            dims_c,
            dims_g,
            rev_x=False,
            rev_y=True,
            transpose=True,
            nface=int(3.5 * Nx),
        )
        DSFacet2 = rotate_vars(DSFacet2)

        if type(DSFacet2) == _dstype:
            for _var in DSFacet2.variables:
                if len(DSFacet2[_var].dims) > 2:
                    DIMS = [dim for dim in DSFacet2[_var].dims]
                    _dims = Dims(DIMS[::-1])
                    dtr = list(_dims)
                    dtr[-2], dtr[-1] = dtr[-1], dtr[-2]
                    DSFacet2[_var] = DSFacet2[_var].transpose(*dtr)
        DSFacet2 = flip_v(DSFacet2)

        # =====
        # combining Facet 1 & 2
        # =====

        FACETS = [DSFacet1, DSFacet2]
        fFACETS = shift_list_ds(FACETS, dims_c.X, dims_g.X, Nx, facet=12)
        DSFacet12 = combine_list_ds(fFACETS)

        # =====
        # Facet 3

        fFacet3 = shift_list_ds(faces3, dims_c.Y, dims_g.Y, Nx, facet=3)
        DSFacet3 = combine_list_ds(fFacet3)

        # =====
        # Facet 4
        fFacet4 = shift_list_ds(faces4, dims_c.Y, dims_g.Y, Nx, facet=4)
        DSFacet4 = combine_list_ds(fFacet4)

        # =====
        # combining Facet 3 & 4
        # =====

        FACETS = [DSFacet3, DSFacet4]
        fFACETS = shift_list_ds(FACETS, dims_c.X, dims_g.X, Nx, facet=34)
        DSFacet34 = combine_list_ds(fFACETS)

        # =====
        # combining all facets
        # =====

        if centered == "Pacific":
            FACETS = [DSFacet34, DSFacet12]  # centered on Pacific ocean
        elif centered == "Atlantic":
            FACETS = [DSFacet12, DSFacet34]  # centered at Atlantic ocean

        fFACETS = shift_list_ds(FACETS, dims_c.X, dims_g.X, 2 * Nx, facet=1234)
        DS = combine_list_ds(fFACETS)

        if drop:
            DS = DS.isel(X=slice(0, -1), Y=slice(0, -1))

        # rechunk data. In the ECCO data this is done automatically
        if chunks:
            DS = DS.chunk(chunks)

        return DS


def arct_connect(ds, varName, faces="all"):

    arc_cap = 6
    Nx_ac_nrot = []
    Ny_ac_nrot = []
    Nx_ac_rot = []
    Ny_ac_rot = []
    ARCT = [0, 0, 0, 0]  # initialize the list.
    arc_faces = [0, 0, 0, 0]
    metrics = [
        "dxC",
        "dyC",
        "dxG",
        "dyG",
        "hFacW",
        "hFacS",
    ]  # metric variables defined at vector points

    if isinstance(faces, str):
        if faces == "all":
            faces = [k for k in range(13)]

    if arc_cap in faces:
        for k in faces:
            if k == 2:
                fac = 1
                arc_faces[0] = k
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != "face"]
                dims = Dims(DIMS[::-1])
                dtr = list(dims)[::-1]
                dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                mask2 = _xr.ones_like(ds[_varName].isel(face=arc_cap))
                # TODO: Eval where, define argument outside
                mask2 = mask2.where(
                    _np.logical_and(
                        ds[dims.X] < ds[dims.Y],
                        ds[dims.X] < len(ds[dims.Y]) - ds[dims.Y],
                    )
                )
                x0, xf = 0, int(len(ds[dims.Y]) / 2)  # TODO: CHECK here!
                y0, yf = 0, int(len(ds[dims.X]))
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_nrot.append(0)
                Ny_ac_nrot.append(len(ds[dims.Y][y0:yf]))
                da_arg = {"face": arc_cap, dims.X: xslice, dims.Y: yslice}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                if len(dims.X) + len(dims.Y) == 4:
                    if len(dims.Y) == 3 and _varName not in metrics:
                        fac = -1
                arct = fac * ds[_varName].isel(**da_arg)
                Mask = mask2.isel(**mask_arg)
                arct = arct * Mask
                ARCT[0] = arct

            elif k == 5:
                fac = 1
                arc_faces[1] = k
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != "face"]
                dims = Dims(DIMS[::-1])
                mask5 = _xr.ones_like(ds[_varName].isel(face=arc_cap))
                mask5 = mask5.where(
                    _np.logical_and(
                        ds[dims.X] > ds[dims.Y],
                        ds[dims.X] < len(ds[dims.Y]) - ds[dims.Y],
                    )
                )
                x0, xf = 0, int(len(ds[dims.X]))
                y0, yf = 0, int(len(ds[dims.Y]) / 2)
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_nrot.append(0)
                Ny_ac_nrot.append(len(ds[dims.X][y0:yf]))
                if len(dims.X) + len(dims.Y) == 4:
                    if len(dims.Y) == 1 and _varName not in metrics:
                        fac = -1
                da_arg = {"face": arc_cap, dims.X: xslice, dims.Y: yslice}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                arct = ds[_varName].isel(**da_arg)
                Mask = mask5.isel(**mask_arg)
                arct = arct * Mask
                ARCT[1] = arct

            elif k == 7:
                fac = 1
                arc_faces[2] = k
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != "face"]
                dims = Dims(DIMS[::-1])
                dtr = list(dims)[::-1]
                dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                mask7 = _xr.ones_like(ds[_varName].isel(face=arc_cap))
                mask7 = mask7.where(
                    _np.logical_and(
                        ds[dims.X] > ds[dims.Y],
                        ds[dims.X] > len(ds[dims.Y]) - ds[dims.Y],
                    )
                )
                x0, xf = int(len(ds[dims.Y]) / 2), int(len(ds[dims.Y]))
                y0, yf = 0, int(len(ds[dims.X]))
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_rot.append(len(ds[dims.Y][x0:xf]))
                Ny_ac_rot.append(0)
                da_arg = {"face": arc_cap, dims.X: xslice, dims.Y: yslice}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                arct = fac * ds[_varName].isel(**da_arg)
                Mask = mask7.isel(**mask_arg)
                arct = arct * Mask
                ARCT[2] = arct

            elif k == 10:
                fac = 1
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != "face"]
                dims = Dims(DIMS[::-1])
                dtr = list(dims)[::-1]
                dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                arc_faces[3] = k
                mask10 = _xr.ones_like(ds[_varName].isel(face=arc_cap))
                mask10 = mask10.where(
                    _np.logical_and(
                        ds[dims.X] < ds[dims.Y],
                        ds[dims.X] > len(ds[dims.Y]) - ds[dims.Y],
                    )
                )
                x0, xf = 0, int(len(ds[dims.X]))
                y0, yf = int(len(ds[dims.Y]) / 2), int(len(ds[dims.Y]))
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_rot.append(0)
                Ny_ac_rot.append(len(ds[dims.Y][y0:yf]))
                da_arg = {"face": arc_cap, dims.X: xslice, dims.Y: yslice}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                arct = fac * ds[_varName].isel(**da_arg)
                Mask = mask10.isel(**mask_arg)
                arct = (arct * Mask).transpose(*dtr)
                ARCT[3] = arct

    return arc_faces, Nx_ac_nrot, Ny_ac_nrot, Nx_ac_rot, Ny_ac_rot, ARCT


def mates(ds):
    vars_mates = [
        "ADVx_SLT",
        "ADVy_SLT",
        "ADVx_TH",
        "ADVy_TH",
        "DFxE_TH",
        "DFyE_TH",
        "DFxE_SLT",
        "DFyE_SLT",
        "maskW",
        "maskS",
        "TAUX",
        "TAUY",
        "U",
        "V",
        "UVELMASS",
        "VVELMASS",
        "dxC",
        "dyC",
        "dxG",
        "dyG",
        "hFacW",
        "hFacS",
        "rAw",
        "rAs",
    ]
    for k in range(int(len(vars_mates) / 2)):
        nk = 2 * k
        if vars_mates[nk] in ds.variables:
            ds[vars_mates[nk]].attrs["mate"] = vars_mates[nk + 1]
            ds[vars_mates[nk + 1]].attrs["mate"] = vars_mates[nk]
    return ds


def rotate_vars(_ds):
    """using the attribures `mates`, when this function is called it swaps the variables names.
    This issue is only applicable to llc grid in which the
    grid topology makes it so that u on a rotated face transforms to `+- v` on a lat lon grid.
    """
    if type(_ds) == _dstype:  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)
        _vars = [var for var in _ds.variables]
        rot_names = {}
        for v in _vars:
            if "mate" in _ds[v].attrs:
                rot_names = {**rot_names, **{v: _ds[v].mate}}

        _ds = _ds.rename(rot_names)
    return _ds


def shift_dataset(_ds, dims_c, dims_g):
    """shifts a dataset along a dimension, setting its first element to zero. Need to provide the dimensions in the
    form of [center, corner] points. This rotation is only used in the horizontal, and so dims_c is either one of `i`
    or `j`, and dims_g is either one of `i_g` or `j_g`. The pair most correspond to the same dimension.

    _ds: dataset

    dims_c: string, either 'i' or 'j'

    dims_g: string, either 'i_g' or 'j_g'. Should correspond to same dimension as dims_c.

    """
    if type(_ds) == _dstype:  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)
        for _dim in [dims_c, dims_g]:
            _ds["n" + _dim] = _ds[_dim] - int(_ds[_dim][0].data)
            _ds = (
                _ds.swap_dims({_dim: "n" + _dim})
                .drop_vars([_dim])
                .rename({"n" + _dim: _dim})
            )

        _ds = mates(_ds)
    return _ds


def reverse_dataset(_ds, dims_c, dims_g, transpose=False):
    """reverses the dataset along a dimension. Need to provide the dimensions in the form
    of [center, corner] points. This rotation is only used in the horizontal, and so dims_c is either one of `i`  or
    `j`, and dims_g is either one of `i_g` or `j_g`. The pair most correspond to the same dimension."""

    if type(_ds) == _dstype:  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)

        for _dim in [dims_c, dims_g]:  # This part should be different for j_g points?
            _ds["n" + _dim] = -_ds[_dim] + int(_ds[_dim][-1].data)
            _ds = (
                _ds.swap_dims({_dim: "n" + _dim})
                .drop_vars([_dim])
                .rename({"n" + _dim: _dim})
            )

        _ds = mates(_ds)

        if transpose:
            _ds = _ds.transpose()
    return _ds


def rotate_dataset(
    _ds, dims_c, dims_g, rev_x=False, rev_y=False, transpose=False, nface=1
):
    """Rotates a dataset along its horizontal dimensions (e.g. center and corner). It can also shift the dataset
    along a dimension, reserve its orientaton and transpose the whole dataset.

    _ds : dataset

    dims_c = [dims_c.X, dims_c.Y]
    dims_c = [dims_g.X, dims_g.Y]

    nface=1: flag. A single dataset is being manipulated.
    nface=int: correct number to use. This is the case a merger/concatenated dataset is being manipulated. Nij is no
        longer the size of the face.

    """
    if type(_ds) == _dstype:  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)
        Nij = max(
            len(_ds[dims_c.X]), len(_ds[dims_c.Y])
        )  # max number of points of a face. If ECCO data, Nij  = 90. If LLC4320, Nij=4320

        if rev_x is False:
            fac_x = 1
            x0 = 0
        elif rev_x is True:
            fac_x = -1
            if nface == 1:
                x0 = int(Nij) - 1
            else:
                x0 = nface
        if rev_y is False:
            fac_y = 1
            y0 = 0
        elif rev_y is True:
            fac_y = -1
            if nface == 1:
                y0 = int(Nij) - 1
            else:
                y0 = nface - 1

        for _dimx, _dimy in [[dims_c.X, dims_c.Y], [dims_g.X, dims_g.Y]]:
            _ds["n" + _dimx] = fac_x * _ds[_dimy] + x0
            _ds["n" + _dimy] = fac_y * _ds[_dimx] + y0

            _ds = _ds.swap_dims({_dimx: "n" + _dimy, _dimy: "n" + _dimx})
            _ds = _ds.drop_vars({_dimx, _dimy}).rename(
                {"n" + _dimx: _dimx, "n" + _dimy: _dimy}
            )

        _ds = mates(_ds)

        if transpose:
            _ds = _ds.transpose()
    return _ds


def shift_list_ds(_DS, dims_c, dims_g, Ni, facet=1):
    """given a list of n-datasets, each element of the list gets shifted along the dimensions provided (dims_c and dims_g)
    so that there is no overlap between them.
    """
    _DS = _copy.deepcopy(_DS)
    fac = 1
    if facet in [1, 2]:
        facs = [0.5, 1, 1, 1]
    elif facet in [3, 4, 12, 34]:
        facs = [1, 1, 1, 1]
    elif facet == 1234:
        facs = [1, 1, 1, 1]
        fac = 0  # only here.
    if len(_DS) > 1:
        dim0 = 0
        for ii in range(1, len(_DS)):
            if type(_DS[ii - 1]) == int:
                dim0 = int(Ni * sum(facs[:ii]))
            else:
                for _dim in [dims_c, dims_g]:
                    dim0 = int(_DS[ii - 1][_dim][-1].data + 1)
            if type(_DS[ii]) == _dstype:
                for _dim in [dims_c, dims_g]:
                    _DS[ii]["n" + _dim] = (
                        _DS[ii][_dim] - (fac * int(_DS[ii][_dim][0].data)) + dim0
                    )
                    _DS[ii] = (
                        _DS[ii]
                        .swap_dims({_dim: "n" + _dim})
                        .drop_vars([_dim])
                        .rename({"n" + _dim: _dim})
                    )
        DS = []
        for lll in range(len(_DS)):
            if type(_DS[lll]) == _dstype:
                DS.append(_DS[lll])
    else:
        DS = _DS
    return DS


def combine_list_ds(_DSlist):
    """combines a list of n-datasets"""
    if len(_DSlist) == 0:
        _DSFacet = 0  # No dataset to combine. Return empty
    elif len(_DSlist) == 1:  # a single face
        _DSFacet = _DSlist[0]
    elif len(_DSlist) == 2:
        if type(_DSlist[0]) == int:  # one is empty, pass directly
            _DSFacet = _DSlist[1]
        elif type(_DSlist[1]) == int:  # the other is empty pass directly
            _DSFacet = _DSlist[0]
        else:  # if there are two datasets then combine
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                _DSFacet = _DSlist[0].combine_first(_DSlist[1])
    elif len(_DSlist) > 2:
        _DSFacet = _copy.deepcopy(_DSlist[0])
        for ii in range(1, len(_DSlist)):
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                _DSFacet = _DSFacet.combine_first(_copy.deepcopy(_DSlist[ii]))

        _DSFacet = mates(_DSFacet)

    return _DSFacet


def flip_v(_ds, co_list=metrics):
    """reverses the sign of the velocity field v."""
    if type(_ds) == _dstype:
        for _varName in _ds.variables:
            DIMS = [dim for dim in _ds[_varName].dims if dim != "face"]
            _dims = Dims(DIMS[::-1])
            if "mate" in _ds[_varName].attrs:
                if _varName not in co_list and len(_dims.Y) == 3:
                    _ds[_varName] = -_ds[_varName]
    return _ds


class Dims:
    axes = "XYZT"  # shortcut axis names

    def __init__(self, vars):
        self._vars = tuple(vars)

    def __iter__(self):
        return iter(self._vars)

    def __repr__(self):
        vars = reprlib.repr(self._vars)
        return "{}".format(vars)

    def __str__(self):
        return str(tuple(self))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __len__(self):
        return len(self._vars)

    def __getattr__(self, name):
        cls = type(self)
        if len(name) == 1:
            pos = cls.axes.find(name)
            if 0 <= pos < len(self._vars):
                return self._vars[pos]
        msg = "{.__name__!r} object has not attribute {!r}"
        raise AttributeError(msg.format(cls, name))

    def __setattr__(self, name, value):
        cls = type(self)
        if len(name) == 1:
            if name in cls.axes:
                error = "read-only attribute {attr_name!r}"
            elif name.islower():
                error = "can`t set attributes `a` to `z` in {cls_name!r}"
            else:
                error = ""
            if error:
                msg = error.format(cls_name=cls.__name__, attr_name=name)
                raise AttributeError(msg)
        super().__setattr__(name, value)
