"""
OceanSpy functionality that transforms a dataset with LLC geometry characterized by
13 faces (or tiles), into one with simple geometry.
"""

import copy as _copy
import reprlib

import dask
import numpy as _np
import xarray as _xr
from xgcm import Grid

from .utils import _rel_lon, _reset_range, get_maskH

# metric variables defined at vector points, defined as global within this file
metrics = ["dxC", "dyC", "dxG", "dyG", "HFacW", "HFacS", "rAs", "rAw", "maskS", "maskW"]


_datype = _xr.core.dataarray.DataArray
_dstype = _xr.core.dataset.Dataset


class LLCtransformation:
    """A class containing the transformation types of LLCgrids."""

    def __init__(
        self,
        ds,
        varList=None,
        add_Hbdr=0,
        XRange=None,
        YRange=None,
        faces=None,
        centered=False,
        chunks=None,
    ):
        self._ds = ds  # xarray.DataSet
        self._varList = varList  # variables names to be transformed
        self._add_Hbdr = add_Hbdr
        self._XRange = XRange  # lon range of data to retain
        self._YRange = YRange  # lat range of data to retain.
        self._chunks = chunks  # dict.
        self._faces = faces  # faces involved in transformation
        self._centered = centered
        self._chunks = chunks

    @classmethod
    def arctic_crown(
        self,
        ds,
        varList=None,
        add_Hbdr=0,
        XRange=None,
        YRange=None,
        faces=None,
        centered=None,
        chunks=None,
    ):
        """This transformation splits the arctic cap (face=6) into four triangular
        regions and combines all faces in a quasi lat-lon grid. The triangular
        arctic regions form a crown atop faces {7, 10, 2, 5}. The final size of
        the transformed dataset depends XRange, YRange or faces.

        Parameters
        ----------
        dataset: xarray.Dataset
            The multi-dimensional, in memory, array database. e.g., `oceandataset._ds`.
        varList: 1D array_like, str, or None
            List of variables (strings).
        YRange: 1D array_like, scalar, or None
            Y axis limits (e.g., latitudes).
            If len(YRange)>2, max and min values are used.
        XRange: 1D array_like, scalar, or None
            X axis limits (e.g., longitudes). Can handle (periodic) discontinuity at
            lon=180 deg E.
        add_Hbdr: bool, scal
            If scalar, add and subtract `add_Hbdr` to the the horizontal range.
            of the horizontal ranges.
            If True, automatically estimate add_Hbdr.
            If False, add_Hbdr is set to zero.
        faces: 1D array_like, scalar, or None
            List of faces to be transformed.
            If None, entire dataset is transformed.
            When both [XRange, YRange] and faces are defined, [XRange, YRange] is used.
        centered: str or bool.
            If 'Atlantic' (default), the transformation creates a dataset in which the
            Atlantic Ocean lies at the center of the domain.
            If 'Pacific', the transformed data has a layout in which the Pacific Ocean
            lies at the center of the domain.
            This option is only relevant when transforming the entire dataset.
        chunks: bool or dict.
            If False (default) - chunking is automatic.
            If dict, rechunks the dataset according to the spefications of the
            dictionary. See xarray.chunk().

        Returns
        -------

        ds: xarray.Dataset
            face is no longer a dimension of the dataset.


        Notes
        -----
        This functionality is very similar to, takes on similar arguments and is used
        internally by subsample.cutout when extracting cutout regions of datasets with
        face as a dimension.


        References
        ----------
        https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html

        https://docs.xarray.dev/en/stable/generated/xarray.Dataset.chunk.html


        See Also
        --------
        subsample.cutout
        """

        print("Warning: This is an experimental feature")
        if "face" not in ds.dims:
            raise ValueError("face does not appear as a dimension of the dataset")

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

        if Nx == 90:  # ECCO dataset
            add_Hbdr = add_Hbdr + 2
        else:
            add_Hbdr = add_Hbdr + 0.25

        if varList is None:
            varList = ds.data_vars

        varList = list(varList)

        # store original attributes
        attrs = {}
        for var in varList:
            attrs = {var: ds[var].attrs, **attrs}

        #
        if faces is None:
            faces = _np.arange(13)

        if XRange is not None and YRange is not None:
            XRange = _np.array(XRange)
            YRange = _np.array(YRange)
            if _np.max(abs(XRange)) > 180 or _np.max(abs(YRange)) > 90:
                raise ValueError("Range of lat and/or lon is not acceptable.")
            else:
                XRange, ref_lon = _reset_range(XRange)
                maskH, dmaskH, XRange, YRange = get_maskH(
                    ds, add_Hbdr, XRange, YRange, ref_lon=ref_lon
                )
                faces = list(dmaskH["face"].values)
                ds = mask_var(ds, XRange, YRange, ref_lon)  # masks latitude
                _var_ = "nYG"  # copy variable created in mask_var. Will discard
                varList = varList + [_var_]
                cuts = arc_limits_mask(ds, _var_, faces, dims_g, XRange, YRange)

                opt = True
        else:

            opt = False
            cuts = None

        print("faces in the cutout", faces)

        #
        dsa2 = []
        dsa5 = []
        dsa7 = []
        dsa10 = []
        ARCT = [dsa2, dsa5, dsa7, dsa10]

        for var_name in varList:
            if "face" in ds[var_name].dims:
                arc_faces, *nnn, DS = arct_connect(
                    ds, var_name, faces=faces, masking=False, opt=opt, ranges=cuts
                )
                ARCT[0].append(DS[0])
                ARCT[1].append(DS[1])
                ARCT[2].append(DS[2])
                ARCT[3].append(DS[3])
            else:
                ARCT[0].append(ds[var_name])
                ARCT[1].append(ds[var_name])
                ARCT[2].append(ds[var_name])
                ARCT[3].append(ds[var_name])

        for i in range(len(ARCT)):
            if all(type(item) != int for item in ARCT[i]):
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

        # Slicing the faces to remove nan-edges.
        # Only when XRange and YRange given.
        if XRange is not None and YRange is not None:
            for axis in range(2):
                edges1 = _edge_facet_data(faces1, _var_, dims_g, axis)
                faces1 = slice_datasets(faces1, dims_c, dims_g, edges1, axis)
                edges2 = _edge_facet_data(faces2, _var_, dims_g, axis)
                faces2 = slice_datasets(faces2, dims_c, dims_g, edges2, axis)
                edges3 = _edge_facet_data(faces3, _var_, dims_g, axis)
                faces3 = slice_datasets(faces3, dims_c, dims_g, edges3, axis)
                edges4 = _edge_facet_data(faces4, _var_, dims_g, axis)
                faces4 = slice_datasets(faces4, dims_c, dims_g, edges4, axis)

            # Here, address shifts in Arctic
            # arctic exchange with face 10
            if type(faces2[0]) == _dstype:
                faces2[0]["Yp1"] = faces2[0]["Yp1"] + 1

            # Arctic exchange with face 2
            if type(faces3[3]) == _dstype:
                faces3[3]["Xp1"] = faces3[3]["Xp1"] + 1

        # =====
        # Facet 1

        Facet1 = shift_list_ds(faces1, dims_c.X, dims_g.X, Nx)
        DSFacet1 = combine_list_ds(Facet1)
        DSFacet1 = flip_v(DSFacet1)
        DSFacet1 = reverse_dataset(DSFacet1, dims_c.X, dims_g.X)
        DSFacet1 = rotate_dataset(DSFacet1, dims_c, dims_g)
        DSFacet1 = rotate_vars(DSFacet1)

        # =====
        # Facet 2

        Facet2 = shift_list_ds(faces2, dims_c.X, dims_g.X, Nx)
        DSFacet2 = combine_list_ds(Facet2)
        DSFacet2 = flip_v(DSFacet2)
        DSFacet2 = reverse_dataset(DSFacet2, dims_c.X, dims_g.X)
        DSFacet2 = rotate_dataset(DSFacet2, dims_c, dims_g)
        DSFacet2 = rotate_vars(DSFacet2)

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
        DSFacet34 = shift_dataset(DSFacet34, dims_c.Y, dims_g.Y)

        # =====
        # determine `centered` , i.e. order in which facets are combined
        # only a factor is there is data in facets with different topology
        # =====

        if centered is None:  # estimates the centering based on cutout
            centered = "Atlantic"  # default, below scenarios to change this
        if type(DSFacet3) == int:
            centered = "Pacific"

        # =====
        # combining all facets
        # =====

        # First, check if there is data in both DSFacet12 and DSFacet34.
        # If not, then there is no need to transpose data in DSFacet12.

        if type(DSFacet12) == _dstype:
            if type(DSFacet34) == _dstype:
                # two lines below asserts correct
                # staggering of center and corner points
                # in latitude (otherwise, lat has a jump)
                if YRange is not None:
                    DSFacet12["Y"] = DSFacet12["Y"] - 1
                    DSFacet12 = DSFacet12.isel(Y=slice(0, -1))
                elif YRange is None:
                    DSFacet34["Yp1"] = DSFacet34["Yp1"] - 1
                    DSFacet34 = DSFacet34.isel(Yp1=slice(0, -1))
                for _var in DSFacet12.data_vars:
                    DIMS = [dim for dim in DSFacet12[_var].dims]
                    dims = Dims(DIMS[::-1])
                    if len(dims) > 1 and "nv" not in DIMS:
                        dtr = list(dims)[::-1]
                        dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                        if YRange is not None and XRange is not None:
                            DSFacet12[_var] = DSFacet12[_var].transpose(*dtr).persist()
                        else:
                            DSFacet12[_var] = DSFacet12[_var].transpose(*dtr)

        if centered == "Pacific":
            FACETS = [DSFacet34, DSFacet12]  # centered on Pacific ocean
        elif centered == "Atlantic":
            FACETS = [DSFacet12, DSFacet34]  # centered at Atlantic ocean

        fFACETS = shift_list_ds(FACETS, dims_c.X, dims_g.X, 2 * Nx, facet=1234)
        DS = combine_list_ds(fFACETS)

        if "face" in DS.coords:
            # only relevant when the transformation involves a single face
            DS = DS.drop_vars(["face"])

        # #  shift
        DS = shift_dataset(DS, dims_c.X, dims_g.X)
        DS = shift_dataset(DS, dims_c.Y, dims_g.Y)

        if type(DSFacet34) == int:
            DS = _reorder_ds(DS, dims_c, dims_g).persist()

        DS = _LLC_check_sizes(DS)

        # rechunk data. In the ECCO data this is done automatically
        if chunks:
            DS = DS.chunk(chunks)

        if XRange is not None and YRange is not None:
            # drop copy var = 'nYg' (line 101)
            DS = DS.drop_vars(_var_)

        DS = llc_local_to_lat_lon(DS)

        # restore original attrs if lost
        for var in varList:
            if var in DS.reset_coords().data_vars:
                DS[var].attrs = attrs[var]

        return DS


def arct_connect(ds, varName, faces=None, masking=False, opt=False, ranges=None):
    """
    Splits the arctic into four triangular regions.
    if `masking = True`: does not transpose data. Only use when masking for data not
        surviving the cutout. Default is `masking=False`, which implies data in arct10
        gets transposed.

    `opt=True` must be accompanied by a list `range` with len=4. Each element of
        `range` is either a pair of zeros (implies face does not survive the cutout),
        or a pair of integers of the form `[X0, Xf]` or `[Y0, Yf]`. `opt=True` only
        when optimizing the cutout so that the transformation of the arctic is done
        only with surviving data.
    """

    arc_cap = 6
    Nx_ac_nrot = []
    Ny_ac_nrot = []
    Nx_ac_rot = []
    Ny_ac_rot = []
    ARCT = [0, 0, 0, 0]  # initialize the list.
    arc_faces = [0, 0, 0, 0]

    if faces is None:
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
                elif _varName == "CS":
                    fac = -1
                arct = fac * ds[_varName].isel(**da_arg)
                Mask = mask2.isel(**mask_arg)
                if opt:
                    [Xi_2, Xf_2] = [ranges[0][0], ranges[0][1]]
                    cu_arg = {dims.X: slice(Xi_2, Xf_2)}
                    arct = (arct.sel(**cu_arg) * Mask.sel(**cu_arg)).persist()
                else:
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
                if opt:
                    [Yi_5, Yf_5] = [ranges[1][0], ranges[1][1]]
                    cu_arg = {dims.Y: slice(Yi_5, Yf_5)}
                    arct = (arct.sel(**cu_arg) * Mask.sel(**cu_arg)).persist()
                else:
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
                if opt:
                    [Xi_7, Xf_7] = [ranges[2][0], ranges[2][1]]
                    cu_arg = {dims.X: slice(Xi_7, Xf_7)}
                    arct = (arct.sel(**cu_arg) * Mask.sel(**cu_arg)).persist()
                else:
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
                if len(dims.X) + len(dims.Y) == 4:
                    if len(dims.Y) == 1 and _varName not in metrics:
                        fac = -1
                elif _varName == "SN":
                    fac = -1
                da_arg = {"face": arc_cap, dims.X: xslice, dims.Y: yslice}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                arct = fac * ds[_varName].isel(**da_arg)
                Mask = mask10.isel(**mask_arg)
                if masking:
                    if opt:
                        [Yi_10, Yf_10] = [ranges[-1][0], ranges[-1][1]]
                        cu_arg = {dims.Y: slice(Yi_10, Yf_10)}
                        arct = arct.sel(**cu_arg) * Mask.sel(**cu_arg)
                    else:
                        arct = arct * Mask
                else:
                    if opt:
                        [Yi_10, Yf_10] = [ranges[-1][0], ranges[-1][1]]
                        cu_arg = {dims.Y: slice(Yi_10, Yf_10)}
                        arct = (
                            (arct.sel(**cu_arg) * Mask.sel(**cu_arg))
                            .transpose(*dtr)
                            .persist()
                        )
                    else:
                        arct = (arct * Mask).transpose(*dtr)
                ARCT[3] = arct

    return arc_faces, Nx_ac_nrot, Ny_ac_nrot, Nx_ac_rot, Ny_ac_rot, ARCT


def mates(ds):
    """Defines, when needed, the variable pair and stores the name of the pair (mate)
    variable as an attribute. This is needed to accurately rotate a vector field.
    """
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
        "HFacW",
        "HFacS",
        "rAw",
        "rAs",
        "CS",
        "SN",
    ]
    for k in range(int(len(vars_mates) / 2)):
        nk = 2 * k
        if vars_mates[nk] in ds.variables:
            ds[vars_mates[nk]].attrs["mate"] = vars_mates[nk + 1]
            ds[vars_mates[nk + 1]].attrs["mate"] = vars_mates[nk]
    return ds


def rotate_vars(_ds):
    """Using the attribures `mates`, when this function is called it swaps the
    variables names. This issue is only applicable to llc grid in which the grid
    topology makes it so that u on a rotated face transforms to `+- v` on a lat lon
    grid.
    """
    if type(_ds) == _dstype:  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)
        _vars = [var for var in _ds.variables]
        rot_names = {}
        for v in _vars:
            if "mate" in _ds[v].attrs:
                rot_names = {**rot_names, **{v: _ds[v].mate}}

        _ds = _ds.rename(rot_names)
        _ds = mates(_ds)
    return _ds


def shift_dataset(_ds, dims_c, dims_g):
    """Shifts a dataset along a dimension, setting its first element to zero. Need
    to provide the dimensions in the form of [center, corner] points. This rotation
    is only used in the horizontal, and so dims_c is either one of `i` or `j`, and
    dims_g is either one of `i_g` or `j_g`. The pair most correspond to the same
    dimension.

    _ds: dataset

    dims_c: string, either 'i' or 'j'

    dims_g: string, either 'i_g' or 'j_g'. Should correspond to same dimension as
        dims_c.

    """
    if type(_ds) == _dstype:  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)
        for _dim in [dims_c, dims_g]:
            if int(_ds[_dim][0].data) < int(_ds[_dim][1].data):
                _ds["n" + _dim] = _ds[_dim] - int(_ds[_dim][0].data)
                _ds = (
                    _ds.swap_dims({_dim: "n" + _dim})
                    .drop_vars([_dim])
                    .rename({"n" + _dim: _dim})
                )

        _ds = mates(_ds)
    return _ds


def reverse_dataset(_ds, dims_c, dims_g, transpose=False):
    """Reverses the dataset along a dimension. Need to provide the dimensions in the
    form of [center, corner] points. This rotation is only used in the horizontal, and
    so dims_c is either one of `i`  or `j`, and dims_g is either one of `i_g` or `j_g`.
    The pair most correspond to the same dimension."""

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
    """Rotates a dataset along its horizontal dimensions (e.g. center and corner). It
    can also shift the dataset along a dimension, reserve its orientaton and transpose
    the whole dataset.

    _ds : dataset

    dims_c = [dims_c.X, dims_c.Y]
    dims_g = [dims_g.X, dims_g.Y]

    nface=1: flag. A single dataset is being manipulated.
    nface=int: correct number to use. This is the case a merger/concatenated dataset is
    being manipulated. Nij is no longer the size of the face.
    """
    if type(_ds) == _dstype:  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)
        Nij = max(len(_ds[dims_c.X]), len(_ds[dims_c.Y]))

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
    """Given a list of n-datasets with matching dimensions, each element of the list
    gets shifted along the dimensions provided (by dims_c and dims_g) so that there
    is no overlap of values between them.
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
    """Combines a list of N-xarray.datasets along a dimension. Datasets must have
    matching dimensions. See `xr.combine_first()`

    """
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
                _DSFacet = _DSlist[0].combine_first(_DSlist[1])  #
    elif len(_DSlist) > 2:
        _DSFacet = _copy.deepcopy(_DSlist[0])
        for ii in range(1, len(_DSlist)):
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                _DSFacet = _DSFacet.combine_first(_copy.deepcopy(_DSlist[ii]))

        _DSFacet = mates(_DSFacet)

    return _DSFacet


def flip_v(_ds, co_list=metrics, dims=True, _len=3):
    """Reverses the sign of the vector fields by default along the corner coordinate
    (Xp1 or Yp1). If dims is True, for each variable we infer the dimensions. Otherwise,
    dims is given

    """
    if type(_ds) == _dstype:
        for _varName in _ds.variables:
            if dims:
                DIMS = [dim for dim in _ds[_varName].dims if dim != "face"]
                _dims = Dims(DIMS[::-1])
            if "mate" in _ds[_varName].attrs:
                if _varName not in co_list and len(_dims.X) == _len:
                    _ds[_varName] = -_ds[_varName]
                elif _varName == "SN":
                    _ds[_varName] = -_ds[_varName]
                # elif _varName == "CS":
                #     _ds[_varName] = -_ds[_varName]
    return _ds


def _edge_arc_data(_da, _face_ind, _dims):
    """Determines the edge of the non-masked data values on each of the four triangles
    that make up the arctic cap and that that will be retained (not dropped) in the
    cutout process. Only this subset of the face needs to be transformed.

    Output: Index location of the data edge of face = _face_ind along the geographical
    north dimension.
    """
    if _face_ind == 5:  # finds the first nan value along local y dim.
        _value = True
        _dim = _dims.Y
        _shift = -1  # shifts back a position to the edge of the data.
    elif _face_ind == 2:  # finds the first nan value along local x dim.
        _value = True
        _dim = _dims.X
        _shift = -1  # shifts back a position to the edge of the data.
    elif _face_ind == 10:  # find the first value along local y.
        _value = False
        _dim = _dims.Y
        _shift = 0  # shifts back a position to the edge of the data.
    elif _face_ind == 7:
        _value = False
        _dim = _dims.X
        _shift = 0

    _da = _da.load()  # load into memory single face single variable.
    for i in list(_da[_dim].data):
        arg = {_dim: i}
        if _np.isnan(_np.array(_da.sel(**arg).data)).all() == _value:
            X0 = i + _shift
            break
    return X0


def mask_var(_ds, XRange=None, YRange=None, ref_lon=180):
    """Returns a dataset with masked latitude at C and G points (YC and YG).
    The masking region is determined by XRange and YRange. Used to estimate the
    extend of actual data to be retained.
    """

    _ds = _copy.deepcopy(mates(_ds.reset_coords()))

    nYG = _copy.deepcopy(_ds["YG"])
    _ds["nYG"] = nYG

    if YRange is None:
        minY = _ds["YG"].min().values
        maxY = _ds["YG"].max().values
    else:
        minY = YRange[0]
        maxY = YRange[1]

    if XRange is None:
        minX = _ds["XG"].min().values
        maxX = _ds["XG"].max().values
    else:
        minX = XRange[0]
        maxX = XRange[1]

    maskG = _xr.where(
        _np.logical_and(
            _np.logical_and(_ds["YG"] >= minY, _ds["YG"] <= maxY),
            _np.logical_and(
                _rel_lon(_ds["XG"], ref_lon) >= _rel_lon(minX, ref_lon),
                _rel_lon(_ds["XG"], ref_lon) <= _rel_lon(maxX, ref_lon),
            ),
        ),
        1,
        0,
    ).persist()

    _ds["nYG"] = _ds["nYG"].where(maskG, drop=True)
    return _ds


def arc_limits_mask(_ds, _var, _faces, _dims, XRange, YRange):
    """Estimates the limits of the masking region of the arctic."""
    dsa2 = []
    dsa5 = []
    dsa7 = []
    dsa10 = []
    ARCT = [dsa2, dsa5, dsa7, dsa10]

    *nnn, DS = arct_connect(
        _ds, _var, faces=_faces, masking=True, opt=False
    )  # This only works in the case the transformation involves the whole domain
    ARCT[0].append(DS[0])
    ARCT[1].append(DS[1])
    ARCT[2].append(DS[2])
    ARCT[3].append(DS[3])

    for i in range(len(ARCT)):  # Not all faces survive the cutout
        if type(ARCT[i][0]) == _datype:
            ARCT[i] = _xr.merge(ARCT[i])

    DSa2, DSa5, DSa7, DSa10 = ARCT

    if type(DSa2) != _dstype:
        DSa2 = 0
        [Xi_2, Xf_2] = [0, 0]
    else:
        if XRange is None and YRange is None:
            Xf_2 = int(DSa2[_var][_dims.X][-1])
        else:
            Xf_2 = _edge_arc_data(DSa2[_var], 2, _dims)
        Xi_2 = int(DSa2[_var][_dims.X][0])
    if type(DSa5) != _dstype:
        DSa5 = 0
        [Yi_5, Yf_5] = [0, 0]
    else:
        if XRange is None and YRange is None:
            Yf_5 = int(DSa5[_var][_dims.Y][-1])
        else:
            Yf_5 = _edge_arc_data(DSa5[_var], 5, _dims)
        Yi_5 = int(DSa5[_var][_dims.Y][0])
    if type(DSa7) != _dstype:
        DSa7 = 0
        [Xi_7, Xf_7] = [0, 0]
    else:
        if XRange is None and YRange is None:
            Xi_7 = int(DSa7[_var][_dims.X][0])
        else:
            Xi_7 = _edge_arc_data(DSa7[_var], 7, _dims)
        Xf_7 = int(DSa7[_var][_dims.X][-1])

    if type(DSa10) != _dstype:
        DSa10 = 0
        [Yi_10, Yf_10] = [0, 0]
    else:
        if XRange is None and YRange is None:
            Yi_10 = int(DSa10[_var][_dims.Y][0])
        else:
            Yi_10 = _edge_arc_data(DSa10[_var], 10, _dims)
        Yf_10 = int(DSa10[_var][_dims.Y][-1])

    arc_edges = [[Xi_2, Xf_2], [Yi_5, Yf_5], [Xi_7, Xf_7], [Yi_10, Yf_10]]

    return arc_edges


def _edge_facet_data(_Facet_list, _var, _dims, _axis):
    """Determines the edge of the non-masked data values on each of the four Facets,
    and that that will be retained (not dropped) in the cutout process.
    Only this subset of the face needs to be transformed.

    Output: Index location of the data edge of face = _face_ind along the geographical
    north dimension.
    """
    if _axis == 0:
        _dim = _dims.Y
    elif _axis == 1:
        _dim = _dims.X

    XRange = []
    for i in range(len(_Facet_list)):
        if type(_Facet_list[i]) == _dstype:
            # there is data
            _da = _Facet_list[i][_var].load()  # load into memory 2d data.
            X0 = []
            for j in list(_da[_dim].data):
                arg = {_dim: j}
                if _np.isnan(_np.array(_da.sel(**arg).data)).all():
                    X0.append(0)
                else:
                    X0.append(1)
            x0 = _np.where(_np.array(X0) == 1)[0][0]
            xf = _np.where(_np.array(X0) == 1)[0][-1]
            XRange.append([x0, xf])  # xf+1?
        else:
            XRange.append([_np.nan, _np.nan])  # no data
    return XRange


def slice_datasets(_DSfacet, dims_c, dims_g, _edges, _axis):
    """
    Slices a list of dataset along an axis. The range of index retained is
    defined in Ranges, an argument of the function. How the list of dataset,
    which together define a Facet with facet index (1-4), depends on the facet
    index and the axis (0 or 1).
    """
    if _axis == 0:  # local y always the case for all facets
        _dim_c = dims_c.Y
        _dim_g = dims_g.Y
    elif _axis == 1:  # local x always the case.
        _dim_c = dims_c.X
        _dim_g = dims_g.X

    _DSFacet = _copy.deepcopy(_DSfacet)
    for i in range(len(_DSFacet)):
        # print(i)
        if type(_DSFacet[i]) == _dstype:
            for _dim in [_dim_c, _dim_g]:
                if len(_edges) == 1:
                    ii_0 = int(_edges[0])
                    ii_1 = int(_edges[1])
                else:
                    ii_0 = int(_edges[i][0])
                    ii_1 = int(_edges[i][1])
                arg = {_dim: slice(ii_0, ii_1 + 1)}
                _DSFacet[i] = _DSFacet[i].isel(**arg)
    return _DSFacet


def _LLC_check_sizes(_DS):
    """
    Checks and asserts len of center and corner points are in agreement.
    """
    YG = _DS["YG"].dropna("Yp1", "all")
    y0 = int(YG["Yp1"][0])
    y1 = int(YG["Yp1"][-1]) + 1

    _DS = _copy.deepcopy(_DS.isel(Yp1=slice(y0, y1)))
    _DS = _copy.deepcopy(_DS.isel(Y=slice(y0, y1 - 1)))

    DIMS = [dim for dim in _DS["XC"].dims]
    dims_c = Dims(DIMS[::-1])

    DIMS = [dim for dim in _DS["XG"].dims]
    dims_g = Dims(DIMS[::-1])

    # total number of scalar points.
    Nx_c = len(_DS[dims_c.X])
    Ny_c = len(_DS[dims_c.Y])
    Nx_g = len(_DS[dims_g.X])
    Ny_g = len(_DS[dims_g.Y])

    if Nx_c == Nx_g:
        arg = {dims_c.X: slice(0, -1)}
        _DS = _copy.deepcopy(_DS.isel(**arg))
    else:
        delta = Nx_g - Nx_c
        if delta < 0:
            raise ValueError(
                "Inconsistent sizes at corner (_g) and center (_c) points"
                "after cutout `len(_g) < len(_c)."
            )
        else:
            if delta == 2:  # len(_g) = len(_c)+2. Can but shouldn't happen.
                arg = {dims_g: slice(0, -1)}
                _DS = _copy.deepcopy(_DS.isel(**arg))
    if Ny_c == Ny_g:
        arg = {dims_c.Y: slice(0, -1)}
        _DS = _copy.deepcopy(_DS.isel(**arg))

    return _DS


def _reorder_ds(_ds, dims_c, dims_g):
    """Only needed when Pacific-centered data. Corrects the ordering
    of y-dim and transposes the data, all lazily."""

    _DS = _copy.deepcopy(_ds)
    for _dim in [dims_c.Y, dims_g.Y]:
        _DS["n" + _dim] = -(_DS[_dim] - (int(_DS[_dim][0].data)))
        _DS = (
            _DS.swap_dims({_dim: "n" + _dim})
            .drop_vars([_dim])
            .rename({"n" + _dim: _dim})
        )

    for var in _ds.data_vars:
        DIMS = [dim for dim in _ds[var].dims]
        dims = Dims(DIMS[::-1])
        if len(dims) > 1 and "nv" not in DIMS:
            dtr = list(dims)[::-1]
            dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
            _da = _ds[var].transpose(*dtr)[::-1, :]
            _DS[var] = _da
    return _DS


def llc_local_to_lat_lon(ds, co_list=metrics):
    """
    Takes all vector fields and rotates them to orient them along geographical
    coordinates.
    """
    _ds = mates(_copy.deepcopy(ds))

    grid_coords = {
        "Y": {"center": "Y", "outer": "Yp1"},
        "X": {"center": "X", "outer": "Xp1"},
        "Z": {"center": "Z", "outer": "Zp1", "right": "Zu", "left": "Zl"},
        "time": {"center": "time_midp", "left": "time"},
    }

    # create grid object to interpolate
    grid = Grid(_ds, coords=grid_coords, periodic=[])

    CS = _ds["CS"]  # cosine of angle between logical and geo axis. At tracer points
    SN = _ds["SN"]  # sine of angle between logical and geo axis. At tracer points

    CSU = grid.interp(CS, axis="X", boundary="extend")  # cos at u-point
    CSV = grid.interp(CS, axis="Y", boundary="extend")  # cos at v-point

    SNU = grid.interp(SN, axis="X", boundary="extend")  # sin at u-point
    SNV = grid.interp(SN, axis="Y", boundary="extend")  # sin at v-point

    data_vars = [var for var in _ds.data_vars if len(ds[var].dims) > 1]

    for var in data_vars:
        DIMS = [dim for dim in _ds[var].dims]
        dims = Dims(DIMS[::-1])
        if len(dims.X) + len(dims.Y) == 4:  # vector field (metric)
            if len(dims.Y) == 1 and var not in co_list:  # u vector
                _da = _copy.deepcopy(_ds[var])
                if "mate" in _ds[var].attrs:
                    mate = _ds[var].mate
                    _ds = _ds.drop_vars([var])
                    VU = grid.interp(
                        grid.interp(_ds[mate], axis="Y", boundary="extend"),
                        axis="X",
                        boundary="extend",
                    )
                    _ds[var] = _da * CSU - VU * SNU
            elif len(dims.Y) == 3 and var not in co_list:  # v vector
                _da = _copy.deepcopy(_ds[var])
                if "mate" in _ds[var].attrs:
                    mate = _ds[var].mate
                    _ds = _ds.drop_vars([var])
                    UV = grid.interp(
                        grid.interp(_ds[mate], axis="X", boundary="extend"),
                        axis="Y",
                        boundary="extend",
                    )
                    _ds[var] = UV * SNV + _da * CSV

    return _ds


class Dims:
    """Creates a shortcut for dimension`s names associated with an arbitrary
    variable."""

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
