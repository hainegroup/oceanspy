"""
OceanSpy functionality that transforms a dataset with LLC geometry characterized by
13 faces (or tiles), into one with simple geometry.
"""

import copy as _copy
import reprlib

import dask
import numpy as _np
import xarray as _xr
from shapely import Point, Polygon
from xarray import DataArray, Dataset

from .utils import _rel_lon, _reset_range, connector, get_maskH, reset_dim

# metric variables defined at vector points, defined as global within this file
metrics = [
    "dxC",
    "dyC",
    "dxG",
    "dyG",
    "HFacW",
    "HFacS",
    "rAs",
    "rAw",
    "maskS",
    "maskW",
    "XU",
    "YU",
    "XV",
    "YV",
]


class LLCtransformation(object):
    """A class containing the transformation types of LLCgrids."""

    @classmethod
    def arctic_crown(
        self,
        ds,
        varList=None,
        YRange=None,
        XRange=None,
        add_Hbdr=0,
        faces=None,
        centered=None,
        persist=False,
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
        persist: bool.
            If `False` (default), transformation of rotated and arctic data is not
            persisted. See `xarray.Dataset.persist()`.

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

        https://docs.xarray.dev/en/stable/generated/xarray.Dataset.persist.html

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

        if Nx == 90:
            add_Hbdr = add_Hbdr + 2  # ECCO
        else:  # pragma: no cover
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
                    ds,
                    var_name,
                    faces=faces,
                    masking=False,
                    opt=opt,
                    ranges=cuts,
                    persist=persist,
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
            if all(not isinstance(item, int) for item in ARCT[i]):
                ARCT[i] = _xr.merge(ARCT[i])

        DSa2, DSa5, DSa7, DSa10 = ARCT
        if not isinstance(DSa2, Dataset):
            DSa2 = 0
        if not isinstance(DSa5, Dataset):
            DSa5 = 0
        if not isinstance(DSa7, Dataset):
            DSa7 = 0
        if not isinstance(DSa10, Dataset):
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
            if isinstance(faces2[0], Dataset):
                faces2[0]["Yp1"] = faces2[0]["Yp1"] + 1

            # Arctic exchange with face 2
            if isinstance(faces3[3], Dataset):
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
        if isinstance(DSFacet3, int):
            centered = "Pacific"

        # =====
        # combining all facets
        # =====

        # First, check if there is data in both DSFacet12 and DSFacet34.
        # If not, then there is no need to transpose data in DSFacet12.

        if isinstance(DSFacet12, Dataset):
            if isinstance(DSFacet34, Dataset):
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
                        DSFacet12[_var] = DSFacet12[_var].transpose(*dtr)
                if persist:  # pragma : no cover
                    DSFacet12 = DSFacet12.persist()

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

        if isinstance(DSFacet34, int):
            DS = _reorder_ds(DS, dims_c, dims_g).persist()

        DS = _LLC_check_sizes(DS)

        if "nYG" in DS.reset_coords().data_vars:
            DS = DS.drop_vars(_var_)

        # restore original attrs if lost
        for var in varList:
            if var in DS.reset_coords().data_vars:
                DS[var].attrs = attrs[var]

        return DS


def arct_connect(
    ds, varName, faces=None, masking=False, opt=False, ranges=None, persist=False
):
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
                if opt:  # pragma: no cover
                    [Xi_2, Xf_2] = [ranges[0][0], ranges[0][1]]
                    cu_arg = {dims.X: slice(Xi_2, Xf_2)}
                    arct = arct.sel(**cu_arg) * Mask.sel(**cu_arg)
                    if persist:  # pragma: no cover
                        arct = arct.persist()
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
                if opt:  # pragma: no cover
                    [Yi_5, Yf_5] = [ranges[1][0], ranges[1][1]]
                    cu_arg = {dims.Y: slice(Yi_5, Yf_5)}
                    arct = arct.sel(**cu_arg) * Mask.sel(**cu_arg)
                    if persist:  # pragma: no cover
                        arct = arct.persist()
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
                if opt:  # pragma : no cover
                    [Xi_7, Xf_7] = [ranges[2][0], ranges[2][1]]
                    cu_arg = {dims.X: slice(Xi_7, Xf_7)}
                    arct = arct.sel(**cu_arg) * Mask.sel(**cu_arg)
                    if persist:  # pragma : no cover
                        arct = arct.persist()
                else:  # pragma: no cover
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
                if masking:  # pragma: no cover
                    if opt:
                        [Yi_10, Yf_10] = [ranges[-1][0], ranges[-1][1]]
                        cu_arg = {dims.Y: slice(Yi_10, Yf_10)}
                        arct = arct.sel(**cu_arg) * Mask.sel(**cu_arg)
                    else:
                        arct = arct * Mask
                else:  # pragma: no cover
                    if opt:
                        [Yi_10, Yf_10] = [ranges[-1][0], ranges[-1][1]]
                        cu_arg = {dims.Y: slice(Yi_10, Yf_10)}
                        arct = (arct.sel(**cu_arg) * Mask.sel(**cu_arg)).transpose(*dtr)
                        if persist:  # pragma: no cover
                            arct = arct.persist()
                    else:
                        arct = (arct * Mask).transpose(*dtr)
                ARCT[3] = arct

    return arc_faces, Nx_ac_nrot, Ny_ac_nrot, Nx_ac_rot, Ny_ac_rot, ARCT


def mates(ds, pair=[]):
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
        "oceTAUX",
        "oceTAUY",
        "U",
        "V",
        "UVELMASS",
        "VVELMASS",
        "XU",
        "YV",
        "XV",
        "YU",
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
        "SIuice",
        "SIvice",
    ]

    if len(pair) > 0 and len(pair) % 2 == 0:
        vars_mates += pair

    for k in range(int(len(vars_mates) / 2)):
        nk = 2 * k
        if vars_mates[nk] in ds.variables:
            ds[vars_mates[nk]].attrs["mate"] = vars_mates[nk + 1]
            ds[vars_mates[nk + 1]].attrs["mate"] = vars_mates[nk]
        elif vars_mates[nk] in pair and vars_mates[nk] not in ds.variables:
            raise ValueError(
                "Variable pair `vars` [{}, {}] not present in dataset."
                "".format(vars_mates[nk], vars_mates[nk + 1])
            )
    return ds


def rotate_vars(_ds):
    """Using the attribures `mates`, when this function is called it swaps the
    variables names. This issue is only applicable to llc grid in which the grid
    topology makes it so that u on a rotated face transforms to `+- v` on a lat lon
    grid.
    """
    if isinstance(_ds, Dataset):  # if a dataset transform otherwise pass
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
    if isinstance(_ds, Dataset):  # if a dataset transform otherwise pass
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

    if isinstance(_ds, Dataset):  # if a dataset transform otherwise pass
        _ds = _copy.deepcopy(_ds)

        for _dim in [dims_c, dims_g]:  # This part should be different for j_g points?
            _ds["n" + _dim] = -_ds[_dim] + int(_ds[_dim][-1].data)
            _ds = (
                _ds.swap_dims({_dim: "n" + _dim})
                .drop_vars([_dim])
                .rename({"n" + _dim: _dim})
            )

        _ds = mates(_ds)

        if transpose:  # pragma: no cover
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
    if isinstance(_ds, Dataset):  # if a dataset transform otherwise pass
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
            if isinstance(_DS[ii - 1], int):
                dim0 = int(Ni * sum(facs[:ii]))
            else:
                for _dim in [dims_c, dims_g]:
                    dim0 = int(_DS[ii - 1][_dim][-1].data + 1)
            if isinstance(_DS[ii], Dataset):
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
            if isinstance(_DS[lll], Dataset):
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
        if isinstance(_DSlist[0], int):  # one is empty, pass directly
            _DSFacet = _DSlist[1]
        elif isinstance(_DSlist[1], int):  # the other is empty pass directly
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
    if isinstance(_ds, Dataset):
        for _varName in _ds.variables:
            if dims:  # pragma: no cover
                DIMS = [dim for dim in _ds[_varName].dims if dim != "face"]
                _dims = Dims(DIMS[::-1])
            if "mate" in _ds[_varName].attrs:
                if _varName not in co_list and len(_dims.X) == _len:
                    _ds[_varName] = -_ds[_varName]
                elif _varName == "SN":
                    _ds[_varName] = -_ds[_varName]
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

    _ds["nYG"] = _ds["nYG"].where(maskG.compute(), drop=True)
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
        if isinstance(ARCT[i][0], DataArray):
            ARCT[i] = _xr.merge(ARCT[i])

    DSa2, DSa5, DSa7, DSa10 = ARCT

    if not isinstance(DSa2, Dataset):
        DSa2 = 0
        [Xi_2, Xf_2] = [0, 0]
    else:
        if XRange is None and YRange is None:
            Xf_2 = int(DSa2[_var][_dims.X][-1])
        else:
            Xf_2 = _edge_arc_data(DSa2[_var], 2, _dims)
        Xi_2 = int(DSa2[_var][_dims.X][0])
    if not isinstance(DSa5, Dataset):
        DSa5 = 0
        [Yi_5, Yf_5] = [0, 0]
    else:
        if XRange is None and YRange is None:
            Yf_5 = int(DSa5[_var][_dims.Y][-1])
        else:
            Yf_5 = _edge_arc_data(DSa5[_var], 5, _dims)
        Yi_5 = int(DSa5[_var][_dims.Y][0])
    if not isinstance(DSa7, Dataset):
        DSa7 = 0
        [Xi_7, Xf_7] = [0, 0]
    else:
        if XRange is None and YRange is None:
            Xi_7 = int(DSa7[_var][_dims.X][0])
        else:
            Xi_7 = _edge_arc_data(DSa7[_var], 7, _dims)
        Xf_7 = int(DSa7[_var][_dims.X][-1])

    if not isinstance(DSa10, Dataset):
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
        if isinstance(_Facet_list[i], Dataset):
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
    if _axis == 0:  # pragma: no cover
        # local y always the case for all facets
        _dim_c = dims_c.Y
        _dim_g = dims_g.Y
    elif _axis == 1:  # pragma: no cover
        # local x always the case.
        _dim_c = dims_c.X
        _dim_g = dims_g.X

    _DSFacet = _copy.deepcopy(_DSfacet)
    for i in range(len(_DSFacet)):
        # print(i)
        if isinstance(_DSFacet[i], Dataset):
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
        Nx_c = len(_DS[dims_c.X])
    else:
        delta = Nx_g - Nx_c
        if delta < 0:  # pragma: no cover
            raise ValueError(
                "Inconsistent sizes at corner (_g) and center (_c) points"
                "after cutout `len(_g) < len(_c)."
            )
        else:
            if delta == 2:  # len(_g) = len(_c)+2. Can but shouldn't happen.
                arg = {dims_g.X: slice(0, -1)}
                _DS = _copy.deepcopy(_DS.isel(**arg))
                Nx_g = len(_DS[dims_g.X])

    if Ny_c == Ny_g:
        arg = {dims_c.Y: slice(0, -1)}
        _DS = _copy.deepcopy(_DS.isel(**arg))
        Ny_c = len(_DS[dims_c.Y])

    # lastly, make sure that core dimensions are chunked consistently

    chunks = {"X": Nx_c, "Xp1": Nx_g, "Y": Ny_c, "Yp1": Ny_g}

    return _DS.chunk(**chunks)


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


def eval_dataset(_ds, _ix, _iy, _iface=None, _dim_name="mooring"):
    """
    Evaluates a dataset along (spatial) trajectory in the plane as defined by the
    indexes in the plane.
    The data in the new xarray.dataset has a new dimension/coordinate.

    Parameters:
    ----------
    _ds: xarray.Dataset
        contains all x, y coordinates (but may be subsampled in Z or time)
    _ix, _iy: 1D array, int
        index values identifying the location in X Y (lat, lon) space
    _iface: int, None (bool)
        None (default) implies no complex topology in the dataset. Otherwise,
        _iface indicates the face index which, along which the provided ix, iy,
        identify the spatial (geo) coordinate location in lat/lon space.
    _dim_name: str
        names the new dimension along the pathway. By default this is 'mooring',
        but can also be 'station' (when discrete, argo-like isolated coordinates).

    Returns:
        xarray.Dataset
    """

    nz = len(_ds.Z)
    nzu = len(_ds.Zu)
    nzp1 = len(_ds.Zp1)
    nzl = len(_ds.Zl)

    # rechunk in time and z
    chunks = {"Z": nz, "Zu": nzu, "Zp1": nzp1, "Zl": nzl}
    _ds = _ds.chunk(chunks)

    new_dim = DataArray(
        _np.arange(len(_ix)),
        dims=(_dim_name),
        attrs={"long_name": "index of " + _dim_name, "units": "none"},
    )
    y = DataArray(
        _np.arange(1),
        dims=("y"),
        attrs={"long_name": "j-index of cell center", "units": "none"},
    )
    x = DataArray(
        _np.arange(1),
        dims=("x"),
        attrs={"long_name": "i-index of cell center", "units": "none"},
    )
    yp1 = DataArray(
        _np.arange(2),
        dims=("yp1"),
        attrs={"long_name": "j-index of cell corner", "units": "none"},
    )
    xp1 = DataArray(
        _np.arange(2),
        dims=("xp1"),
        attrs={"long_name": "i-index of cell corner", "units": "none"},
    )

    # Transform indexes in DataArray
    iY = DataArray(
        _np.reshape(_iy, (len(new_dim), len(y))),
        coords={_dim_name: new_dim, "y": y},
        dims=(_dim_name, "y"),
    )
    iX = DataArray(
        _np.reshape(_ix, (len(new_dim), len(x))),
        coords={_dim_name: new_dim, "x": x},
        dims=(_dim_name, "x"),
    )

    iYp1 = DataArray(
        _np.stack((_iy, _iy + 1), 1),
        coords={_dim_name: new_dim, "yp1": yp1},
        dims=(_dim_name, "yp1"),
    )

    iXp1 = DataArray(
        _np.stack((_ix, _ix + 1), 1),
        coords={_dim_name: new_dim, "xp1": xp1},
        dims=(_dim_name, "xp1"),
    )

    if _iface is not None:
        if _iface == [6]:
            return arctic_eval(_ds, _ix, _iy, _dim_name)
        elif _iface in _np.arange(7, 13):
            iXp1 = DataArray(
                _np.stack((_ix + 1, _ix), 1),
                coords={_dim_name: new_dim, "xp1": xp1},
                dims=(_dim_name, "xp1"),
            )

    args = {
        "X": iX,
        "Y": iY,
        "Xp1": iXp1,
        "Yp1": iYp1,
    }

    rename = {"yp1": "Yp1", "xp1": "Xp1", "x": "X", "y": "Y"}

    if _iface is not None:
        args = {"face": _iface, **args}
        if _iface in _np.arange(7, 13):
            rename = {"yp1": "Xp1", "xp1": "Yp1", "x": "Y", "y": "X"}

    new_ds = _ds.isel(**args).drop_vars(["Xp1", "Yp1", "X", "Y"])
    new_ds = new_ds.rename_dims(rename).rename_vars(rename)
    if _iface is not None and _iface in _np.arange(7, 13):
        new_ds = rotate_vars(new_ds)

    if "face" in new_ds.reset_coords().data_vars:
        new_ds = new_ds.drop_vars(["face"])

    return new_ds


def arctic_eval(_ds, _ix, _iy, _dim_name="mooring"):
    """
    Evaluates a dataset
    """
    _ds = mates(_ds.isel(face=6))

    nz = len(_ds.Z)
    nzu = len(_ds.Zu)
    nzp1 = len(_ds.Zp1)
    nzl = len(_ds.Zl)

    # rechunk in time and z
    chunks = {"Z": nz, "Zu": nzu, "Zp1": nzp1, "Zl": nzl}
    _ds = _ds.chunk(chunks)

    _XC = _ds.reset_coords()["XC"]
    XR5 = _np.min(_XC.isel(Y=0, X=0).values), _np.max(_XC.isel(Y=0, X=-1).values)
    XR2 = _np.min(_XC.isel(X=0, Y=-1).values), _np.max(_XC.isel(X=0, Y=0).values)
    XR7 = _np.min(_XC.isel(X=-1, Y=0).values), _np.max(_XC.isel(X=-1, Y=-1).values)
    XR10 = _np.min(_XC.isel(X=-1, Y=-1).values), _np.max(_XC.isel(X=0, Y=-1).values)
    co_list = [var for var in _ds.coords]

    y = DataArray(
        _np.arange(1),
        dims=("y"),
        attrs={"long_name": "j-index of cell center", "units": "none"},
    )
    x = DataArray(
        _np.arange(1),
        dims=("x"),
        attrs={"long_name": "i-index of cell center", "units": "none"},
    )
    yp1 = DataArray(
        _np.arange(2),
        dims=("yp1"),
        attrs={"long_name": "j-index of cell corner", "units": "none"},
    )
    xp1 = DataArray(
        _np.arange(2),
        dims=("xp1"),
        attrs={"long_name": "i-index of cell corner", "units": "none"},
    )

    # get all lons values
    nY = DataArray(_iy, coords={"temp_dim": _np.arange(len(_iy))}, dims=("temp_dim",))
    nX = DataArray(_ix, coords={"temp_dim": _np.arange(len(_iy))}, dims=("temp_dim",))

    p = _XC.isel(X=nX, Y=nY).compute().data

    # cluster points by lon ranges
    p2 = _np.argwhere(_np.logical_and(p > XR2[0], p <= XR2[-1])).flatten()
    p5 = _np.argwhere(_np.logical_and(p > XR5[0], p <= XR5[-1])).flatten()
    p7 = _np.argwhere(_np.logical_or(p > XR7[0], p <= XR7[-1])).flatten()
    p10 = _np.argwhere(_np.logical_and(p > XR10[0], p <= XR10[-1])).flatten()

    Ps = [p2, p5, p7, p10]

    Regs = [2, 5, 7, 10]  # these are face connections

    attrs = {"long_name": "index of " + _dim_name, "units": "none"}

    DS = []

    for i in range(len(Ps)):
        if len(Ps[i]) > 0 and Regs[i] == 2:  # XR2
            new_dim = DataArray(
                Ps[i],
                dims=(_dim_name),
                attrs=attrs,
            )
            iY = DataArray(
                _np.reshape(_iy[Ps[i]], (len(new_dim), len(y))),
                coords={_dim_name: new_dim, "y": y},
                dims=(_dim_name, "y"),
            )
            iX = DataArray(
                _np.reshape(_ix[Ps[i]], (len(new_dim), len(x))),
                coords={_dim_name: new_dim, "x": x},
                dims=(_dim_name, "x"),
            )

            iYp1 = DataArray(
                _np.stack((_iy[Ps[i]], _iy[Ps[i]] + 1)[::-1], 1),
                coords={_dim_name: new_dim, "yp1": yp1},
                dims=(_dim_name, "yp1"),
            )
            iXp1 = DataArray(
                _np.stack((_ix[Ps[i]], _ix[Ps[i]] + 1), 1),
                coords={_dim_name: new_dim, "xp1": xp1},
                dims=(_dim_name, "xp1"),
            )
            args = {
                "X": iX,
                "Y": iY,
                "Xp1": iXp1,
                "Yp1": iYp1,
            }
            rename = {"yp1": "Xp1", "xp1": "Yp1", "x": "Y", "y": "X"}
            new_ds = _ds.isel(**args).drop_vars(["Xp1", "Yp1", "X", "Y"])
            new_ds = new_ds.rename_dims(rename).rename_vars(rename)
            new_ds = rotate_vars(new_ds)

            for _varName in new_ds.variables:
                if "mate" in new_ds[_varName].attrs:
                    _dims = new_ds[_varName].dims
                    if _varName not in co_list and "Xp1" in _dims:
                        new_ds[_varName] = -new_ds[_varName]
                if _varName == "SN":
                    new_ds[_varName] = -new_ds[_varName]

        if len(Ps[i]) > 0 and Regs[i] == 5:  # XR5
            new_dim = DataArray(
                Ps[i],
                dims=(_dim_name),
                attrs=attrs,
            )

            iY = DataArray(
                _np.reshape(_iy[Ps[i]], (len(new_dim), len(y))),
                coords={_dim_name: new_dim, "y": y},
                dims=(_dim_name, "y"),
            )
            iX = DataArray(
                _np.reshape(_ix[Ps[i]], (len(new_dim), len(x))),
                coords={_dim_name: new_dim, "x": x},
                dims=(_dim_name, "x"),
            )

            iYp1 = DataArray(
                _np.stack((_iy[Ps[i]], _iy[Ps[i]] + 1), 1),
                coords={_dim_name: new_dim, "yp1": yp1},
                dims=(_dim_name, "yp1"),
            )
            iXp1 = DataArray(
                _np.stack((_ix[Ps[i]], _ix[Ps[i]] + 1), 1),
                coords={_dim_name: new_dim, "xp1": xp1},
                dims=(_dim_name, "xp1"),
            )
            args = {
                "X": iX,
                "Y": iY,
                "Xp1": iXp1,
                "Yp1": iYp1,
            }
            rename = {"yp1": "Yp1", "xp1": "Xp1", "x": "X", "y": "Y"}

            new_ds = _ds.isel(**args).drop_vars(["Xp1", "Yp1", "X", "Y"])
            new_ds = new_ds.rename_dims(rename).rename_vars(rename)

        if len(Ps[i]) > 0 and Regs[i] == 7:  # XR7
            new_dim = DataArray(
                Ps[i],
                dims=(_dim_name),
                attrs=attrs,
            )

            iY = DataArray(
                _np.reshape(_iy[Ps[i]], (len(new_dim), len(y))),
                coords={_dim_name: new_dim, "y": y},
                dims=(_dim_name, "y"),
            )
            iX = DataArray(
                _np.reshape(_ix[Ps[i]], (len(new_dim), len(x))),
                coords={_dim_name: new_dim, "x": x},
                dims=(_dim_name, "x"),
            )

            iYp1 = DataArray(
                _np.stack((_iy[Ps[i]], _iy[Ps[i]] + 1), 1),
                coords={_dim_name: new_dim, "yp1": yp1},
                dims=(_dim_name, "yp1"),
            )
            iXp1 = DataArray(
                _np.stack((_ix[Ps[i]], _ix[Ps[i]] + 1)[::-1], 1),
                coords={_dim_name: new_dim, "xp1": xp1},
                dims=(_dim_name, "xp1"),
            )
            args = {
                "X": iX,
                "Y": iY,
                "Xp1": iXp1,
                "Yp1": iYp1,
            }
            rename = {"yp1": "Xp1", "xp1": "Yp1", "x": "Y", "y": "X"}
            new_ds = _ds.isel(**args).drop_vars(["Xp1", "Yp1", "X", "Y"])
            new_ds = new_ds.rename_dims(rename).rename_vars(rename)
            new_ds = rotate_vars(new_ds)

            for _varName in new_ds.variables:
                if "mate" in new_ds[_varName].attrs:
                    _dims = new_ds[_varName].dims
                    if _varName not in co_list and "Yp1" in _dims:
                        new_ds[_varName] = -new_ds[_varName]
                if _varName == "CS":
                    new_ds[_varName] = -new_ds[_varName]

        if len(Ps[i]) > 0 and Regs[i] == 10:  # XR10
            # elif p > XR10[0] and p < XR10[-1]:
            new_dim = DataArray(
                Ps[i],
                dims=(_dim_name),
                attrs=attrs,
            )

            iY = DataArray(
                _np.reshape(_iy[Ps[i]], (len(new_dim), len(y))),
                coords={_dim_name: new_dim, "y": y},
                dims=(_dim_name, "y"),
            )
            iX = DataArray(
                _np.reshape(_ix[Ps[i]], (len(new_dim), len(x))),
                coords={_dim_name: new_dim, "x": x},
                dims=(_dim_name, "x"),
            )

            iYp1 = DataArray(
                _np.stack((_iy[Ps[i]], _iy[Ps[i]] + 1)[::-1], 1),
                coords={_dim_name: new_dim, "yp1": yp1},
                dims=(_dim_name, "yp1"),
            )

            iXp1 = DataArray(
                _np.stack((_ix[Ps[i]], _ix[Ps[i]] + 1)[::-1], 1),
                coords={_dim_name: new_dim, "xp1": xp1},
                dims=(_dim_name, "xp1"),
            )
            args = {
                "X": iX,
                "Y": iY,
                "Xp1": iXp1,
                "Yp1": iYp1,
            }
            rename = {"yp1": "Yp1", "xp1": "Xp1", "x": "X", "y": "Y"}

            new_ds = _ds.isel(**args).drop_vars(["Xp1", "Yp1", "X", "Y"])
            new_ds = new_ds.rename_dims(rename).rename_vars(rename)

            for _varName in new_ds.variables:
                if "mate" in new_ds[_varName].attrs:
                    _dims = new_ds[_varName].dims
                    if _varName not in co_list and ("Yp1" in _dims or "Xp1" in _dims):
                        new_ds[_varName] = -new_ds[_varName]
                if _varName == "SN":
                    new_ds[_varName] = -new_ds[_varName]
                if _varName == "CS":
                    new_ds[_varName] = -new_ds[_varName]
        if len(Ps[i]) > 0:
            DS.append(new_ds)
    if len(DS) > 1:
        new_ds = _xr.concat(DS, dim=_dim_name).sortby(_dim_name)
    elif len(DS) == 1:  # pragma: no cover
        new_ds = DS[0]
    return new_ds


def ds_edge_sametx(_ds, iX, iY, iXp1, iYp1, face1, face2, _dim, moor, **kwargs):
    _Nx = len(_ds.X) - 1
    rotS = _np.arange(7, 13)

    dim_arg = {_dim: moor}
    iXn = iX.isel(**dim_arg)  #
    iYn = iY.isel(**dim_arg)  #

    uvars = kwargs["u"]
    vvars = kwargs["v"]
    gvars = kwargs["g"]
    cvars = kwargs["c"]

    args = {"xp1": slice(1)}
    rename = {"x": "xp1"}
    revar = "xp1"
    iXp1n = iXp1.isel(**dim_arg, **args)
    iYp1n = iYp1.isel(**dim_arg)
    iargs = {"Y": iYn, "Yp1": iYp1n, "Xp1": iXp1n - _Nx}

    vds = _ds.isel(face=face2, **iargs)
    vds = vds.reset_coords()[uvars + gvars]

    # get the rest of the points
    argsn = {
        "face": face1,
        "X": iXn,
        "Y": iYn,
        "Xp1": iXp1n,
        "Yp1": iYp1n,
    }
    mds = _ds.isel(**argsn).reset_coords()  # regular eval
    mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])
    if face1 in rotS:
        mds = reset_dim(mds, 1, revar)
    elif face1 not in rotS:
        vds = reset_dim(vds, 1, revar)
    ugmds = _xr.combine_by_coords([mds[uvars + gvars], vds])

    cvmds = mds.reset_coords()[cvars + vvars]
    nds = _xr.combine_by_coords([cvmds, ugmds])
    co_list = [var for var in nds.data_vars if "time" not in nds[var].dims]
    nds = nds.set_coords(co_list)
    rename = {"x": "X", "y": "Y", "xp1": "Xp1", "yp1": "Yp1"}
    nds = nds.rename_dims(rename).rename_vars(rename)

    return nds, vds, mds


def ds_edge_samety(
    _ds, iX, iY, _ix, xp1, iXp1, iYp1, face1, face2, _dim, moor, **kwargs
):
    """
    same topology, axis=`y`.
    """
    rotS = _np.arange(7, 13)

    _Nx = len(_ds.X) - 1
    dim_arg = {_dim: moor}
    iXn = iX.isel(**dim_arg)  #
    iYn = iY.isel(**dim_arg)  #

    uvars = kwargs["u"]
    vvars = kwargs["v"]
    gvars = kwargs["g"]
    cvars = kwargs["c"]

    args = {"yp1": slice(1)}
    rename = {"y": "yp1"}
    if face1 in rotS:  # reverse the order of x points
        new_dim = DataArray(
            _np.arange(len(_ix)),
            dims=(_dim),
            attrs={"long_name": "index of " + _dim, "units": "none"},
        )

        iXp1 = DataArray(
            _np.stack((_ix, _ix + 1)[::-1], 1),
            coords={_dim: new_dim, "xp1": xp1},
            dims=(_dim, "xp1"),
        )
    iXp1n = iXp1.isel(**dim_arg)
    iYp1n = iYp1.isel(**dim_arg, **args)
    iargs = {"X": iXn, "Xp1": iXp1n, "Yp1": iYp1n - _Nx}
    revar = "yp1"
    vds = _ds.isel(face=face2, **iargs)
    vds = vds.reset_coords()[vvars + gvars]

    # get the rest of the points
    argsn = {
        "face": face1,
        "X": iXn,
        "Y": iYn,
        "Xp1": iXp1n,
        "Yp1": iYp1n,
    }
    mds = _ds.isel(**argsn).reset_coords()  # regular eval
    mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])
    vds = reset_dim(vds, 1, revar)
    vgmds = _xr.combine_by_coords([mds[vvars + gvars], vds])

    cumds = mds.reset_coords()[cvars + uvars]
    nds = _xr.combine_by_coords([cumds, vgmds])
    co_list = [var for var in nds.data_vars if "time" not in nds[var].dims]
    nds = nds.set_coords(co_list)
    rename = {"x": "X", "y": "Y", "xp1": "Xp1", "yp1": "Yp1"}
    nds = nds.rename_dims(rename).rename_vars(rename)

    return nds, vds, mds


def ds_edge_difftx(_ds, iX, iY, iXp1, iYp1, face1, face2, _dim, moor, **kwargs):
    """different topology, axis=`x`"""
    _Nx = len(_ds.X) - 1

    dim_arg = {_dim: moor}
    iXn = iX.isel(**dim_arg)  #
    iYn = iY.isel(**dim_arg)  #

    uvars = kwargs["u"]
    vvars = kwargs["v"]
    gvars = kwargs["g"]
    cvars = kwargs["c"]

    args = {"xp1": slice(1)}
    rename = {"x": "xp1"}
    revar = "xp1"
    iXp1n = iXp1.isel(**dim_arg, **args)
    iYp1n = iYp1.isel(**dim_arg)
    iargs = {"X": _Nx - iYn, "Xp1": _Nx - iYp1n + 1, "Yp1": iXn - _Nx}

    dds = rotate_vars(_ds)[uvars + gvars]  # u and g variables
    vds = dds.isel(face=face2, **iargs)  # this is next face

    nvds = vds.rename_dims({"x": "xp1"}).rename_vars({"x": "xp1"})
    nvds = reset_dim(nvds, 1, revar)
    nvds = nvds.drop_vars(["Yp1", "X", "Xp1"])
    for var in nvds.reset_coords().data_vars:
        nvds[var].attrs = {}  # remove metadata for now

    argsn = {
        "face": face1,
        "X": iXn,
        "Y": iYn,
        "Xp1": iXp1n,
        "Yp1": iYp1n,
    }

    mds = _ds.isel(**argsn)  # regular eval
    mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])

    ugmds = _xr.combine_by_coords([mds[uvars + gvars], nvds])

    # get rest of u and center data
    cvmds = mds.reset_coords()[cvars + vvars]
    nds = _xr.combine_by_coords([cvmds, ugmds])
    co_list = [var for var in nds.data_vars if "time" not in nds[var].dims]
    nds = nds.set_coords(co_list)

    rename = {"x": "X", "xp1": "Xp1", "yp1": "Yp1", "y": "Y"}
    nds = nds.rename_dims(rename).rename_vars(rename)

    for var in nds.reset_coords().data_vars:
        nds[var].attrs = {}

    return nds, vds, mds


def ds_edge_diffty(_ds, iX, iY, _ix, xp1, iYp1, face1, face2, _dim, moor, **kwargs):
    """different topology, axis=`y`"""
    _Nx = len(_ds.X) - 1
    dim_arg = {_dim: moor}
    iXn = iX.isel(**dim_arg)  #
    iYn = iY.isel(**dim_arg)  #

    uvars = kwargs["u"]
    vvars = kwargs["v"]
    gvars = kwargs["g"]
    cvars = kwargs["c"]

    new_dim = DataArray(
        _np.arange(len(_ix)),
        dims=(_dim),
        attrs={"long_name": "index of " + _dim, "units": "none"},
    )

    # have to redefine iXp1 in decreasing order
    iXp1 = DataArray(
        _np.stack((_ix, _ix + 1)[::-1], 1),
        coords={_dim: new_dim, "xp1": xp1},
        dims=(_dim, "xp1"),
    )

    args = {"yp1": slice(1)}
    rename = {"y": "yp1"}
    iXp1n = iXp1.isel(**dim_arg)
    iYp1n = iYp1.isel(**dim_arg, **args)
    iargs = {"Xp1": iYn - _Nx, "Y": _Nx - iXn, "Yp1": _Nx - iXp1n + 1}
    dds = rotate_vars(_ds)[vvars + gvars]  # v and g variables

    # sample from the next face
    vds = dds.isel(face=face2, **iargs)
    nvds = vds.rename_dims({"y": "yp1"}).rename_vars({"y": "yp1"})
    nvds = reset_dim(nvds, 1, "yp1")
    nvds = nvds.drop_vars(["Xp1", "Y", "Yp1"])

    for var in nvds.reset_coords().data_vars:
        nvds[var].attrs = {}  # remove metadata for now

    # evaluate at edge of present face -- missing Yp1 data
    argsn = {
        "face": face1,
        "X": iXn,
        "Y": iYn,
        "Xp1": iXp1n,
        "Yp1": iYp1n,
    }
    mds = _ds.isel(**argsn)  # regular eval
    mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])  # always drop these

    # combine to create complete, edge data at v and g points
    vgmds = _xr.combine_by_coords([mds[vvars + gvars], nvds])

    # get rest of u and center data
    cumds = mds.reset_coords()[cvars + uvars]
    nds = _xr.combine_by_coords([cumds, vgmds])
    co_list = [var for var in nds.data_vars if "time" not in nds[var].dims]
    nds = nds.set_coords(co_list)

    rename = {"x": "X", "xp1": "Xp1", "yp1": "Yp1", "y": "Y"}
    nds = nds.rename_dims(rename).rename_vars(rename)

    for var in nds.reset_coords().data_vars:
        nds[var].attrs = {}

    return nds, vds, mds


def ds_edge(_ds, _ix, _iy, _ifaces, ii, _face_topo, _dim="mooring", **kwargs):
    """
    Given an array of index point that ends at the
    face boundary, it samplest from the neighbor faced data
    the corresponding vector value.

    Parameters:
    ----------

        _ds: xarray.dataset
            faced data.
        _ix, _iy: 1d array-like
            Integers. array of index positions for the present
            ith-face. It may end or beginning at the face edge.
        _ifaces: 1d array-like. integers
            full array of all faces sampled along the entire
            mooring trajectory.
        ii: int
            identifies the present face.
        _face_topo: dict
            dictionary with face connections - topology
        _Nx: int
            Last index along the x or y direction. Default=89
            associated with ECCO.

    Returns:
    --------

    """

    _Niter = len(_ifaces)
    rotS = set(_np.arange(7, 13))
    nrotS = set(_np.arange(6))
    _Nx = len(_ds.X) - 1

    if "pair" in kwargs.keys():
        pair = kwargs.pop("pair", None)
    else:
        pair = []

    _dim_name = _dim
    new_dim = DataArray(
        _np.arange(len(_ix)),
        dims=(_dim_name),
        attrs={"long_name": "index of " + _dim_name, "units": "none"},
    )
    y = DataArray(
        _np.arange(1),
        dims=("y"),
        attrs={"long_name": "j-index of cell center", "units": "none"},
    )
    x = DataArray(
        _np.arange(1),
        dims=("x"),
        attrs={"long_name": "i-index of cell center", "units": "none"},
    )
    yp1 = DataArray(
        _np.arange(2),
        dims=("yp1"),
        attrs={"long_name": "j-index of cell corner", "units": "none"},
    )
    xp1 = DataArray(
        _np.arange(2),
        dims=("xp1"),
        attrs={"long_name": "i-index of cell corner", "units": "none"},
    )

    # Transform indexes in DataArray
    iY = DataArray(
        _np.reshape(_iy, (len(new_dim), len(y))),
        coords={_dim_name: new_dim, "y": y},
        dims=(_dim_name, "y"),
    )
    iX = DataArray(
        _np.reshape(_ix, (len(new_dim), len(x))),
        coords={_dim_name: new_dim, "x": x},
        dims=(_dim_name, "x"),
    )

    iYp1 = DataArray(
        _np.stack((_iy, _iy + 1), 1),
        coords={_dim_name: new_dim, "yp1": yp1},
        dims=(_dim_name, "yp1"),
    )

    iXp1 = DataArray(
        _np.stack((_ix, _ix + 1), 1),
        coords={_dim_name: new_dim, "xp1": xp1},
        dims=(_dim_name, "xp1"),
    )

    Yval = iYp1.where(iYp1 == _Nx + 1, drop=True)
    ymoor = Yval[_dim_name]

    Xval = iXp1.where(iXp1 == _Nx + 1, drop=True)
    xmoor = Xval[_dim_name]

    connect = False
    axis = None
    moor = []
    moors = []  # local direction of crossing
    if len(ymoor) + len(xmoor) > 0:
        connect = True  # array crosses into other face
        for imoor in [xmoor, ymoor]:
            if imoor.size:
                moors.append(imoor)
        axes = ["x", "y"]  # for debugging purpose
        indm = [i for i, e in enumerate([len(xmoor), len(ymoor)]) if e != 0]
        if len(indm) == 1:  # crossing long a single direction
            indm = indm[0]
            axis = axes[indm]  # for debug purpose
            moor = [xmoor, ymoor][indm]
        else:  # crossing in two directions (both `x` and `y`)
            # can only take one direction at a time
            # see if the direction to considered is provided
            # as input/argument
            axis = kwargs.pop("axis", None)
            if axis is None:  # pragma: no cover
                # pick one => `x`
                indm = indm[0]
                axis = axes[indm]  # for debug purpose
            elif axis in ["x", "y"]:
                # pick the given
                indm = indm[["x", "y"].index(axis)]
            else:
                raise ValueError("axis given is not appropriate")
            moor = moors[indm]

    if ii < _Niter - 1:
        fdir = face_direction(_ifaces[ii], _ifaces[ii + 1], _face_topo)
        face1, face2 = _ifaces[ii : ii + 2]
        if fdir in [0, 2]:
            # the array begins at the edge with another face, and
            # array advances towards left in `x` or `y`.
            # Will neeed to sumplement at the boundary
            # need to infer what the face is -- it may not be defined
            # within `_ifaces`
            face2 = face_adjacent(
                [_ix[moor[0]]], [_iy[moor[0]]], _ifaces[ii], _face_topo, _Nx
            )[0]

    else:
        if connect:  # index = 0 is at far right
            face1, face2 = _ifaces[ii], _ifaces[ii - 1]
            fdir = face_direction(face2, face1, _face_topo)
        else:
            fdir = None

    zvars = [
        var
        for var in _ds.reset_coords().data_vars
        if len(_ds[var].dims) == 1 and var not in _ds.dims
    ]
    # 1D dataset : scalars that are depth dependent, or time dependent.
    ds1D = _ds[zvars]

    varlist = [var for var in _ds.reset_coords().data_vars]
    zcoords = ["Zl", "Zu", "Zp1"]
    tcoords = ["time_midp"]
    uvars = zcoords + tcoords  # u-points
    vvars = zcoords + tcoords  # v-points
    gvars = zcoords + tcoords  # corner points
    cvars = zcoords + tcoords
    for var in varlist:
        if set(["Xp1", "Y"]).issubset(_ds[var].dims):
            uvars.append(var)
        if set(["Xp1", "Yp1"]).issubset(_ds[var].dims):
            gvars.append(var)
        if set(["Yp1", "X"]).issubset(_ds[var].dims):
            vvars.append(var)
        if set(["Y", "X"]).issubset(_ds[var].dims):
            cvars.append(var)

    vkwargs = {"u": uvars, "v": vvars, "g": gvars, "c": cvars}

    if connect:
        if set([6]).issubset([face1, face2]):
            nds = ds_arcedge(_ds, _ix, _iy, moor, face1, face2, _dim)
            return nds, connect, moor, moors
        else:
            if set([face1, face2]).issubset(nrotS) or set([face1, face2]).issubset(
                rotS
            ):
                # same topology across faces
                if axis == "x":
                    nds, *a = ds_edge_sametx(
                        _ds, iX, iY, iXp1, iYp1, face1, face2, _dim, moor, **vkwargs
                    )
                if axis == "y":
                    nds, *a = ds_edge_samety(
                        _ds,
                        iX,
                        iY,
                        _ix,
                        xp1,
                        iXp1,
                        iYp1,
                        face1,
                        face2,
                        _dim,
                        moor,
                        **vkwargs,
                    )

            else:
                # there is a change in topology across faces
                if axis == "x":
                    nds, *a = ds_edge_difftx(
                        _ds, iX, iY, iXp1, iYp1, face1, face2, _dim, moor, **vkwargs
                    )

                if axis == "y":
                    nds, *a = ds_edge_diffty(
                        _ds, iX, iY, _ix, xp1, iYp1, face1, face2, _dim, moor, **vkwargs
                    )

            # correct topology of rotated face
            if face1 in rotS:
                nds = rotate_vars(mates(nds, pair=pair))
                rename_rdims1 = {"Xp1": "nYp1", "Yp1": "nXp1", "X": "nY", "Y": "nX"}
                rename_rdims2 = {"nXp1": "Xp1", "nYp1": "Yp1", "nX": "X", "nY": "Y"}
                nds = nds.rename_dims(rename_rdims1).rename_vars(rename_rdims1)
                nds = nds.rename_dims(rename_rdims2).rename_vars(rename_rdims2)

            # append vertical variables
            moor = moor.values
            nds = _xr.merge([nds, ds1D])
            # make sure to remove `face` as coord
            if "face" in nds.reset_coords().data_vars:
                nds = nds.drop_vars(["face"])
    else:
        nds = None
        moor = None
    return nds, connect, moor, moors


def ds_arcedge(_ds, _ix, _iy, moor, face1, face2, _dim="mooring"):
    """
    Given an array of index points that right ends at the edge between the arctic and
    another face, returns the complete set of center point and  corner/velocity points.

    Parameters:
    ----------
        _ds: xarray.dataset
            `face` is a dimension.
        _ix, _iy: 1d array-like
            Integers. Index positions for the present
            ith-face that evaluate at the right edge between faces.
            Either all `_ix` or all `_iy` are `len(_ds.X)-1`
        face1: Int
            present face. index evaluates at this face
        face2: Int
            adjacent face from which to sample.
        _dim: str
            name of dimension. either `mooring` or `station`
    Returns:
    --------
        xarray.DataSet.
    """
    _Nx = len(_ds.X) - 1
    dim_list = _ds.dims
    co_list = [
        var
        for var in _ds.variables
        if "time" not in _ds[var].dims and var not in dim_list
    ]

    #     # set up id
    # this repeats in 4 fns. Need to refactor here
    _dim_name = _dim
    new_dim = DataArray(
        _np.arange(len(_ix)),
        dims=(_dim_name),
        attrs={"long_name": "index of " + _dim_name, "units": "none"},
    )
    y = DataArray(
        _np.arange(1),
        dims=("y"),
        attrs={"long_name": "j-index of cell center", "units": "none"},
    )
    x = DataArray(
        _np.arange(1),
        dims=("x"),
        attrs={"long_name": "i-index of cell center", "units": "none"},
    )
    yp1 = DataArray(
        _np.arange(2),
        dims=("yp1"),
        attrs={"long_name": "j-index of cell corner", "units": "none"},
    )
    xp1 = DataArray(
        _np.arange(2),
        dims=("xp1"),
        attrs={"long_name": "i-index of cell corner", "units": "none"},
    )

    # Transform indexes in DataArray
    iY = DataArray(
        _np.reshape(_iy, (len(new_dim), len(y))),
        coords={_dim_name: new_dim, "y": y},
        dims=(_dim_name, "y"),
    )
    iX = DataArray(
        _np.reshape(_ix, (len(new_dim), len(x))),
        coords={_dim_name: new_dim, "x": x},
        dims=(_dim_name, "x"),
    )

    iYp1 = DataArray(
        _np.stack((_iy, _iy + 1), 1),
        coords={_dim_name: new_dim, "yp1": yp1},
        dims=(_dim_name, "yp1"),
    )

    iXp1 = DataArray(
        _np.stack((_ix, _ix + 1), 1),
        coords={_dim_name: new_dim, "xp1": xp1},
        dims=(_dim_name, "xp1"),
    )

    # prepare coords (C-grid)
    zvars = [
        var
        for var in _ds.reset_coords().data_vars
        if len(_ds[var].dims) == 1 and var not in _ds.dims
    ]
    # 1D dataset : scalars that are depth dependent, or time dependent.
    ds1D = _ds[zvars]

    varlist = [var for var in _ds.reset_coords().data_vars]
    zcoords = ["Zl", "Zu", "Zp1"]
    tcoords = ["time_midp"]
    uvars = zcoords + tcoords  # u-points
    vvars = zcoords + tcoords  # v-points
    gvars = zcoords + tcoords  # corner points
    cvars = zcoords + tcoords
    for var in varlist:
        if set(["Xp1", "Y"]).issubset(_ds[var].dims):
            uvars.append(var)
        if set(["Xp1", "Yp1"]).issubset(_ds[var].dims):
            gvars.append(var)
        if set(["Yp1", "X"]).issubset(_ds[var].dims):
            vvars.append(var)
        if set(["Y", "X"]).issubset(_ds[var].dims):
            cvars.append(var)

    dim_arg = {_dim_name: moor}
    iXn = iX.isel(**dim_arg)  #
    iYn = iY.isel(**dim_arg)  #

    if face1 == 6 and face2 == 7:
        args = {"xp1": slice(1)}  # used by both
        rename = {"x": "xp1"}
        revar = "xp1"
        iXp1n = iXp1.isel(**dim_arg, **args)
        iYp1n = iYp1.isel(**dim_arg)
        iargs = {"Y": iYn, "Yp1": iYp1n, "Xp1": iXp1n - _Nx}
        # sample from next face
        vds = _ds.isel(face=face2, **iargs)
        vds = vds.reset_coords()[uvars + gvars]
        # get the rest of the points
        argsn = {"face": face1, "X": iXn, "Y": iYn, "Xp1": iXp1n, "Yp1": iYp1n}
        mds = _ds.isel(**argsn).reset_coords()  # regular eval
        mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])
        mds = reset_dim(mds, 1, revar)
        ugmds = _xr.combine_by_coords([mds[uvars + gvars], vds])

        cvmds = mds.reset_coords()[cvars + vvars]
        nds = _xr.combine_by_coords([cvmds, ugmds])
        co_list = [var for var in nds.data_vars if "time" not in nds[var].dims]
        nds = nds.set_coords(co_list)
        rename = {"yp1": "Xp1", "xp1": "Yp1", "x": "Y", "y": "X"}
        nds = nds.rename_dims(rename).rename_vars(rename)
        nds = rotate_vars(nds)

        for _varName in nds.variables:
            if "mate" in nds[_varName].attrs:
                _dims = nds[_varName].dims
                if _varName not in co_list and "Yp1" in _dims:
                    nds[_varName] = -nds[_varName]
            if _varName == "CS":
                nds[_varName] = -nds[_varName]

    if face1 == 6 and face2 == 10:
        # have to redefine iXp1 in decreasing order
        iXp1 = DataArray(
            _np.stack((_ix, _ix + 1)[::-1], 1),
            coords={_dim_name: new_dim, "xp1": xp1},
            dims=(_dim_name, "xp1"),
        )
        args = {"yp1": slice(1)}
        rename = {"y": "yp1"}

        iXp1n = iXp1.isel(**dim_arg)
        iYp1n = iYp1.isel(**dim_arg, **args)
        iargs = {"Xp1": iYn - _Nx, "Y": _Nx - iXn, "Yp1": _Nx - iXp1n + 1}
        dds = rotate_vars(_ds.reset_coords())[vvars + gvars]  # v and g variables

        # sample from the next face
        vds = dds.isel(face=face2, **iargs)
        vds = vds.rename_dims({"y": "yp1"}).rename_vars({"y": "yp1"})
        vds = vds.drop_vars(["Xp1", "Y", "Yp1"])
        if "face" in vds.data_vars:  # pragma: no cover
            vds = vds.drop_vars(["face"])
        for var in vds.reset_coords().data_vars:
            vds[var].attrs = {}  # remove metadata for now

        argsn = {
            "face": face1,
            "X": iXn,
            "Y": iYn,
            "Xp1": iXp1n,
            "Yp1": iYp1n,
        }
        mds = _ds.isel(**argsn)  # regular eval
        mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])  # always drop these
        mds = reset_dim(mds.reset_coords(), 1, "yp1")
        if "face" in mds.data_vars:  # pragma: no cover
            mds = mds.drop_vars(["face"])

        vgmds = _xr.combine_by_coords([mds[vvars + gvars], vds])

        # get rest of u and center data
        cumds = mds.reset_coords()[cvars + uvars]
        nds = _xr.combine_by_coords([cumds, vgmds])
        nds = nds.set_coords(
            [var for var in nds.data_vars if "time" not in nds[var].dims]
        )

        rename = {"x": "X", "xp1": "Xp1", "yp1": "Yp1", "y": "Y"}
        nds = nds.rename_dims(rename).rename_vars(rename)

        for _varName in nds.variables:
            if "mate" in nds[_varName].attrs:
                _dims = nds[_varName].dims
                if _varName not in co_list and ("Yp1" in _dims or "Xp1" in _dims):
                    nds[_varName] = -nds[_varName]
            if _varName == "SN":
                nds[_varName] = -nds[_varName]
            if _varName == "CS":
                nds[_varName] = -nds[_varName]

    if face1 == 2 and face2 == 6:
        iXp1 = DataArray(
            _np.stack((_ix, _ix + 1), 1),
            coords={_dim_name: new_dim, "xp1": xp1},
            dims=(_dim_name, "xp1"),
        )
        args = {"yp1": slice(1)}
        rename = {"y": "yp1"}
        iXp1n = iXp1.isel(**dim_arg)
        iYp1n = iYp1.isel(**dim_arg, **args)
        iargs = {"Xp1": iYn - _Nx, "Y": _Nx - iXn, "Yp1": _Nx - iXp1n + 1}
        dds = rotate_vars(_ds)[vvars + gvars]  # v and g variables

        # sample from the next face
        vds = dds.isel(face=face2, **iargs)
        nvds = vds.rename_dims({"y": "yp1"}).rename_vars({"y": "yp1"})
        nvds = reset_dim(nvds, 1, "yp1")
        nvds = nvds.drop_vars(["Xp1", "Y", "Yp1"])
        for var in nvds.reset_coords().data_vars:
            nvds[var].attrs = {}  # remove metadata for now

        # evaluate at edge of present face -- missing Yp1 data
        argsn = {
            "face": face1,
            "X": iXn,
            "Y": iYn,
            "Xp1": iXp1n,
            "Yp1": iYp1n,
        }
        mds = _ds.isel(**argsn)  # regular eval
        mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])  # always drop these
        # combine to create complete, edge data at v and g points
        vgmds = _xr.combine_by_coords([mds[vvars + gvars], nvds])

        # get rest of u and center data
        cumds = mds.reset_coords()[cvars + uvars]
        nds = _xr.combine_by_coords([cumds, vgmds])
        nds = nds.set_coords(
            [var for var in nds.data_vars if "time" not in nds[var].dims]
        )
        rename = {"x": "X", "xp1": "Xp1", "yp1": "Yp1", "y": "Y"}
        nds = nds.rename_dims(rename).rename_vars(rename)

        for var in nds.reset_coords().data_vars:
            nds[var].attrs = {}

    if face1 == 5 and face2 == 6:
        args = {"yp1": slice(1)}
        rename = {"y": "yp1"}
        iXp1n = iXp1.isel(**dim_arg)
        iYp1n = iYp1.isel(**dim_arg, **args)
        iargs = {"X": iXn, "Xp1": iXp1n, "Yp1": iYp1n - _Nx}
        revar = "yp1"
        vds = _ds.isel(face=face2, **iargs)
        vds = vds.reset_coords()[vvars + gvars]
        # get the rest of the points
        argsn = {
            "face": face1,
            "X": iXn,
            "Y": iYn,
            "Xp1": iXp1n,
            "Yp1": iYp1n,
        }
        mds = _ds.isel(**argsn).reset_coords()  # regular eval
        mds = mds.drop_vars(["Yp1", "Xp1", "X", "Y"])
        vds = reset_dim(vds, 1, revar)
        vgmds = _xr.combine_by_coords([mds[vvars + gvars], vds])

        cumds = mds.reset_coords()[cvars + uvars]
        nds = _xr.combine_by_coords([cumds, vgmds])
        co_list = [var for var in nds.data_vars if "time" not in nds[var].dims]
        nds = nds.set_coords(co_list)
        rename = {"x": "X", "y": "Y", "xp1": "Xp1", "yp1": "Yp1"}
        nds = nds.rename_dims(rename).rename_vars(rename)

    nds = _xr.merge([nds, ds1D])

    return nds


def face_direction(face1, face2, face_connections):
    """
    from the topology `face_connections`, infers the direction
    of the array: `left (0)`, `right (1)`, `bottom (2)`, `top (3)`.
    """
    left, right = face_connections[face1]["X"]
    bot, top = face_connections[face1]["Y"]

    perimeter = []
    for edge in [left, right, bot, top]:
        if edge is not None:
            perimeter.append(edge[0])
        if edge is None:  # faces 0, 3, 9, 12
            perimeter.append(edge)

    if set([face2]).issubset(perimeter):
        return perimeter.index(face2)
    else:
        if face1 == face2:
            raise ValueError("faces {} and {} must be different.".format(face1, face2))
        else:
            raise ValueError("faces {} and {} are not contiguous.".format(face1, face2))


def splitter(_ix, _iy, _ifaces):
    """
    Takes the output from `connector(_ix, _iy)` as input, and splits it into
    the many faces the array grows through. The numner of faces is determine
    by the array `_ifaces`, or equal length as each element of `connector()`.
    Then `ifaces` has the same element, there is only one face (or simple
    topology), and the output is a list of len == 1.
    """

    # identify if and where there is a change in face
    ll = _np.where(abs(_np.diff(_ifaces)))[0]

    X0, Y0 = [], []
    for ii in range(len(ll) + 1):
        if ii == 0:
            x0, y0 = _ix[: ll[ii] + 1], _iy[: ll[ii] + 1]
        if ii > 0 and ii < len(ll):
            x0, y0 = (
                _ix[ll[ii - 1] + 1 : ll[ii] + 1],
                _iy[ll[ii - 1] + 1 : ll[ii] + 1],
            )
        if ii == len(ll):
            x0, y0 = _ix[ll[ii - 1] + 1 :], _iy[ll[ii - 1] + 1 :]
        X0.append(x0)
        Y0.append(y0)
    return X0, Y0


def edge_completer(_x, _y, face_dir=None, ind=-1, _N=89):
    """verifies that an array begins and ends at the edge of a face.

    Parameters:
    ----------

        _x, _y: list, list
            indexes of morring array in logical space.
        face_dir: int
            output of `od.llc_rearrange.face_direction. Indicates the direction
            towards which the left endpoint of (mooring) array must reach the
            edge of face.
        ind: int, 0 or -1 (default).
            indicates the index of the (mooring) array.
        _N: int, default=89.
            last index of each faceted dimension. (len(X)-1).
    """

    if face_dir == 1:  # towards local right in x (increase x-index)
        _mx, _my = connector([_x[ind], _N], [_y[ind], _y[ind]])
        if ind == -1:
            _x, _y = _np.append(_x, _mx), _np.append(_y, _my)
        if ind == 0:
            _x, _y = _np.append(_mx[::-1], _x), _np.append(_my, _y)
    if face_dir == 0:  # towards local left in x (increase x-index)
        _mx, _my = connector([0, _x[ind]], [_y[ind], _y[ind]])
        if ind == 0:
            _x, _y = _np.append(_mx, _x), _np.append(_my, _y)
        if ind == -1:
            _x, _y = _np.append(_x, _mx[::-1]), _np.append(_y, _my)
    if face_dir == 3:  # towards local right in y
        _mx, _my = connector([_x[ind], _x[ind]], [_y[ind], _N])
        if ind == -1:
            _x, _y = _np.append(_x, _mx), _np.append(_y, _my)
        if ind == 0:
            _x, _y = _np.append(_mx, _x), _np.append(_my[::-1], _y)
    if face_dir == 2:
        _mx, _my = connector([_x[ind], _x[ind]], [0, _y[ind]])
        if ind == 0:
            _x, _y = _np.append(_mx, _x), _np.append(_my, _y)
        if ind == -1:  # last entry
            _x, _y = _np.append(_x, _mx), _np.append(_y, _my[::-1])

    mask = _np.abs(_np.diff(_x)) + _np.abs(_np.diff(_y)) == 0
    _x, _y = (_np.delete(ii, _np.argwhere(mask)) for ii in (_x, _y))

    return _x, _y


def edge_slider(x1, y1, f1, x2, y2, f2, face_connections, _N=89):
    """
    Looks at the edge points between faces f1 (present)
    and f2 (next). Returns a point in f1 that is aligned
    with the first element in f2.

    Parameters:
    ----------
        [x1, y1, f1]: list of integers.
            Present face (`f1`) coordinates.
        [x2, y2, f2] : list of integers.
            Next face (`f2`) coordinates
        face_connections: dict.
            topology of grid.
        _N: int
            last index along `X` or `Y`

    Returns:
        newP: list
            It's elements are int values for present face `f1`
    """
    # cannot handle upper right corner (with 3 face data).
    crns = []
    for p in [[_N, _N]]:
        crns.append(p == [x1, y1])
    if crns.count(True):
        # TODO:  check if this is an actual problem
        raise ValueError("`[x1, y1]` can not be on a face corner")

    rotS = _np.arange(7, 13)
    nrotS = _np.arange(6)
    arc = _np.array([6])
    # it only matters is one case.
    fdir = face_direction(f1, f2, face_connections)

    # see if array left-ends (at 0) or right-ends (at len(ds.X)-1)
    set0 = set([x1, y1])
    ind0, ind1 = set([0]).issubset(set0), set([_N]).issubset(set0)
    # identify the local axis at which the array ends
    if ind0:
        i = (x1, y1).index(0)
    if ind1:
        i = (x1, y1).index(_N)

    if set([6, 7]) == set([f1, f2]):
        # match in y. No shift necessary
        new_P = [x2, y2]
        new_P[i] = [x1, y1][i]
    if set([6, 2]) == set([f1, f2]):
        if fdir == 3:
            new_P = [x2, _N - y2][::-1]
            new_P[i] = [x1, y1][i]
        else:
            new_P = [_N - x2, y2][::-1]
            new_P[i] = [x1, y1][i]
    if set([6, 5]) == set([f1, f2]):
        new_P = [x2, y2]
        new_P[i] = [x1, y1][i]
    if set([6, 10]) == set([f1, f2]):
        if fdir == 0:
            new_P = [_N - x2, y2][::-1]
            new_P[i] = [x1, y1][i]
        else:
            new_P = [x2, _N - y2][::-1]
            new_P[i] = [x1, y1][i]
    if arc[0] not in [f1, f2]:
        if set([f1, f2]).issubset(rotS) or set([f1, f2]).issubset(nrotS):
            new_P = [x2, y2]
            new_P[i] = [x1, y1][i]
        else:
            new_P = [a - b for a, b in zip(2 * [_N], [x2, y2][::-1])]
            new_P[i] = [x1, y1][i]
    return new_P


def fill_path(_X, _Y, _faces, k, _face_conxs, _N=89):
    """
    Given a sequence of index arrays (each within a face)
    makes sure that it always begins and ends at the face edge, expend
    the end faced-data which can either end or begin at the face edge. To be
    used when len(_faces)>1. Otherwise, use `connector`.

    Parameters:
    ----------
        X, Y: each 1d array-like of ints
            len(X) == len(Y) >= 1
        face: 1d array-like.
            len(face)==len(X)>1. Identifies which face the array is sampling
            from
        k: int
            identifies the kth-array pair (X, Y) with kth-face.
        _N: int
            Length of x- or y- dimension. Default is 89, associated with
    TODO:

    incorporate the changes above.
    """
    # import numpy as _np

    Ntot = len(_faces)
    x, y = connector(_X[k], _Y[k])

    # if Ntot > 1: # there is

    # ASSUMPTION:
    # Array normally increases monotonically with i and j at its end points.
    # Under such assumption, the 1st faceted array always is completed at its
    # right end point `index = -1`, but NOT at its left end point `index=0`.
    # To generalize, I need to include an option that allows the first array
    # (`k=0`) to be completed ONLY at the `index=0`, and the last faceted
    # array (`k=-1`) to be completed. The rest of the

    if k == 0:
        # Under assumption, this always happens with first Face. But it does
        # NOT happen with last face
        # k=-1.
        dir1 = face_direction(
            _faces[k], _faces[k + 1], _face_conxs
        )  # right end of array

        x, y = edge_completer(x, y, face_dir=dir1, ind=-1, _N=_N)
        x1, y1 = connector(_X[k + 1], _Y[k + 1])
        dir2 = face_direction(
            _faces[k + 1], _faces[k], _face_conxs
        )  # check direction to complete its `index=0`
        x1, y1 = edge_completer(
            x1, y1, face_dir=dir2, ind=0, _N=_N
        )  # include the indedx =0 and the face_direction.

        P = edge_slider(
            x[-1], y[-1], _faces[k], x1[0], y1[0], _faces[k + 1], _face_conxs, _N
        )

        x, y = connector(_np.append(x, P[0]), _np.append(y, P[1]))

    if k > 0 and k < Ntot - 1:
        # interior faces - aalways get completed to left and right.
        dir1 = face_direction(
            _faces[k], _faces[k + 1], _face_conxs
        )  # right end of array

        x, y = edge_completer(x, y, face_dir=dir1, ind=-1, _N=_N)

        # if not first of array, also complete towards beginning of array
        dir0 = face_direction(
            _faces[k], _faces[k - 1], _face_conxs
        )  # check direction to complete `index=0`
        # with previous face.
        x, y = edge_completer(x, y, face_dir=dir0, ind=0, _N=_N)

        # check next face, and how it intersect face edge to its left.

        x1, y1 = connector(_X[k + 1], _Y[k + 1])
        dir2 = face_direction(
            _faces[k + 1], _faces[k], _face_conxs
        )  # check direction to complete its `index=0`
        x1, y1 = edge_completer(
            x1, y1, face_dir=dir2, ind=0, _N=_N
        )  # include the indedx =0 and the face_direction.

        P = edge_slider(
            x[-1], y[-1], _faces[k], x1[0], y1[0], _faces[k + 1], _face_conxs, _N
        )

        x, y = connector(_np.append(x, P[0]), _np.append(y, P[1]))

    if k == Ntot - 1:
        # last faceted array. Under present assumption, need to
        # complete the `index=0` but NOT `index=-1`.
        dir0 = face_direction(
            _faces[k], _faces[k - 1], _face_conxs
        )  # check direction to complete `index=0`
        # with previous face.
        x, y = edge_completer(x, y, face_dir=dir0, ind=0, _N=_N)

    return x, y


def face_adjacent(_ix, _iy, _iface, _face_connections, _N=89):
    """
    Given a collection of data points within a face, returns the adjacent
    face next to boundary data. If data does not eval at the
    boundary between two faces, returns -1.

    Parameters:
    ----------
        _ix: 1d array-like, int data
        _iy: 1d array-like. int data
        _iface: int
            face index value where array lives.
        _face_connections: dict
            contains topology of data.
        _N: int. default=89 (ECCO)
            last index along  i or j index in faceted data.
    """
    adj_faces = []
    fleft, fright = _face_connections[_iface]["X"]
    fbot, ftop = _face_connections[_iface]["Y"]

    for i in range(len(_ix)):
        loc_data = -1  # initialize -- implies do edge data

        set0 = set([_ix[i], _iy[i]])
        ind0, ind1 = set([0]).issubset(set0), set([_N]).issubset(set0)

        if ([_ix[i], _iy[i]]).count(_N) > 1:
            raise ValueError(
                "OceanSpy cannot subsample data from upper right corner of a face"
            )

        if ind0:
            k = ([_ix[i], _iy[i]]).index(0)
            if k > 0:
                loc_data = 2
            else:
                loc_data = 0
        if ind1:
            k = ([_ix[i], _iy[i]]).index(_N)
            if k > 0:
                loc_data = 3
            else:
                loc_data = 1

        if loc_data == 0:
            adj_faces.append(fleft[0])
        if loc_data == 1:
            if fright is not None:
                adj_faces.append(fright[0])
            else:
                adj_faces.append(-1)  # singularity south pole.
        if loc_data == 2:
            if fbot is not None:
                adj_faces.append(fbot[0])
            else:
                adj_faces.append(-1)  # singularity south pole.
        if loc_data == 3:
            adj_faces.append(ftop[0])
        if loc_data == -1:
            adj_faces.append(loc_data)
    return adj_faces


def edgesid(_iX, _iY, _N=89):
    """
    From an array of isolated logical indexes within a face, extracts the ones
    that lie at the edge between one or more faces. It also removes repeated
    entries from input array. This function does not preserve the ordering of
    the data.
    """
    unique = set(tuple([_iX[i], _iY[i]]) for i in range(len(_iX)))
    _iX = _np.array([list(unit)[0] for unit in unique])
    _iY = _np.array([list(unit)[1] for unit in unique])

    # identify all x-edges, if any
    ixe0 = _np.where(_iX == 0)[0]
    ixe1 = _np.where(_iX == _N)[0]
    # y-edges, if any
    iye0 = _np.where(_iY == 0)[0]
    iye1 = _np.where(_iY == _N)[0]
    _index = list(ixe0) + list(ixe1) + list(iye0) + list(iye1)

    return _iX, _iY, _index


def index_splitter(ix, iy, _N):
    """
    Takes the index pair (ix, iy) of ordered, continuous and equidistant
    (unit) distanced array, and identifies the location at which the pair
    reaches the edge of the face and reenters the same face (no crossing).
    The edge of the face is identified by `_N`, the last index along each
    dimention. If array only reches edge of face at end points, then returns
    empty list. This allows to split the array while preserving its original
    order.
    """
    nI = []  # indexes of partition
    Nx = len(ix)

    iix = _np.where(ix == _N)[0]
    iiy = _np.where(iy == _N)[0]

    if iix.shape[0] + iiy.shape[0] > 0:
        Ii = []  # there is right-edged data.
        if iix.shape[0] > 0:
            xb = _np.where(_np.diff(iix) > 1)[0]
        else:
            xb = []
        if iiy.shape[0] > 0:
            yb = _np.where(_np.diff(iiy) > 1)[0]
        else:
            yb = []

        if len(yb) == 0 and len(iiy) > 0:  # only one set of edge data
            Ii.append(list(iiy))
        if len(yb) > 0:
            for k in range(len(yb) + 1):
                if k == 0:
                    Ii.append(list(iiy[: yb[k] + 1]))
                if k > 0 and k < len(yb):
                    Ii.append(list(iiy[yb[k - 1] + 1 : yb[k] + 1]))
                if k == len(yb):
                    Ii.append(list(iiy[yb[k - 1] + 1 :]))
        if len(xb) == 0 and len(iix) > 0:
            Ii.append(list(iix))
        if len(xb) > 0:
            for k in range(len(xb) + 1):
                if k == 0:
                    Ii.append(list(iix[: xb[k] + 1]))
                if k > 0 and k < len(xb):
                    Ii.append(list(iix[xb[k - 1] + 1 : xb[k] + 1]))
                if k == len(xb):
                    Ii.append(list(iix[xb[k - 1] + 1 :]))

        if len(xb) + len(yb) > 0:
            i0 = [Ii[k][0] for k in range(len(Ii))]
            ii = _np.argsort(i0)  # order

            for k in range(len(ii)):
                endpoint = set([Nx - 1]).issubset(Ii[ii[k]])
                origin = set([0]).issubset(Ii[ii[k]])
                if not endpoint and not origin:
                    nI.append(Ii[ii[k]])
    return nI


def order_from_indexing(_ix, _in):
    """Given an array of bounded integers (indexing array), and a list of
    indexes that subsets the indexing array, returns the mapping associated
    with the subsetting array.
    """
    Nn = len(_in)
    nx = []
    if Nn == 0:  # preserves original data
        mI = []  # there is no mapping/no data at edge of face
        nx = _ix
    if Nn > 0:  # there is edge data.
        nx = [_ix[: _in[0][0]]]
        for jj in range(1, Nn):
            nx.append(_ix[_in[jj - 1][0] : _in[jj - 1][-1] + 1])
            nx.append(_ix[_in[jj - 1][-1] + 1 : _in[jj][0]])
        nx.append(_ix[_in[-1][0] : _in[-1][-1] + 1])
        nx.append(_ix[_in[-1][-1] + 1 :])

        mI = [[k for k in range(len(nx[0]))]]
        for ii in range(1, len(nx)):
            val = mI[ii - 1][-1] + 1
            mI.append(list([val + k for k in range(len(nx[ii]))]))
    return mI, nx


def ds_splitarray(
    _ds, _iXn, _iYn, _faces, _iface, _nI, _face_connections, _dim_name="mooring"
):
    """
    Creates a dataset from an array that reaches the edges of the face/tile
    once or multiple times, without crossing into a different face, but can
    end or begin at the edge of the face (which is to be interpreted more
    generally as crossing from or into a different face)
    """

    # construct entire index mapper that reconstructs iXn from broken array (nI)
    _ni, _ = order_from_indexing(_iXn, _nI)
    if len(_faces) == 1:
        # single face with multiple edge connections
        fdir = 0
    else:
        if _iface < len(_faces) - 1:
            fdir = face_direction(_faces[_iface], _faces[_iface + 1], _face_connections)
        else:  # last face index=0 at far right.
            fdir = face_direction(_faces[_iface - 1], _faces[_iface], _face_connections)

    # construct a list of adjacent faces where array does not end.
    adj_faces = []
    for ii in range(len(_nI)):
        # sample single point.
        nx, ny, face = _iXn[_nI[ii][:1]], _iYn[_nI[ii][:1]], _faces[_iface]
        afaces = face_adjacent(nx, ny, face, _face_connections)
        adj_faces.append(afaces)

    j = 0  # counter for face eval.
    eds = []  # each item will be a dataset
    for i in range(len(_ni)):  # parallelize this. It could take some time.
        if i % 2 == 0:
            if i in [0, len(_ni) - 1]:
                nds, connect, moor, *a = ds_edge(
                    _ds, _iXn[_ni[i]], _iYn[_ni[i]], _faces, _iface, _face_connections
                )
                if connect:  # subarry end at right edge
                    if len(moor) == 0 or fdir in [0, 2]:
                        # array ends at 0 index
                        nnx, nny = (
                            _iXn[_ni[i]][moor[-1] + 1 :],
                            _iYn[_ni[i]][moor[-1] + 1 :],
                        )
                        ds0 = eval_dataset(
                            _ds, nnx, nny, _iface=_faces[_iface], _dim_name=_dim_name
                        )
                        shift = len(nds.mooring)
                        ds0 = reset_dim(ds0, shift, _dim_name)
                        nds = _xr.combine_by_coords([nds, ds0])
                    else:
                        nnx, nny = _iXn[_ni[i]][: moor[0]], _iYn[_ni[i]][: moor[0]]
                        ds0 = eval_dataset(
                            _ds, nnx, nny, _iface=_faces[_iface], _dim_name=_dim_name
                        )
                        nds = _xr.combine_by_coords([ds0, nds])
                    del ds0
                else:  # safe to eval everywhere
                    nnx, nny = _iXn[_ni[i]], _iYn[_ni[i]]
                    nds = eval_dataset(
                        _ds, nnx, nny, _iface=_faces[_iface], _dim_name=_dim_name
                    )
            else:
                nnx, nny = _iXn[_ni[i]], _iYn[_ni[i]]
                nds = eval_dataset(
                    _ds, nnx, nny, _iface=_faces[_iface], _dim_name=_dim_name
                )
        else:
            nds, *a = ds_edge(
                _ds,
                _iXn[_ni[i]],
                _iYn[_ni[i]],
                [_faces[_iface]] + adj_faces[j],
                0,
                _face_connections,
            )
            j += 1  # update the count for eval at a face edge
        if i > 0:
            shift = int(eds[i - 1].mooring.values[-1]) + 1
            nds = reset_dim(nds, shift)
        eds.append(nds)
    dsf = _xr.combine_by_coords(eds).chunk({_dim_name: len(_iXn)})
    del eds
    return dsf


def fdir_completer(_ix, _iy, _faces, _iface, _Nx, _face_connections):
    """completes the next directional face , whether the face is defined
    within the path of the array, or if the next face eval is an adjacent
    face. This matters when the arrays ends of beginnins at an endpoint
    """
    if _iface < len(_faces) - 1:
        fdir = face_direction(_faces[_iface], _faces[_iface + 1], _face_connections)
    else:  # last face array (fdir may not matter)
        _idx, _idy, _index = edgesid(_ix, _iy, _Nx)
        if len(_index) == 0:
            # no edge data
            fdir = None
        else:
            # infer the direction from face topology.
            aface = face_adjacent(_ix, _iy, _faces[_iface], _face_connections)
            fdir = face_direction(_faces[_iface], aface[0], _face_connections)
    return fdir


def mooring_singleface(_ds, _ix, _iy, _faces, _iface, _face_connections):
    """
    evaluates the mooring array within a single face.
    """
    _Nx = len(_ds.X) - 1
    _ixn, _iyn = connector(_ix, _iy)
    nI = index_splitter(_ixn, _iyn, _Nx)
    if len(nI) > 0:
        # data reaches edge of face, but reenters. Need to split intp
        # subarrays in order to eval corner/vel points at adjacent
        # face which is NOT necessarily next in ordered sequence array.
        args = {
            "_ds": _ds,
            "_iXn": _ixn,
            "_iYn": _iyn,
            "_nI": nI,
            "_faces": _faces,
            "_iface": _iface,
            "_face_connections": _face_connections,
            "_dim_name": "mooring",
        }
        dsf = ds_splitarray(**args)
    else:
        # no need to split into subarrays
        iix = _np.where(_ixn == _Nx)[0]
        iiy = _np.where(_iyn == _Nx)[0]

        if iix.size + iiy.size == 0:
            # array does not end at right edge
            dsf = eval_dataset(_ds, _ixn, _iyn, _faces[_iface], _dim_name="mooring")
        else:
            # there is at least one right-edge point
            # must split into subarray (edge+interior)
            DSt = []
            nds, connect, moor, moors, *a = ds_edge(
                _ds, _ixn, _iyn, _faces, _iface, _face_connections
            )

            # check twice-appearing right ends (same axis or different)
            if len(moors) == 1:
                # array reaches right edge once along either `x` or `y`,
                # need to check it arrays begins and ends same index eval
                diffm = _np.diff(nds.mooring)
                jump = _np.argwhere(abs(diffm) != 1)
                if jump.size:  # two right-end points
                    moor = jump
                    ds2 = nds.isel(
                        mooring=slice(int(moor[-1]) + 1, len(nds.mooring) + 1)
                    )
                    moor2 = ds2.mooring.values[0]
                    nds = nds.isel(mooring=slice(int(moor[-1]) + 1))  # 1st end point
                    DSt.append(ds2)
            elif len(moors) == 2:
                # array ends/begins at a right edge at different axes.
                jump = _np.array([])  # no repeated ends
                # `ds_edge` can only extract edged-data from a single axis
                # at a time. need to evaluate again.
                if (moors[0].values == moor).all():
                    # the `x` axis was picked in previous eval.
                    kwargs = {"axis": "y"}  # need y-axis endpoint
                    _ind = moors[1].mooring.values[0]
                else:
                    kwargs = {"axis": "x"}
                # select adjacent face to index `_ind`.
                new_face = face_adjacent(
                    [_ixn[_ind]], [_iyn[_ind]], _faces[_iface], _face_connections
                )
                present_face = 0
                dst, *a = ds_edge(
                    _ds,
                    _ixn,
                    _iyn,
                    [_faces[_iface]] + new_face,
                    present_face,
                    _face_connections,
                    **kwargs,
                )
                # from the two edge evals - I need to order them
                moor_a, moor_b = moor, dst.mooring.values
                ds_a, ds_b = nds, dst
                ind1 = [moor_a.min(), moor_b.min()].index(
                    min([moor_a.min(), moor_b.min()])
                )
                ind2 = [moor_a.min(), moor_b.min()].index(
                    max([moor_a.min(), moor_b.min()])
                )
                moor = [moor_a, moor_b][ind1]
                moor2 = [moor_a, moor_b][ind2][0]  # only care ab out first element
                # can now order the datasets with edge data
                nds = [ds_a, ds_b][ind1]  # mooring near zero
                ds2 = [ds_a, ds_b][ind2]  # mooring near end
                DSt.append(ds2)
            DSt.append(nds)

            # get interior points
            if connect:
                shift = None
                _eval = True
                if jump.size or len(moors) > 1:
                    # interior inbetween two right edge points
                    nnx, nny = (
                        _ixn[int(moor[-1]) + 1 : moor2],
                        _iyn[int(moor[-1]) + 1 : moor2],
                    )
                    shift = len(nds.mooring)  # shift from 0
                elif len(_ixn) == len(moor):
                    # No interior point
                    DS0, _eval = [], False
                else:
                    if 0 in moor:
                        # edge of array towards beginning of array
                        nnx, nny = _ixn[int(moor[-1]) + 1 :], _iyn[int(moor[-1]) + 1 :]
                        shift = len(nds.mooring)
                    else:
                        # edge of array towards end - interior before
                        nnx, nny = _ixn[: int(moor[0])], _iyn[: int(moor[0])]
                if _eval:
                    ds0 = eval_dataset(
                        _ds, nnx, nny, _iface=_faces[_iface], _dim_name="mooring"
                    )
                    if shift is not None:
                        ds0 = reset_dim(ds0, shift, "mooring")
                    DS0 = [ds0]
                DSt = DSt + DS0
                dsf = _xr.combine_by_coords(DSt)
                del nds, DS0, DSt

    if "face" in dsf.reset_coords().data_vars:
        dsf = dsf.drop_vars(["face"])
    return dsf, _ixn, _iyn


def station_singleface(_ds, _ix, _iy, _faces, _iface, _face_connections):
    """Extracts isolated station values from dataset from the given horizontal
    index values (`iface`, '_iy', '_ix'). These are not ordered as the original
    coords.
    """
    shift = True
    iX, iY, ind = edgesid(_ix, _iy)
    # get edge data
    eX, eY = iX[ind], iY[ind]
    # remove from original array
    iX, iY = _np.delete(iX, ind), _np.delete(iY, ind)
    # data with index=0 somewhere is safe to eval at current face.
    # find such data and restore it to original index array
    aface = face_adjacent(eX, eY, _faces[_iface], _face_connections)
    directions = _np.array(
        [
            face_direction(_faces[_iface], aface[i], _face_connections)
            for i in range(len(aface))
        ]
    )
    dirs = []
    if set([0]).issubset(directions):  # these can safely be eval at face
        dirs += list([item[0] for item in _np.argwhere(directions == 0)])
    if set([2]).issubset(directions):
        dirs += list([item[0] for item in _np.argwhere(directions == 2)])
    neX, neY = (
        eX[dirs],
        eY[dirs],
    )
    eX, eY, aface = (
        _np.delete(eX, dirs),
        _np.delete(eY, dirs),
        _np.delete(_np.array(aface), dirs),
    )
    iX, iY = _np.append(iX, neX), _np.append(iY, neY)
    dsf = eval_dataset(_ds, iX, iY, _faces[_iface], _dim_name="station")
    if iX.size == 0:
        shift = None
    if eX.shape[0] > 0:
        # evaluate at edge of between faces. need adjascent face and appropriate indexes
        DSe = []
        ii = 0
        for adjface in set(aface):
            aind = _np.where(aface == adjface)[0]
            dse, *a = ds_edge(
                _ds,
                eX[aind],
                eY[aind],
                [_faces[_iface]] + [adjface],
                0,
                _face_connections,
                _dim="station",
            )
            if ii > 0:
                shift = int(DSe[ii - 1]["station"].values[-1]) + 1
                dse = reset_dim(dse, shift, dim="station")
            DSe.append(dse)
            ii += 1
        dse = _xr.combine_by_coords(DSe)
        del DSe
        if shift is None:
            # no interior point. only edge data
            dsf = dse
        else:
            shift = int(dsf["station"].values[-1]) + 1
            dse = reset_dim(dse, shift, dim="station")
            dsf = _xr.combine_by_coords([dsf, dse])

    return dsf


def cross_face_diffs(_ds, _ix, _iy, _faces, _iface, _face_connections):
    """computes the unit distance between the location of index spaces in
    both directions diffX and diffY when data has complex topology.

    Parameters:
    ----------
        _ds: xarray.Dataset
            contains `face`, and original dataset.
        _ix, _iy: 1d-array like.
            elements are int values. Output from `connector()`, whch
        _faces: list
            contains the face indexes of the ordered track. len()>=1
        _iface: int
            index of current face.
        _face_connection: dict
            face topology.
    returns
        diffX: 1D array-like
        diffY: 1D array-like
    """

    # exclude the arctic for now
    Rot = _np.arange(7, 13)

    # define all 4 unit vectors defining crossing from face i to face i+1
    # inherits face topo (ordering), and assumes that of non rot faces.
    # format: [[xvec, yvec], [.], ...] and inherets logics from `face_directions`
    fdirs_options = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    diffX, diffY = _np.diff(_ix), _np.diff(_iy)

    if _faces[_iface] in Rot:  # correct for topology
        # redefine options with corrected topology
        # keep same result from `face_direction`
        fdirs_options = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        # create copy and rotate
        ndiffX, ndiffY = _copy.deepcopy(diffX), _copy.deepcopy(diffY)
        diffX = ndiffY
        diffY = -ndiffX

    if _faces[_iface] == 6:
        # arctic array
        fdirs_options = [[0, -1], [0, -1], [0, -1], [0, -1]]
        diffX, diffY, *a = arct_diffs(_ds, _ix, _iy)

    # get direction between the edge point of present face
    # only when there is another face.
    if _iface < len(_faces) - 1:
        # local face direction for next face
        fdir = face_direction(_faces[_iface], _faces[_iface + 1], _face_connections)
        # when crossing into other face - get logical dir vectors
        tdiffx, tdiffy = fdirs_options[fdir]
        diffX, diffY = _np.append(diffX, tdiffx), _np.append(diffY, tdiffy)
    else:
        tdiffx, tdiffy = _np.array([]), _np.array([])

    return diffX, diffY, _np.array([tdiffx]), _np.array([tdiffy])


def arct_diffs(_ds, _Xind, _Yind):
    _Nx = len(_ds.X) - 1

    # define triangular areas that split the arctic
    XR5 = Polygon([(0, -1), (0, 0), (_Nx / 2, _Nx / 2), (_Nx, 0), (_Nx, -1)])
    XR7 = Polygon(
        [(_Nx + 1, 0), (_Nx, 0), (_Nx / 2, _Nx / 2), (_Nx, _Nx), (_Nx + 1, _Nx)]
    )
    XR10 = Polygon(
        [(0, _Nx + 1), (0, _Nx), (_Nx / 2, _Nx / 2), (_Nx, _Nx), (_Nx, _Nx + 1)]
    )
    XR2 = Polygon([(-1, _Nx), (0, _Nx), (_Nx / 2, _Nx / 2), (0, 0), (-1, 0)])

    # define a small polygon that contains the theoretical line
    # dividing the areas above

    lower_left = Polygon(
        [
            (0, 3),
            ((_Nx + 1) // 2 - 3, (_Nx + 1) // 2),
            ((_Nx + 1) // 2, (_Nx + 1) // 2),
            ((_Nx + 1) // 2, (_Nx + 1) // 2 - 3),
            (3, 0),
            (0, 0),
        ]
    )
    lower_right = Polygon(
        [
            ((_Nx + 1) // 2, (_Nx + 1) // 2),
            ((_Nx + 1) // 2, (_Nx + 1) // 2 - 3),
            (_Nx - 3, 0),
            (_Nx, 0),
            (_Nx, 3),
            ((_Nx + 1) // 2 + 3, (_Nx + 1) // 2),
            ((_Nx + 1) // 2, (_Nx + 1) // 2),
        ]
    )
    upper_right = Polygon(
        [
            ((_Nx + 1) // 2, (_Nx + 1) // 2),
            ((_Nx + 1) // 2 + 3, (_Nx + 1) // 2),
            (_Nx, _Nx - 3),
            (_Nx, _Nx),
            (_Nx - 3, _Nx),
            ((_Nx + 1) // 2, (_Nx + 1) // 2 + 3),
        ]
    )
    upper_left = Polygon(
        [
            (0, _Nx),
            (0, _Nx - 3),
            ((_Nx + 1) // 2 - 3, (_Nx + 1) // 2),
            ((_Nx + 1) // 2, (_Nx + 1) // 2),
            ((_Nx + 1) // 2, (_Nx + 1) // 2 + 3),
            (3, _Nx),
        ]
    )

    def _mask_array(iX, iY, polygon):  # pragma no cover
        mask = []
        for i in range(len(iX)):
            point = Point(iX[i], iY[i])
            mask.append(polygon.contains(point))
        return _np.array(mask).nonzero()[0]

    # define masks
    maskR5 = _mask_array(_Xind, _Yind, XR5)
    maskR7 = _mask_array(_Xind, _Yind, XR7)
    maskR10 = _mask_array(_Xind, _Yind, XR10)
    maskR2 = _mask_array(_Xind, _Yind, XR2)

    # applu masks and consolidate points
    niXR2 = _Xind[maskR2]
    niYR2 = _Yind[maskR2]
    setR2 = set(tuple((niXR2[i], niYR2[i])) for i in range(len(niYR2)))

    niXR5 = _Xind[maskR5]
    niYR5 = _Yind[maskR5]
    setR5 = set(tuple((niXR5[i], niYR5[i])) for i in range(len(niYR5)))

    niXR7 = _Xind[maskR7]
    niYR7 = _Yind[maskR7]
    setR7 = set(tuple((niXR7[i], niYR7[i])) for i in range(len(niYR7)))

    niXR10 = _Xind[maskR10]
    niYR10 = _Yind[maskR10]
    setR10 = set(tuple((niXR10[i], niYR10[i])) for i in range(len(niYR10)))

    # assert I am capturing all data points
    captured_set = setR2.union(setR5).union(setR7).union(setR10)

    ndiffX, ndiffY = [], []
    for i in range(len(_Xind) - 1):
        dX = _Xind[i + 1] - _Xind[i]
        dY = _Yind[i + 1] - _Yind[i]
        pair1 = set(tuple((_Xind[j], _Yind[j])) for j in range(i, i + 1))
        pair2 = set(tuple((_Xind[j + 1], _Yind[j + 1])) for j in range(i, i + 1))
        set_pair = pair1.union(pair2)
        if set_pair.issubset(setR5):
            # same topology
            ndiffX.append(dX)
            ndiffY.append(dY)
        elif set_pair.issubset(setR10):
            # same topo - oposite ordering in both directions.
            ndiffX.append(-dX)
            ndiffY.append(-dY)
        elif set_pair.issubset(setR7):
            ndiffX.append(dY)
            ndiffY.append(-dX)
        elif set_pair.issubset(setR2):
            ndiffX.append(-dY)
            ndiffY.append(dX)
        else:
            # edge data - need add place holder value
            # for accurate indexing reference
            ndiffX.append(None)
            ndiffY.append(None)

    # identify the missing index values
    miss = []
    for i in range(len(_Xind)):
        pair = set(tuple((_Xind[j], _Yind[j])) for j in range(i, i + 1))
        if not pair.issubset(captured_set):
            miss.append(i)

    for i in miss:
        # forward from c
        if i < len(_Xind) - 1:
            dXf = _Xind[i + 1] - _Xind[i]
            dYf = _Yind[i + 1] - _Yind[i]
            pairf = set(tuple((_Xind[j + 1], _Yind[j + 1])) for j in range(i, i + 1))
        else:
            pairf = set(tuple((None, None)) for j in range(i, i + 1))
        # behind from c
        if i > 0:
            dXb = _Xind[i] - _Xind[i - 1]
            dYb = _Yind[i] - _Yind[i - 1]
            pairb = set(tuple((_Xind[i - 1], _Yind[i - 1])) for j in range(i, i + 1))
        else:
            pairb = set(tuple((None, None)) for j in range(i, i + 1))

        pointc = Point(_Xind[i], _Yind[i])

        # tried nested if statements, but to incorporate
        # end points (non-cyclic) this worked best

        if lower_left.contains(pointc) and pairb.issubset(setR2):
            ndiffX[i - 1], ndiffY[i - 1] = -dYb, dXb
        if lower_left.contains(pointc) and pairb.issubset(setR5):
            ndiffX[i - 1], ndiffY[i - 1] = dXb, dYb
        if lower_left.contains(pointc) and pairf.issubset(setR5):
            ndiffX[i], ndiffY[i] = dXf, dYf
        if lower_left.contains(pointc) and pairf.issubset(setR2):
            ndiffX[i], ndiffY[i] = -dYf, dXf
        if lower_right.contains(pointc) and pairb.issubset(setR5):
            ndiffX[i - 1], ndiffY[i - 1] = dXb, dYb
        if lower_right.contains(pointc) and pairb.issubset(setR7):
            ndiffX[i - 1], ndiffY[i - 1] = dYb, -dXb
        if lower_right.contains(pointc) and pairf.issubset(setR7):
            ndiffX[i], ndiffY[i] = dYf, -dXf
        if lower_right.contains(pointc) and pairf.issubset(setR5):
            ndiffX[i], ndiffY[i] = dXf, dYf
        if upper_right.contains(pointc) and pairb.issubset(setR7):
            ndiffX[i - 1], ndiffY[i - 1] = dYb, -dXb
        if upper_right.contains(pointc) and pairb.issubset(setR10):
            ndiffX[i - 1], ndiffY[i - 1] = -dXb, -dYb
        if upper_right.contains(pointc) and pairf.issubset(setR10):
            ndiffX[i], ndiffY[i] = -dXf, -dYf
        if upper_right.contains(pointc) and pairf.issubset(setR7):
            ndiffX[i], ndiffY[i] = dYf, -dXf
        if upper_left.contains(pointc) and pairb.issubset(setR10):
            ndiffX[i - 1], ndiffY[i - 1] = -dXb, -dYb
        if upper_left.contains(pointc) and pairb.issubset(setR2):
            ndiffX[i - 1], ndiffY[i - 1] = -dYb, dXb
        if upper_left.contains(pointc) and pairf.issubset(setR2):
            ndiffX[i], ndiffY[i] = -dYf, dXf
        if upper_left.contains(pointc) and pairf.issubset(setR10):
            ndiffX[i], ndiffY[i] = -dXf, -dYf
    return _np.array(ndiffX), _np.array(ndiffY), captured_set, miss


class Dims:  # pragma: no cover
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
