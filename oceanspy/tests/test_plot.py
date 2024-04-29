# TODO: add tests for aliased datasets.

# Import modules
# Matplotlib (keep it below oceanspy)
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

# From OceanSpy
from oceanspy import OceanDataset, open_oceandataset
from oceanspy.plot import TS_diagram, horizontal_section, time_series, vertical_section

# Directory
Datadir = "./oceanspy/tests/Data/"

# Test oceandataset
od = open_oceandataset.from_netcdf("{}MITgcm_rect_nc.nc" "".format(Datadir))

ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
ECCOod = open_oceandataset.from_catalog("LLC", ECCO_url)
ECCOod._ds = ECCOod._ds.rename_vars(
    {"hFacS": "HFacS", "hFacW": "HFacW", "hFacC": "HFacC"}
)
co_list = [var for var in ECCOod._ds.data_vars if "time" not in ECCOod._ds[var].dims]
ECCOod._ds = ECCOod._ds.set_coords(co_list)
if "timestep" in ECCOod._ds.data_vars:
    ECCOod._ds = ECCOod._ds.drop_vars(["timestep"])

# Create mooring, sruvey, and particles
Xmoor = [od.dataset["XC"].min().values, od.dataset["XC"].max().values]
Ymoor = [od.dataset["YC"].min().values, od.dataset["YC"].max().values]
od_moor = od.subsample.mooring_array(Xmoor=Xmoor, Ymoor=Ymoor)

Xsurv = [
    od.dataset["XC"].min().values,
    od.dataset["XC"].mean().values,
    od.dataset["XC"].max().values,
]
Ysurv = [
    od.dataset["YC"].min().values,
    od.dataset["YC"].mean().values,
    od.dataset["YC"].max().values,
]
od_surv = od.subsample.survey_stations(Xsurv=Xsurv, Ysurv=Ysurv)

times = od.dataset["time"]
n_parts = 10
Ypart = np.empty((len(times), n_parts))
Xpart = np.empty((len(times), n_parts))
Zpart = np.zeros((len(times), n_parts))
for p in range(n_parts):
    Ypart[:, p] = np.random.choice(od.dataset["Y"], len(times))
    Xpart[:, p] = np.random.choice(od.dataset["X"], len(times))
# Extract particles
# Warning due to time_midp
with pytest.warns(UserWarning):
    od_part = od.subsample.particle_properties(
        times=times, Ypart=Ypart, Xpart=Xpart, Zpart=Zpart
    )


# ==========
# TS diagram
# ==========
@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize(
    "Tlim, Slim, dens",
    [
        (None, [1], None),
        ([1], None, None),
        (None, None, xr.DataArray(np.random.randn(2, 3))),
    ],
)
def test_TS_error(od_in, Tlim, Slim, dens):
    with pytest.raises(ValueError):
        TS_diagram(od_in, Tlim=Tlim, Slim=Slim, dens=dens)


# Test settings
@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize("Tlim", [[0, 1]])
@pytest.mark.parametrize("Slim", [[0, 1]])
@pytest.mark.parametrize("ax", [True])
@pytest.mark.parametrize("cutout_kwargs", [True])
@pytest.mark.parametrize("cmap_kwargs", [{"robust": True}])
@pytest.mark.parametrize("contour_kwargs", [{"levels": 10}])
@pytest.mark.parametrize("clabel_kwargs", [{"fontsize": 10}])
@pytest.mark.parametrize(
    "dens",
    [
        xr.DataArray(
            np.random.randn(2, 3),
            coords={"Temp": np.arange(2), "S": np.arange(3)},
            dims=("Temp", "S"),
        )
    ],
)
@pytest.mark.parametrize("plotFreez", [False])
def test_TS_diagram_set(
    od_in,
    Tlim,
    Slim,
    ax,
    cutout_kwargs,
    cmap_kwargs,
    contour_kwargs,
    clabel_kwargs,
    dens,
    plotFreez,
):
    plt.close()
    if cutout_kwargs is True:
        cutout_kwargs = {
            "XRange": [
                od_in.dataset["XC"].min().values,
                od_in.dataset["XC"].max().values,
            ]
        }
    if ax is True:
        _, ax = plt.subplots(1, 1)

    ax = TS_diagram(
        od_in,
        Tlim=Tlim,
        Slim=Slim,
        ax=ax,
        cutout_kwargs=cutout_kwargs,
        cmap_kwargs=cmap_kwargs,
        contour_kwargs=contour_kwargs,
        clabel_kwargs=clabel_kwargs,
        dens=dens,
        plotFreez=plotFreez,
    )

    if Tlim is not None:
        assert ax.get_ylim() == tuple(Tlim)
    if Slim is not None:
        assert ax.get_xlim() == tuple(Slim)


# Test fields
@pytest.mark.parametrize("od_in", [od, od_moor, od_surv, od_part])
@pytest.mark.parametrize("meanAxes", [None, "time"])
@pytest.mark.parametrize("colorName", [None, "Temp", "Depth", "Eta", "U", "W"])
def test_TS_diagram_field(od_in, meanAxes, colorName):
    plt.close()
    contour_kwargs = {"levels": 10, "cmap": "viridis"}
    ax = TS_diagram(
        od_in, colorName=colorName, meanAxes=meanAxes, contour_kwargs=contour_kwargs
    )
    assert isinstance(ax, plt.Axes)


# ===========
# Time series
# ===========
@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize(
    "varName, meanAxes, intAxes",
    [
        ("Depth", True, False),
        ("Temp", None, False),
        ("Temp", "X", False),
        ("Temp", False, "X"),
        ("Temp", True, True),
        ("Temp", "time", False),
        ("Temp", False, "time"),
    ],
)
def test_timeSeries_error(od_in, varName, meanAxes, intAxes):
    if meanAxes is None:
        with pytest.raises(TypeError):
            plt.close()
            time_series(od_in, varName=varName, meanAxes=meanAxes, intAxes=intAxes)
    else:
        with pytest.raises(ValueError):
            plt.close()
            time_series(od_in, varName=varName, meanAxes=meanAxes, intAxes=intAxes)


@pytest.mark.parametrize("od_in", [od, od_moor, od_surv, od_part])
def test_timeSeries(od_in):
    plt.close()
    cutout_kwargs = {
        "timeRange": [od_in.dataset["time"][0].values, od_in.dataset["time"][-1].values]
    }
    ax = time_series(od_in, varName="Temp", intAxes=True, cutout_kwargs=cutout_kwargs)
    assert isinstance(ax, plt.Axes)


# ==================
# Horizontal section
# ==================
@pytest.mark.parametrize(
    "od_in, plotType, meanAxes",
    [(od_moor, "contourf", True), (od, "error", True), (od, "contourf", False)],
)
def test_hor_sec_error(od_in, plotType, meanAxes):
    plt.close()
    with pytest.raises(ValueError):
        horizontal_section(od_in, varName="Temp", meanAxes=meanAxes, plotType=plotType)


@pytest.mark.parametrize("od_in", [od])
def test_hor_sec_warn(od_in):
    with pytest.warns(UserWarning):
        plt.close()
        fig, ax = plt.subplots(1, 1)
        ax = horizontal_section(
            od_in.set_projection("NorthPolarStereo"),
            varName="Depth",
            contourName="Depth",
            ax=ax,
            use_coords=False,
        )
        assert isinstance(ax, plt.Axes)

    with pytest.warns(UserWarning):
        plt.close()
        ax = horizontal_section(
            od_in,
            varName="Eta",
            contourName="Depth",
            subplot_kws={"projection": None},
        )
        assert isinstance(ax, xr.plot.FacetGrid)

    with pytest.warns(UserWarning):
        plt.close()
        ax = horizontal_section(
            od_in,
            varName="Eta",
            contourName="V",
            meanAxes=["Z"],
            subplot_kws={"projection": od.projection},
        )
        assert isinstance(ax, xr.plot.FacetGrid)


@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize("varName", ["Temp", "U", "V", "momVort3"])
@pytest.mark.parametrize("contourName", ["Depth", "U", "V", "momVort3"])
@pytest.mark.parametrize("step", [None, 2])
def test_hor_sec(od_in, varName, contourName, step):
    plt.close()
    cutout_kwargs = {
        "timeRange": [od_in.dataset["time"][0].values, od_in.dataset["time"][-1].values]
    }
    contour_kwargs = {"levels": 10, "cmap": "viridis"}
    clabel_kwargs = {"fontsize": 10}
    if varName == "Temp":
        od_in = od_in.set_projection("NorthPolarStereo")
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": od_in.projection})
    ax = horizontal_section(
        od_in,
        varName=varName,
        meanAxes=True,
        ax=ax,
        contourName=contourName,
        contour_kwargs=contour_kwargs,
        clabel_kwargs=clabel_kwargs,
        cutout_kwargs=cutout_kwargs,
        xstep=step,
        ystep=step,
    )
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize("col", [None, "time"])
def test_hor_sec_facet(od_in, col):
    plt.close()
    ax = horizontal_section(od_in.set_projection(None), varName="Eta", col=col)
    assert isinstance(ax, xr.plot.FacetGrid)


# ==================
# Horizontal section
# ==================
@pytest.mark.parametrize("od_in", [od_moor, od_surv])
def test_ver_sec_error(od_in):
    od_in = od_in.compute.weighted_mean(varNameList="Temp")
    with pytest.raises(ValueError):
        vertical_section(od_in, varName="Temp", contourName="w_mean_Temp")


@pytest.mark.parametrize("od_in", [od])
@pytest.mark.parametrize(
    "subsampMethod", ["error", None, "mooring_array", "survey_stations"]
)
def test_ver_sec_subsamp(od_in, subsampMethod):
    cutout_kwargs = {
        "timeRange": [od_in.dataset["time"][0].values, od_in.dataset["time"][-1].values]
    }
    contour_kwargs = {"levels": 10, "cmap": "viridis"}
    plt.close()
    fig, ax = plt.subplots(1, 1)
    if subsampMethod in ["error", None]:
        with pytest.raises(ValueError):
            vertical_section(
                od_in,
                varName="Temp",
                meanAxes=True,
                cutout_kwargs=cutout_kwargs,
                subsampMethod=subsampMethod,
            )
    else:
        if subsampMethod == "mooring_array":
            Xmoor = [od_in.dataset["XC"].min().values, od_in.dataset["XC"].max().values]
            Ymoor = [od_in.dataset["YC"].min().values, od_in.dataset["YC"].max().values]
            subsamp_kwargs = {"Xmoor": Xmoor, "Ymoor": Ymoor}
            use_dist = False
        else:
            Xsurv = [
                od_in.dataset["XC"].min().values,
                od_in.dataset["XC"].mean().values,
                od_in.dataset["XC"].max().values,
            ]
            Ysurv = [
                od_in.dataset["YC"].min().values,
                od_in.dataset["YC"].mean().values,
                od_in.dataset["YC"].max().values,
            ]
            subsamp_kwargs = {"Xsurv": Xsurv, "Ysurv": Ysurv}
            use_dist = True
        ax = vertical_section(
            od_in,
            varName="Temp",
            ax=ax,
            meanAxes=True,
            cutout_kwargs=cutout_kwargs,
            subsampMethod=subsampMethod,
            subsamp_kwargs=subsamp_kwargs,
            use_dist=use_dist,
            contour_kwargs=contour_kwargs,
        )
        assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize("od_in", [od_moor, od_surv])
@pytest.mark.parametrize("varName", ["Temp", "U", "V", "W", "momVort3"])
@pytest.mark.parametrize("contourName", ["Temp", "U", "V", "W", "momVort3"])
@pytest.mark.parametrize("step", [None, 2])
def test_ver_sec(od_in, varName, contourName, step):
    plt.close()
    if "mooring_dist" in od_in.dataset.variables:
        ds = od_in.dataset.drop_vars("mooring_dist")
        od_in = OceanDataset(ds)
        contour_kwargs = {"levels": 10}
        clabel_kwargs = {"fontsize": 10}
        meanAxes = True
        intAxes = False
    else:
        contour_kwargs = None
        clabel_kwargs = None
        meanAxes = False
        intAxes = True

    ax = vertical_section(
        od_in,
        varName=varName,
        contourName=contourName,
        meanAxes=meanAxes,
        intAxes=intAxes,
        contour_kwargs=contour_kwargs,
        clabel_kwargs=clabel_kwargs,
        step=step,
    )
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize("od_in", [od_moor, od_surv])
@pytest.mark.parametrize("contourName", ["Temp", "w_mean_Temp", "Depth"])
def test_ver_facet(od_in, contourName):
    od_in = od_in.compute.weighted_mean(varNameList="Temp", axesList="time")

    plt.close()
    if contourName == "Depth":
        with pytest.raises(ValueError):
            ax = vertical_section(od_in, varName="Temp", contourName=contourName)
    else:
        ax = vertical_section(od_in, varName="Temp", contourName=contourName)
        assert isinstance(ax, xr.plot.FacetGrid)


# New face plotting
# single face
X0, Y0 = np.array([-80, -42]), np.array([12, 50])
# two faces (oppo topo)
X1, Y1 = np.array([-50, 1]), np.array([20, 60])
# > two faces (arctic)
Y90W = np.arange(-60, 77.5, 15)
X90W = np.array([-90] * len(Y90W))

# face2axis for all faces
Face2Axis = {
    0: (3, 3, 0),
    1: (4, 2, 0),
    2: (5, 1, 0),
    3: (6, 0, 2),
    4: (7, 1, 1),
    5: (8, 2, 1),
    6: (9, 3, 1),
    7: (10, 1, 2),
    8: (11, 2, 2),
    9: (12, 3, 2),
    10: (None, 0, 0),
    11: (None, 0, 1),
    12: (None, 0, 3),
    13: (0, 3, 3),
    14: (1, 2, 3),
    15: (2, 1, 3),
    16: (None, 0, 3),
}


@pytest.mark.parametrize("od_in", [ECCOod])
@pytest.mark.parametrize("Rectangular", [True, False])
@pytest.mark.parametrize("Moorings", [(X0, Y0), (X1, Y1), (X90W, Y90W)])
@pytest.mark.parametrize("face2axis", [None, Face2Axis])
def test_faces(od_in, Rectangular, Moorings, face2axis):
    if Rectangular:
        R = None
    else:
        R = 6371.0
    od_in.parameters["rSphere"] = R
    Xm, Ym = Moorings
    plt.close()
    axes_faces = od_in.plot.faces_array(Xmoor=Xm, Ymoor=Ym, face2axis=face2axis)
    if isinstance(axes_faces, np.ndarray):
        if face2axis is None:
            if axes_faces.shape != (4, 4):
                for i in range(len(axes_faces)):
                    assert isinstance(axes_faces[i], plt.Axes)
            else:  # all faces
                for i in range(max(Face2Axis.keys())):
                    if Face2Axis[i][0] is not None:
                        iy, ix = Face2Axis[i][1], Face2Axis[i][2]
                        assert isinstance(axes_faces[iy, ix], plt.Axes)

        else:
            for i in range(max(face2axis.keys())):
                if face2axis[i][0] is not None:
                    iy, ix = face2axis[i][1], face2axis[i][2]
                    assert isinstance(axes_faces[iy, ix], plt.Axes)
    else:
        assert isinstance(axes_faces, plt.Axes)


def test_shortcuts():
    plt.close()
    ax = od.plot.TS_diagram()
    assert isinstance(ax, plt.Axes)

    plt.close()
    ax = od.plot.time_series(varName="Temp", meanAxes=True)
    assert isinstance(ax, plt.Axes)

    plt.close()
    ax = od.plot.horizontal_section(varName="Depth", add_labels=False)
    assert isinstance(ax, plt.Axes)

    plt.close()
    ax = od_surv.plot.vertical_section(
        varName="Temp", contourName="Temp", meanAxes=True
    )
    assert isinstance(ax, plt.Axes)
