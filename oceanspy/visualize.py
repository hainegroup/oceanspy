"""
Visualize: plot dataset
"""

import xarray as _xr
import numpy as _np
import xgcm as _xgcm

def interactive(dsORda,
                info,
                hvplot_kwargs = {}):
    """
    GUI for plotting data interactively using hvPlot.
    Plot maps using Orthographic projection, 
    and recognize vertical sections/profiles

    Parameters
    ----------
    dsORda: xarray.Dataset or xarray.DataArray
    info: oceanspy.open_dataset._info
    hvplot_kwargs: dict
        Keyword arguments for hvPlot: https://hvplot.pyviz.org/user_guide/index.html

    Returns
    -------
    GUI
    """

    import cartopy.crs as _ccrs
    import copy as _copy

    # Import modules
    from ipywidgets import interact
    import hvplot.xarray
    
    # Define global variables to pass to plot_da
    global ds, hv_kwargs 
    if isinstance(dsORda,_xr.DataArray): ds = dsORda.to_dataset()
    elif isinstance(dsORda,_xr.Dataset): ds = dsORda
    hv_kwargs = hvplot_kwargs
    
    # Choose variable interactively
    def plot_da(varname):
        # Make a copy of kwargs
        this_hv_kwargs = _copy.deepcopy(hv_kwargs)
        
        # Extract datarray and print info
        da   = ds[varname]
        for attr in da.attrs: print(attr,':',da.attrs[attr])
        print()
        for dim in da.dims:
            print('Dimension :',dim)
            for attr in da[dim].attrs: print('     ',attr,':',da[dim].attrs[attr])
        
        # Find axis
        axis = {}
        for dim in da.dims:
            if 'axis' in da[dim].attrs:
                axis[da[dim].attrs['axis']] = dim

        # MAP
        if 'X' in axis and 'Y' in axis:
            # Default 
            if 'x' not in this_hv_kwargs: this_hv_kwargs['x'] = axis['X']
            if 'y' not in this_hv_kwargs: this_hv_kwargs['y'] = axis['Y']
            if 'crs' not in this_hv_kwargs:
                crs = _ccrs.PlateCarree()
                this_hv_kwargs['crs'] = crs
            if 'projection' not in this_hv_kwargs: 
                lon = ds[axis['X']].values
                lat = ds[axis['Y']].values
                proj = _ccrs.Orthographic(central_longitude = lon.mean(),
                                          central_latitude  = lat.mean())
                this_hv_kwargs['projection'] = proj
            if 'groupby' not in this_hv_kwargs: 
                this_hv_kwargs['groupby'] = [dim for dim in da.dims if dim not in [axis['X'], axis['Y']]]
            if 'global_extent' not in this_hv_kwargs: this_hv_kwargs['global_extent'] = False
            if 'rasterize' not in this_hv_kwargs: this_hv_kwargs['rasterize'] = True
            if 'cmap' not in this_hv_kwargs: this_hv_kwargs['cmap'] = 'viridis'
            if 'kind' not in this_hv_kwargs: this_hv_kwargs['kind'] = 'quadmesh'
     
        # VERTICAL PROFILE   
        elif 'Z' in axis and len(da.dims)==1:
            # Default
            if 'invert' not in this_hv_kwargs: this_hv_kwargs['invert'] = True
            if 'kind'   not in this_hv_kwargs: this_hv_kwargs['kind'] = 'line'  
        
        # VERTICAL SECTION
        elif ('dist_VS' in info.var_names) and (info.var_names['dist_VS'] in da.dims):
            # 1D
            if len(da.dims)==1:
                # Default
                if 'kind'   not in this_hv_kwargs: this_hv_kwargs['kind'] = 'line'
                if 'groupby' not in this_hv_kwargs: 
                    this_hv_kwargs['groupby'] = []
                    
            # 2D
            if 'Z' in axis and info.var_names['dist_VS'] in da.dims:
                # Default
                if 'x' not in this_hv_kwargs: this_hv_kwargs['x'] = info.var_names['dist_VS']
                if 'y' not in this_hv_kwargs: this_hv_kwargs['y'] = axis['Z']
                if 'groupby' not in this_hv_kwargs: 
                    this_hv_kwargs['groupby'] = [dim for dim in da.dims if dim not in [axis['Z'], info.var_names['dist_VS']]]
                if 'cmap' not in this_hv_kwargs: this_hv_kwargs['cmap'] = 'viridis'
                if 'kind' not in this_hv_kwargs: this_hv_kwargs['kind'] = 'quadmesh'
            
        return da.hvplot(**this_hv_kwargs)
        
    return interact(plot_da,varname=[var for var in ds.variables if 
                                     (var not in ds.dims and var not in ds.coords)])

def TS_diagram(ds_user,info_user,ax=None
               ,time_user=None):
    
    # import modules and libraries
    import oceanspy as _ospy
    import matplotlib.pyplot as _plt
    
    # Get the required variables for the T-S diagram
    temp=ds_user[info_user.var_names["Temp"]]
    salinity=ds_user[info_user.var_names["S"]]
    
    # Check time string
    #if time_user==None:
    #    time_user=str(ds_user.coords['time'][0].values)
    #if isinstance(time_user,str)==False:
     #   raise ValueError('Time argument needs to a string')
    

    # Compute the density grid
    salinity_max=_np.nanmax(salinity.isel(time=0))+_np.nanstd(salinity.isel(time=0))
    temp_max=_np.nanmax(temp.isel(time=0))+_np.nanstd(temp.isel(time=0))
    salinity_min=_np.nanmin(salinity.isel(time=0))-_np.nanstd(salinity.isel(time=0))
    temp_min=_np.nanmin(temp.isel(time=0))-_np.nanstd(temp.isel(time=0))

    # provide option in kwargs for grid spacing, else take default of 10

    grid_spacing_sal=(salinity_max-salinity_min)*0.1
    grid_spacing_temp=(temp_max-temp_min)*0.1

    sal_grid_vals=_np.linspace(salinity_min-grid_spacing_sal,salinity_max+grid_spacing_sal,10)
    temp_grid_vals=_np.linspace(temp_min-grid_spacing_temp,temp_max+grid_spacing_temp,10)

    sal_grid_vals_mesh,temp_grid_vals_mesh=_np.meshgrid(sal_grid_vals,temp_grid_vals)

    # Creating xarray dataset from salinity and temperature meshes to pass ospy.compute for calculating density
    ds_sigma=_xr.Dataset({'Temp' : (['x','y'],temp_grid_vals_mesh), 'S' : (['x','y'],sal_grid_vals_mesh)},
                   coords={'X': (['x', 'y'], sal_grid_vals_mesh),'Y': (['x', 'y'], temp_grid_vals_mesh)})
    print('Plotting...')

    # Change info_sigma based on user input
    info_sigma=info_user    

    # Calling ospy.compute for density computation
    density_contour,info_contour=_ospy.compute.Sigma0(ds_sigma, info_sigma)


    # Plotting the T_S scatter
    if ax==None:
        fig,ax=_plt.subplots()
    CS = ax.contour(sal_grid_vals,temp_grid_vals,density_contour['Sigma0'],linestyles='dashed',colors='grey')
    colors=_np.linspace(_np.nanmin(salinity.coords['Z'].values),_np.nanmax(salinity.coords['Z'].values),
                                                                        len(_np.ravel(temp.isel(time=time_user))))

    ax.clabel(CS, inline=1, fontsize=10,colors='k',fmt='%1.2f')
    sc=_plt.scatter(_np.ravel(salinity.isel(time=time_user)), _np.ravel(temp.isel(time=time_user)),s=1,c=colors,cmap='RdYlBu')
    _plt.xlabel(salinity.attrs['long_name'])
    _plt.ylabel(temp.attrs['long_name']+' in '+temp.attrs['units'])
    _plt.colorbar(sc,extend='both')
    _plt.figtext(0.8,0.1,'(m)')

    return ax

def _create_animation_mpl(plot_func, ds, info, time_idx_min_max):
    
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display

    time_idx_range = range(time_idx_min_max[0], time_idx_min_max[1] + 1)
    n_frames = len(time_idx_range)
        
    fig, ax = plt.subplots()

    def init():
        pass

    def animate(i):
        fig.clear()
        # This creates axis for fig
        ax = fig.subplots()
        plot_func(ds, info, ax, time_idx_range[i])

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=200, blit=False)
 

    display(HTML(anim.to_html5_video()))

    return anim

def interactive_animate(ds, info, plot_func):
    
    """
    GUI for creating animation using matplotlib
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    plot_func: function
        user provided function that generates matplotlib plots
        the plot_func is called with ds, info, matplotlib figure
        and time-index as its arguments
    Returns
    -------
    GUI
    """

    import ipywidgets as ipyw
    from IPython.display import display

    last_time_step = len(ds['time']) - 1
    
    
    def get_params(time_idx_min_max):
        _create_animation_mpl(plot_func, ds, info, time_idx_min_max)

    # Creating widget objects
    int_slider = ipyw.widgets.IntRangeSlider(
            value=[0, last_time_step],
            min=0,
            max=last_time_step,
            step=1,
            description='Time Range:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d')
    begin_time_label = ipyw.widgets.Label(value=str(ds['time'].isel(time=int_slider.min).values))
    end_time_label = ipyw.widgets.Label(value=str(ds['time'].isel(time=int_slider.max).values))
    
    # Adding callback for slider(observe:does the linking btw two widgets)
    def update_labels(change):
        newRange = change['new']
        begin_time_label.value = str(ds['time'].isel(time=newRange[0]).values)
        end_time_label.value = str(ds['time'].isel(time=newRange[1]).values)

    int_slider.observe(update_labels, names='value')
        
    # Show interactive parts (slider)
    ipyw.interact_manual(
        get_params,
        time_idx_min_max=int_slider
    )
    
    # Show labels
    display(begin_time_label, end_time_label)


def animate_batch(ds, info, plot_func, time_idx_min, time_idx_max):

    """
    Function to generate animation given a plot function,
    without any GUI interaction.
    Parameters
    ----------
    ds: xarray.Dataset
    info: oceanspy.open_dataset._info
    plot_func: function
        user provided function that generates matplotlib plots
        the plot_func is called with ds, info, matplotlib figure
        and time-index as its arguments
    time_idx_min: int
        minimum index of time in ds to create an animation from
    time_idx_max: int
        maximum index of time in ds to create an animation from
    Returns
    -------
    anim: matplotlib.Animation
    """
    
    time_idx_min_max = [time_idx_min, time_idx_max]
    anim = _create_animation_mpl(plot_func, ds, info, time_idx_min_max)
    
    return anim