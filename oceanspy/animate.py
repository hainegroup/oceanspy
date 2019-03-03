from . import plot as _plot

def _create_animation_mpl(od, time, plot_func, func_kwargs, **kwargs):

    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display
    from IPython.utils import io
    import tqdm
    
    def animate(i):
        plt.cla()
        func_kwargs['cutout_kwargs'] = {'timeRange': time.isel(time=i).values, 'dropAxes': 'time'}
        with io.capture_output() as captured:
            plot_func(od, **func_kwargs)
        pbar.update(1)
        
        
        

    # call the animator. blit=True means only re-draw the parts that have changed.
    pbar = tqdm.tqdm(total=len(time))
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(time))


    display(HTML(anim.to_html5_video()))

    return anim


def TS_diagram(od, FuncAnimation_kwargs = None, **kwargs):
    
    plot_func = eval('_plot.TS_diagram')
    
    # First, get time
    cutout_kwargs = kwargs.pop('cutout_kwargs', None)
    if cutout_kwargs is not None: od = od.cutout(**cutout_kwargs)
    time = od._ds['time']
    
    # Fixed axis/cmap
    Tlim = kwargs.pop('Tlim', None)
    if Tlim is None:
        Tlim = [od._ds['Temp'].isel(time=0).min().values, od._ds['Temp'].isel(time=0).max().values]
    Slim = kwargs.pop('Slim', None)
    if Slim is None:
        Slim = [od._ds['S'].isel(time=0).min().values, od._ds['S'].isel(time=0).max().values]
    kwargs['Tlim'] = Tlim
    kwargs['Slim'] = Slim
    
    
    
    if FuncAnimation_kwargs is None: FuncAnimation_kwargs = {}
    # Cutout here
    anim = _create_animation_mpl(od, time, plot_func, kwargs, **FuncAnimation_kwargs)
    
    return anim
    