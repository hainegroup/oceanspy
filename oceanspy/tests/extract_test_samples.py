import oceanspy as ospy

# MITgcm rectilinear grid stored in NetCDF format
filename = 'MITgcm_rect_nc.nc'
od = ospy.open_oceandataset.EGshelfIIseas2km_ASR(cropped=True)
cut_od = od.subsample.cutout(XRange=[-18, -17.5], 
                             YRange=[70.25, 70.5], 
                             ZRange=[0, -120], 
                             timeRange=['2007-09-01', '2007-09-05'])
cut_od.to_netcdf('Data/'+filename)

# MITgcm curvilinear grid stored in NetCDF format
filename = 'MITgcm_curv_nc.nc'
od = ospy.open_oceandataset.exp_Arctic_Control()
cut_od = od.subsample.cutout(XRange=[-18, -10],
                             YRange=[70, 71],
                             ZRange=[0, -40],
                             timeRange=['2028-02-08', '2029-02-08'])
cut_od.to_netcdf('Data/'+filename)

# MITgcm rectilinear grid stored in binary format
filename = 'MITgcm_rect_bin.nc'
od = ospy.open_oceandataset.EGshelfSJsec500m(Hydrostatic=False)
cut_od = od.subsample.cutout(XRange=[-34.6, -34.5],
                             YRange=[64.1, 64.15],
                             ZRange=[0, -110],
                             timeRange=['2003-06-01', '2003-06-02T21'])
cut_od = ospy.OceanDataset(cut_od.dataset.isel(Xp1=slice(1,None), Yp1=slice(1,None)))
cut_od.to_netcdf('Data/'+filename)