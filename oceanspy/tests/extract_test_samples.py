import oceanspy as ospy

# MITgcm rectilinear grid stored in NetCDF format
filename = 'MITgcm_rect_nc.nc'
od = ospy.open_oceandataset.EGshelfIIseas2km_ASR(cropped=True)
cut_od = od.subsample.cutout(XRange=[-18, -17.5], 
                             YRange=[70.25, 70.5], 
                             ZRange=[0, -120], 
                             timeRange=['2007-09-01', '2007-09-05'])
cut_od.to_netcdf('Data/'+filename)

