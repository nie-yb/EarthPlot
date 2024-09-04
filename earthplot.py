# %%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
import matplotlib.figure
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeat
from pyproj import proj
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()

tickpad = 3  # 2.5
plt.rcParams['xtick.major.pad'] = tickpad
plt.rcParams['ytick.major.pad'] = tickpad


#%%
def subplots(nrows=1, ncols=1, proj=None, proj_kw=None,
             figsize=None, regular_grid=True,
             wspace=None, hspace=None,
             wratios=None, hratios=None):

    if proj_kw is None:
        proj_kw = {}

    if figsize is None:
        figsize = (ncols * 2., nrows * 2.)
    elif type(figsize) is int:
        figsize = (figsize, figsize)

    proj_dict = {
        'cyl': ccrs.PlateCarree(**proj_kw),
        'robin': ccrs.Robinson(**proj_kw),
        'npstere': ccrs.NorthPolarStereo(**proj_kw),  # central_longitude=90
    }

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig,
                  wspace=wspace, hspace=hspace,
                  width_ratios=wratios, height_ratios=hratios,
                  )

    if regular_grid:
        # normal grids
        if proj is None:
            projs = [None, ] * nrows * ncols
        elif type(proj) is str:
            projs = [proj_dict[proj], ] * nrows * ncols
        elif type(proj) is list or type(proj) is tuple:
            assert len(proj) == nrows * ncols
            projs = [proj_dict[p] if isinstance(p, str) else p for p in proj]
        else:
            projs = [proj, ] * nrows * ncols

        axs = [[None for _ in range(ncols)] for _ in range(nrows)]
        axs = np.array(axs)
        for i in range(nrows):
            for j in range(ncols):
                axs[i][j] = fig.add_subplot(gs[i, j],
                                            projection=projs[i*ncols+j],)

        if nrows == 1 or ncols==1:
            axs = axs.flatten()
        return fig, axs

    else:
        return fig, gs


def formats(axs, labels=True, geo=False, order='C',
            abc=None, abcloc='left', titlesize='large',
            boundinglat=None,
            coast=True, land=False, river=False, lake=False, reso='low',
            coastwidth=1.0, coastcolor='silver',
            ltitle=None, rtitle=None, title=None,
            toplabels=None, toplabels_kw=None,
            latlim=None, lonlim=None, latloc=None, lonloc=None,
            xlim=None, ylim=None,
            xticks=None, yticks=None, xticklabels=None, yticklabels=None,
            xlabel=None, ylabel=None, labelsize='large',
            xminor=None, yminor=None, xtick_params=None, ytick_params=None,
            grid=True, gridline_kw=None,
            land_kw=None, river_kw=None, lake_kw=None,
            sharex=False, sharey=False,
            **kwargs):

    # title_kw = {key: kwargs[key] for key in ('ltitle', 'rtitle', 'title')
    #             if key in kwargs}
    geotick_kw = {
        'latlim':latlim, 'lonlim':lonlim, 'latloc':latloc, 'lonloc':lonloc,
        'xticks':xticks, 'yticks':yticks, 'labelsize':labelsize,
        'boundinglat':boundinglat,
        'coast':coast, 'coastcolor':coastcolor, 'coastwidth':coastwidth,
        'land':land, 'river':river, 'lake':lake, 'reso': reso,
        # 'landcolor':landcolor, 'rivercolor':watercolor, 'lakecolor':watercolor,
        'labels':labels, 'grid':grid, 'gridline_kw':gridline_kw,
        'land_kw':land_kw, 'river_kw':river_kw, 'lake_kw':lake_kw,
        'sharex':sharex, 'sharey':sharey,
    }

    xytick_kw = {
        'xticks':xticks, 'yticks':yticks, 'xminor':xminor, 'yminor':yminor,
        'xticklabels':xticklabels, 'yticklabels':yticklabels, 'labelsize':labelsize,
        'xlim':xlim, 'ylim':ylim, 'sharex':sharex, 'sharey':sharey,
        'xlabel':xlabel, 'ylabel':ylabel,
        'xtick_params':xtick_params, 'ytick_params':ytick_params,
    }

    if type(axs) is np.ndarray:
        if abc is not None:
            addabc(axs, abc, abcloc, order)

    if ltitle is not None:
        addtitle(axs, ltitle, loc='left', fontsize=titlesize)
    if rtitle is not None:
        addtitle(axs, rtitle, loc='right', fontsize=titlesize)
    if title is not None:
        addtitle(axs, title, loc='center', fontsize=titlesize)

    if toplabels is not None:
        assert len(toplabels) == axs.shape[-1]
        if toplabels_kw is None:
            toplabels_kw = {'y':1.2, 'fontsize':'large'}
        for i, ax in enumerate(axs[0, :]):
            ax.text(x=0.5, s=toplabels[i], transform=ax.transAxes,
                    ha='center', **toplabels_kw)
            # ha: horizonta lalignment {'left', 'center', 'right'}
            # fontweight='bold'

    # if 'latlim' in geotick_kw or 'lonlim' in geotick_kw:
    #     geo = True

    if type(axs) is np.ndarray:
        for ax in axs.flat:
            if geo and type(ax) is cartopy.mpl.geoaxes.GeoAxes:
                geoticks(ax, **geotick_kw)
            else:
                xyticks(ax, **xytick_kw)
    else:
        if geo and type(axs) is cartopy.mpl.geoaxes.GeoAxes:
            geoticks(axs, **geotick_kw)
        else:
            xyticks(axs, **xytick_kw)


def geoticks(ax, labels=False, labelsize='large',
             coast=True, coastcolor='silver', coastwidth=1.0, reso='low',
             land=False, river=False, lake=False,
             # borders:bool=False,
             latlim=None, lonlim=None, latloc=None, lonloc=None,
             xticks=None, yticks=None, boundinglat=None,
             # len_major=0, len_minor=0,
             grid=True, gridline_kw=None,
             land_kw=None, river_kw=None, lake_kw=None,
             sharex=False, sharey=False):

    assert reso in ['high', 'med', 'low']
    reso_dict = {'high':'10m', 'med':'50m', 'low':'110m'}

    gridline_default = {'linestyle': '--', 'linewidth': 0.8, 'alpha': 0.8,
                        'xpadding':tickpad, 'ypadding':tickpad,
                        }
    river_default = {'linewidth':0.8, 'alpha':1.} # zorder=0
    lake_default = {'linewidth':0.7, 'alpha':1., 'edgecolor':'silver',
                    'facecolor':'none'} # zorder=0, default: cfeat.COLORS['water']

    if gridline_kw is None:
        gridline_kw = gridline_default
    else:
        gridline_kw = {**gridline_default, **gridline_kw}

    river_kw = river_default if river_kw is None else {**river_default, **river_kw}
    lake_kw = lake_default if lake_kw is None else {**lake_default, **lake_kw}
    land_kw = {'facecolor':'silver'} if land_kw is None else land_kw  #

    # lon1, lon2 = (0, 360) if ax.projection._proj4_params['lon_0']==180 else (-180, 180)
    latlower, latupper = (-90, 90) if latlim is None else latlim
    lonleft, lonright = (-180, 180) if lonlim is None else lonlim

    if boundinglat is not None:
        lonleft, lonright, latlower, latupper = [-180, 180, boundinglat, 90] \
            if boundinglat > 0 else [-180, 180, -90, boundinglat]

    map_extent = [lonleft, lonright, latlower, latupper]
    if latlim is not None or lonlim is not None or boundinglat is not None:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        # ax.map_extent = map_extent

    if boundinglat is not None:
        # 缩减地图范围,并维持圆形的边界.摘自cartopy官网.
        theta = np.linspace(0, 2 * np.pi, 1000)
        # 这里center稍微向上偏移一点,让后面加上的label效果更好.
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    ##----Set labels
    if latloc is None:
        latloc = round(((latupper - latlower) // 6) / 10) * 10
    if lonloc is None:
        lonloc = round(((lonright - lonleft) // 6) / 10) * 10

    if xticks is None:
        xticks = generate_multiples(lonloc, start=lonleft, end=lonright)
        xticks[xticks == 360] = -0
        xticks[xticks > 180] -= 360
    if yticks is None:
        yticks = generate_multiples(latloc, start=max(latlower, -89), end=min(latupper, 89))

    # if labels:
    #     ax.xaxis.set_major_formatter(lon_formatter)
    #     ax.yaxis.set_major_formatter(lat_formatter)
    #     # ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
    #     # ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
    #     ax.tick_params(axis='both', labelsize=labelsize)
    #     ax.tick_params(axis='both', which='major', width=1, length=len_major)
    #     ax.tick_params(axis='both', which='minor', width=0.6, length=len_minor)
    #     # ax.set_xticks(np.arange(xticks[0], xticks[1], xticks[2]), crs=ccrs.PlateCarree())
    #     # ax.set_yticks(np.arange(yticks[0], yticks[1], yticks[2]), crs=ccrs.PlateCarree())
    #     ax.set_xticks(ticks=xticks, crs=ccrs.PlateCarree())
    #     ax.set_yticks(ticks=yticks, crs=ccrs.PlateCarree())

    ##----Add geo information
    if coast:
        ax.add_feature(cfeat.COASTLINE.with_scale(reso_dict[reso]),
                       linewidth=coastwidth, edgecolor=coastcolor) # default: cfeat.COLORS['land']
        # ax.coastlines(resolution=reso_dict[reso])

    if land:
        ax.add_feature(cfeat.LAND.with_scale(reso_dict[reso]), **land_kw)

    if river:
        ax.add_feature(cfeat.RIVERS.with_scale('110m'), **river_kw)  # reso_dict[reso]

    if lake:
        ax.add_feature(cfeat.LAKES.with_scale('110m'), **lake_kw)

    ##----Add gridlines
    if grid & ~hasattr(ax, 'grid_and_label'):
        # ax.grid(True, which='major')  # , draw_labels=False
        gl = ax.gridlines(draw_labels=labels, **gridline_kw)
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        if labels:
            gl.left_labels = True
            gl.right_labels = False
            gl.top_labels = False
            gl.bottom_labels = True
            # gl.xlabel_style = {'size': 10}
            # gl.ylabel_style = {'size': 10}

            subplotspec = ax.get_subplotspec()
            gridspec = subplotspec.get_gridspec()
            # 获取行数和列数
            nrows, ncols = gridspec.get_geometry()
            row, col = subplotspec.rowspan.start, subplotspec.colspan.start
            if sharex and row != nrows - 1:
                gl.bottom_labels = False
            if sharey and col != 0:
                gl.left_labels = False

        if boundinglat is not None:
            gl.xlabel_style = {'rotation': 0}  # 标签水平放置
            gl.y_inline = False

        ax.grid_and_label = True


# def extract_latlon(arr):
#
#     if hasattr(arr, 'longitude'):
#         lon = arr.longitude
#     elif hasattr(arr, 'lon'):
#         lon = arr.lon
#     else:
#         raise ValueError('No Longitude!')
#
#     if hasattr(arr, 'latitude'):
#         lat = arr.latitude
#     elif hasattr(arr, 'lat'):
#         lat = arr.lat
#     else:
#         raise ValueError('No Latitude!')
#
#     return lon, lat


def xyticks(ax, xticks=None, yticks=None, xticklabels=None, yticklabels=None,
            xlim=None, ylim=None, sharex=False, sharey=False,
            xlabel=None, ylabel=None, labelsize='large', # xrotation=0, yrotation=0,
            xminor=None, yminor=None, xtick_params=None, ytick_params=None):

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    if type(xticks) in [int, float]:
        xticks = generate_multiples(xticks, start=x_limits[0], end=x_limits[1])
    if type(yticks) in [int, float]:
        yticks = generate_multiples(yticks, start=y_limits[0], end=y_limits[1])

    if xtick_params is None:
        xtick_params = {}
    if ytick_params is None:
        ytick_params = {}

    xticks_kw, yticks_kw = {}, {}
    xticks_dict = {'xticks':'ticks', 'xticklabels':'labels'}
    yticks_dict = {'yticks':'ticks', 'yticklabels':'labels'}

    # 遍历函数的所有参数
    for key, value in locals().items():
        if key in xticks_dict.keys() and value is not None:
            xticks_kw[xticks_dict[key]] = value
        if key in yticks_dict.keys() and value is not None:
            yticks_kw[yticks_dict[key]] = value

    if xticks_kw != {}:
        ax.set_xticks(**xticks_kw)
    if yticks_kw != {}:
        ax.set_yticks(**yticks_kw)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    subplotspec = ax.get_subplotspec()
    gridspec = subplotspec.get_gridspec()
    # 获取行数和列数
    nrows, ncols = gridspec.get_geometry()
    row, col = subplotspec.rowspan.start, subplotspec.colspan.start
    if sharex and row != nrows - 1:
        ax.tick_params('x', labelbottom=False)
        ax.set_xlabel(None)
    if sharey and col != 0:
        ax.tick_params('y', labelleft=False)
        ax.set_ylabel(None)

    ax.tick_params(axis='x', **xtick_params)  # , rotation=xrotation
    ax.tick_params(axis='y', **ytick_params)

    if xminor is not None or yminor is not None:
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(None if xminor is None else AutoMinorLocator())
        ax.yaxis.set_minor_locator(None if yminor is None else AutoMinorLocator())


def contourf(ax, var, x=None, y=None, levels=None, cmap='RdBu_r', extend='both', zorder=0,
             **kwargs):  # globe=False,

    if (x is None) & (y is None):
        dims = var.dims
        x, y = var[dims[-1]], var[dims[-2]]

    pic = ax.contourf(x, y, var, levels=levels, cmap=cmap,
                      extend=extend, transform=ccrs.PlateCarree(),
                      zorder=zorder, **kwargs)
    return pic


def uv_sig(u, v, p_u, p_v, pvalue=0.05, method=1):

    assert (type(u) == type(v)) & (type(p_u) == type(p_v))
    u0, v0 = u.copy(), v.copy()
    if method==1:
        if type(u) == xr.DataArray:
            u = u.where((p_u <= pvalue) | (p_v <= pvalue))
            v = v.where((p_u <= pvalue) | (p_v <= pvalue))
            u0 = u0.where((p_u > pvalue) & (p_v > pvalue))
            v0 = v0.where((p_u > pvalue) & (p_v > pvalue))
        else:
            if type(p_u) != np.ndarray:
                p_u = p_u.values
                p_v = p_v.values
            u[(p_u > pvalue) & (p_v > pvalue)] = np.nan
            v[(p_u > pvalue) & (p_v > pvalue)] = np.nan
            u0[(p_u <= pvalue) | (p_v <= pvalue)] = np.nan
            v0[(p_u <= pvalue) | (p_v <= pvalue)] = np.nan
    else:
        p_uv = 1 - np.sqrt(((1-p_u)**2 + (1-p_v)**2)/2.)
        u[(p_uv > pvalue)] = np.nan
        v[(p_uv > pvalue)] = np.nan
        u0[(p_uv <= pvalue)] = np.nan
        v0[(p_uv <= pvalue)] = np.nan

    return u, v, u0, v0


def plt_sig(ax, sig, x=None, y=None, pvalue=0.05, hatches='..', color='k',
            method=1, size=0.7, alpha=0.8, zorder=None):

    if (x is None) & (y is None):
        dims = sig.dims
        x, y = sig[dims[-1]], sig[dims[-2]]
        sig = sig.values

    if np.nanmin(sig) < pvalue:
        if method == 1:
            plt.rcParams['hatch.color'] = color
            ax.contourf(x, y, sig, levels=[np.nanmin(sig), pvalue, np.nanmax(sig)],
                        zorder=zorder, hatches=[hatches, None], colors="none")
        else:
            nx, ny = np.meshgrid(x, y)
            ax.scatter(nx[sig < pvalue], ny[sig < pvalue], marker='.', s=size,
                       c=color, alpha=alpha, zorder=zorder)
    else:
        pass


def vector(ax, vx, vy, x=None, y=None, width=None, headwidth=3.,
           scale=None, ec='w', fc='k', lw=0., units='width', zorder=None):

    if (x is None) & (y is None):
        dims = vx.dims
        x, y = vx[dims[-1]], vx[dims[-2]]

    flux = ax.quiver(x, y, vx, vy, pivot='mid', width=width, headwidth=headwidth,
                     ec=ec, fc=fc, lw=lw, scale=scale, units=units, zorder=zorder)
    return flux


def contour(ax, var, x=None, y=None, level=None, color='k', lw=1.5, ls=None,
            clabel=False, fs='small', space=2, fmt='%1.1f', **kwargs):

    if (x is None) or (y is None):
        dims = var.dims
        x, y = var[dims[-1]], var[dims[-2]]

    pic = ax.contour(x, y, var, levels=level, colors=color, linewidths=lw,
                     linestyles=ls, **kwargs)
    if clabel:
        ax.clabel(pic, inline=True, fontsize=fs, fmt=fmt, inline_spacing=space)


def addquiver(ax, flux, x, y, u, unit=None, labelpos='N', labelsep=0.05,
              coordinates='axes', size='medium'):

    ax.quiverkey(flux, X=x, Y=y, U=u,
                 label=str(u) if unit is None else str(u)+' '+unit,
                 labelpos=labelpos, labelsep=labelsep,
                 coordinates=coordinates, fontproperties={'size': size})


def addpatch(ax, domain, proj=None, ec='k', lw=0.5):

    if proj is None:
        proj = ax.projection

    ##pathpatch
    lonmin, lonmax, latmin, latmax = domain
    lon_span = list(np.linspace(lonmin, lonmax, 50))

    if (type(proj) is ccrs.PlateCarree) | (proj=='cyl'):
        path = patches.Rectangle((lonmin, latmin), lonmax-lonmin, latmax-latmin,
                                 edgecolor=ec, fill=False, lw=lw,
                                 transform=ccrs.PlateCarree(), zorder=99)
    else:
        vertices = [(lon, latmin) for lon in lon_span] + \
            [(lon, latmax) for lon in lon_span[::-1]]
        path = patches.Polygon(np.array(vertices), edgecolor=ec, fill=False, lw=lw,
                               transform=ccrs.PlateCarree(), zorder=99)
    ax.add_patch(path)


def addcolorbar(fig, m, ticks=None, loc='b', shrink=1., blank=False,
                aspect=20, pad=None, extendfrac='auto',
                extendrect=False, span=None, labelsize='medium',
                tickdirection='out', ifoutline=True, label='',
                labelfontsize=None, extend=None, ticklength=3,
                labelpad=1.5, align='center',
                # nax=1, linewidth=None, tickwidth=None, extendsize=None,
                **kwargs,
                ):
    # blank: whether there is blank margin of fig in the direction of colorbar

    loc_dict = {'b':'bottom', 't':'top', 'l':'left', 'r':'right'}

    condition = False
    if isinstance(fig, cartopy.mpl.geoaxes.GeoAxes):
        condition = True
        bbox = fig.get_position()
    else:
        if isinstance(fig, np.ndarray):
            if isinstance(fig.flatten()[0], cartopy.mpl.geoaxes.GeoAxes):
                condition = True
                bbox = fig.flatten()[0].get_position()

    if condition:
        # bbox = fig.get_position()
        if blank:
            if loc in ['b', 't']:
                shrink_coef = bbox.width
            else:
                shrink_coef = bbox.height
            pad_coef = 1
        else:
            shrink_coef = 1
            if loc in ['b', 't']:
                pad_coef = bbox.width
            else:
                pad_coef = bbox.height
    else:
        shrink_coef = 1
        pad_coef = 1

    shrink *= shrink_coef

    if pad is None:
        pad = 0.15 if loc in ['b', 't'] else 0.05
        pad *= pad_coef
        # print('pad = ' + str(pad))

    if ticks is None and hasattr(m, 'levels'):
        ticks = m.levels
    else:
        if type(ticks) in [int, float]:
            # ticks = np.arange(m.levels.min(), m.levels.max(), ticks)
            if hasattr(m, 'levels'):
                ticks = generate_multiples(ticks, m.levels.min(), m.levels.max())
            else:
                vmin, vmax = m.get_clim()
                ticks = generate_multiples(ticks, vmin, vmax)
        else:
            pass

    if labelfontsize is None:
        labelfontsize = labelsize

    # cb_default = {key: kwargs[key] for key in ('aspect', 'extendfrac', 'extendrect', 'pad')
    #               if key in kwargs}
    cb_kw = dict(ticks=ticks, location=loc_dict[loc], aspect=aspect, shrink=shrink,
                 pad=pad, extendfrac=extendfrac, extendrect=extendrect) # extendsize=extendsize,

    if span is not None:
        cb_kw['span'] = span
    if extend is not None:
        cb_kw['extend'] = extend

    if type(fig) is matplotlib.figure.Figure:
        cb = fig.colorbar(m, **cb_kw)
    else:
        cb = plt.colorbar(m, ax=fig, **cb_kw)
    # cb = fig.colorbar(m, **cb_kw)  # , align=align,

    tick_kw = dict(labelsize=labelsize, direction=tickdirection)
    if ticklength is not None:
        tick_kw['length'] = ticklength
    cb.ax.tick_params(**tick_kw)  # length=ticklength,

    if label != '':
        cb.set_label(label, fontsize=labelfontsize, labelpad=labelpad)
    else:
        cb.set_label(None)

    if not ifoutline:
        # cb.outline.set_edgecolor('white')
        cb.outline.set_linewidth(0)
    # else:
    #     cb.outline.set_linewidth(linewidth)
    cb.minorticks_off()


def addchina(ax, ec='k', lw=0.5, province=False, scs=False, scs_domain=None, scs_ax_pos=None,
             dir_china=None):

    if province:
        import cartopy.io.shapereader as shpreader
        if dir_china is None:
            dir_china = r'D:\research\data\china-shapefiles\bou2_4m\bou2_4l.shp'
        # self.ax.add_feature(cfeat.RIVERS.with_scale('50m'), linewidth=1.5)
        china = shpreader.Reader(dir_china).geometries()
        plot_obj = china
        ax.add_geometries(plot_obj, fc='none', crs=ccrs.PlateCarree(), ec=ec, lw=lw)  # ax.projection
    else:
        import geopandas as gpd
        if dir_china is None:
            dir_china = r'D:\research\data\china-shapefiles\aliyun\shp\china_country/china_country.shp'
        china = gpd.read_file(dir_china)
        plot_obj = china.geometry[china['type'].isin(['国界', '争议', '港澳'])]
        ax.add_geometries(plot_obj, crs=ccrs.PlateCarree(),
                          facecolor='none', ec=ec, lw=lw, zorder=99)  # '海洋', '省界'  # ccrs.PlateCarree()

    if scs:
        if scs_domain is None:
            scs_domain = [105, 125, 0, 25]
        if scs_ax_pos is None:
            scs_ax_pos = [0.8, 0.05, 0.25, 0.35]
        # if scs_line_kw is None:
        #     scs_line_kw = {'edgecolor':'k', 'linewidth':.3}
        # if scs_ax_kw is None:
        #     scs_ax_kw = {'coast':True, 'grid':False}  # 'reso':'med',
        ax2 = ax.inset_axes(scs_ax_pos)
        ax2.set_extent(scs_domain, crs=ccrs.PlateCarree())
        ax2.format(grid=False)  # **scs_ax_kw
        ax2.add_geometries(plot_obj, crs=ccrs.PlateCarree(), facecolor='none', ec=ec, lw=lw)

        return ax2


def addabc(axs, abc, abcloc, order='C'):
    assert order in ['C', 'F']
    abc_formats = {
        '(a)': "({})",
        'a)': "{})",
        'a.': "{}.",
        'a': "{}"
    }
    start_letter = ord('a')

    if abc not in abc_formats:
        raise ValueError('This type of abc is not supported!')

    axs_flat = axs.flatten(order=order) if order == 'F' else axs.flat
    for i, ax in enumerate(axs_flat):
        ax.set_title(abc_formats[abc].format(chr(start_letter + i)), loc=abcloc)


def addtitle(axs, title, loc, fontsize='large'):
    if type(axs) is np.ndarray:
        for ax in axs:
            title_prefix = ax.get_title(loc)
            ax.set_title(title_prefix + ' ' + title, loc=loc, fontsize=fontsize)
    else:
        title_prefix = axs.get_title(loc)
        axs.set_title(title_prefix + ' ' + title, loc=loc, fontsize=fontsize)


def generate_multiples(base_num, start, end):
    # 计算正负方向的最大倍数
    lim = max(abs(start), abs(end))
    pos_limit = int(lim // base_num)
    # neg_limit = int(-start // base_num)

    # 生成负向和正向的倍数
    # neg_multiples = np.arange(base_num * neg_limit, 0, base_num)
    pos_multiples = np.arange(0, base_num * (pos_limit + 1), base_num)
    neg_multiples = -pos_multiples[1:][::-1]
    all_multiples = np.concatenate((neg_multiples, pos_multiples))

    # 合并负向和正向的结果，并返回
    loc = np.where((all_multiples>=start) & (all_multiples<=end))[0]
    return all_multiples[loc]


