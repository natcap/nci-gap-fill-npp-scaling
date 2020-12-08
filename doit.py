"""
Objectives: To take global maps of timber and grazing production functions
and (1) fill in gaps in those maps using convolution, (2) scale the
production of each timber or grazing production function by NPP based on
current ESA land use, and (3) fill the gaps in between current production of
grazing of forestry to create continuous maps of potential production of
grazing or timber lands if future land use conditions allowed for those
activities.
"""
import logging
import os

from osgeo import gdal
import ecoshard
import pygeoprocessing
import numpy
import scipy.ndimage
import taskgraph

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

ECOSHARD_ROOT = 'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/Data-20201207T190350Z-001_md5_63b09dd643d3e686a5d83a9e340c90d2.zip'
WORKSPACE_DIR = 'workspace'
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
COUNTRY_WORKSPACE_DIR = os.path.join(WORKSPACE_DIR, 'country_workspaces')

COUNTRY_VECTOR_PATH = os.path.join(
    DATA_DIR, 'Data', 'supportingLayers', 'countries_iso3_NCI_May.shp')
ESA_LULC_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'supportingLayers', 'lulc.tif')
NPP_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'supportingLayers',
    'MOD17A3_Science_NPP_mean_00_15.tif')
GRAZING_ZONE_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'supportingLayers', 'aez.tif')
TIMBER_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'timberLayers', 'global_annual_biomass_per_ha.tif')

CURRENT_MEAT_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'current_grass_meat.tif')
POTENTIAL_MEAT_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'potential_methane_gap_filled_cur.tif')
CURRENT_METHANE_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'current_grass_methane.tif')
POTENTIAL_MEAT_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'potential_meat_gap_filled_cur.tif')
MAX_FILL_DIST_DEGREES = 0.5

RASTER_DATA_STACK = [
    ESA_LULC_RASTER_PATH,
    NPP_RASTER_PATH,
    GRAZING_ZONE_RASTER_PATH,
    TIMBER_RASTER_PATH,
    CURRENT_MEAT_PROD_RASTER_PATH,
    POTENTIAL_MEAT_PROD_RASTER_PATH,
    CURRENT_METHANE_PROD_RASTER_PATH,
]

PRODUCTION_ZONE_RASTER_STACK = [
    GRAZING_ZONE_RASTER_PATH,
    TIMBER_RASTER_PATH,
]


def _fill_nodata_op(base, fill, nodata):
    result = numpy.copy(base)
    if nodata is not None:
        nodata_mask = numpy.isclose(base, nodata)
        result[nodata_mask] = fill[nodata_mask]
    return result


def fill_by_convolution(
        base_raster_path, convolve_radius, target_filled_raster_path):
    """Fill nodata holes in base as weighted average of valid pixels.

    Args:
        base_raster_path (str): path to base raster
        convolve_radius (float): maximum convolution distance kernel in
            projected units of base.
        target_filled_raster_path (str): raster created by convolution fill,
            if holes are too far from valid pixels resulting fill will be
            nonsensical, perhaps NaN.

    Returns:
        None.
    """
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    n = max(5, int(convolve_radius / base_raster_info['pixel_size'][0]))
    base = numpy.zeros((n, n))
    base[n//2, n//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(base, n/5)
    target_dir = os.path.dirname(target_filled_raster_path)
    basename = os.path.basename(target_filled_raster_path)
    kernel_raster_path = os.path.join(target_dir, f'kernel_{basename}')
    geotransform = base_raster_info['geotransform']
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, None, base_raster_info['pixel_size'],
        (geotransform[0], geotransform[3]),
        base_raster_info['projection_wkt'], kernel_raster_path)
    backfill_raster_path = os.path.join(target_dir, f'backfill_{basename}')
    pygeoprocessing.convolve_2d(
        (base_raster_path, 1), (kernel_raster_path, 1),
        backfill_raster_path, ignore_nodata_and_edges=True,
        mask_nodata=False, normalize_kernel=True,
        working_dir=target_dir)

    base_nodata = base_raster_info['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (backfill_raster_path, 1),
         (base_nodata, 'raw')], _fill_nodata_op, target_filled_raster_path,
        gdal.GDT_Float32, base_nodata)
    os.remove(kernel_raster_path)
    os.remove(backfill_raster_path)


def clip_raster_stack(
        raster_path_list, bounding_box, target_dir, task_graph):
    """Clip the input raster path to the bounding box.

    Args:
        raster_path_list (list): list of arbitrary raster paths.
        bounding_box (tuple): a 4 tuple indicating minx, miny, maxx, maxy
        target_dir (str): desired path to place clipped rasters from
            ``raster_path_list``, they will have the same basename.
        task_graph (TaskGraph): taskgraph object that can be used for
            scheduling.

    Returns:
        List of Tasks to clip.

    """
    task_list = []
    for raster_path in raster_path_list:
        clip_raster_path = os.path.join(
            target_dir, f'clip_{os.path.basename(raster_path)}')
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        clip_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                raster_path, raster_info['pixel_size'], clip_raster_path,
                'near'),
            kwargs={'target_bb': bounding_box},
            target_path_list=[clip_raster_path],
            task_name=f'warp {raster_path} to {clip_raster_path}')
        fill_raster_path = os.path.join(
            target_dir, f'fill_{os.path.basename(raster_path)}')
        fill_task = task_graph.add_task(
            func=fill_by_convolution,
            args=(
                clip_raster_path, MAX_FILL_DIST_DEGREES, fill_raster_path),
            dependent_task_list=[clip_task],
            target_path_list=[fill_raster_path],
            task_name=f'fill {clip_raster_path} to {fill_raster_path}')
        task_list.append(fill_task)
    return fill_task


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, DATA_DIR, COUNTRY_WORKSPACE_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)
    token_path = os.path.join(DATA_DIR, 'download.token')
    fetch_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(ECOSHARD_ROOT, DATA_DIR),
        kwargs={'target_token_path': token_path},
        target_path_list=[token_path],
        task_name=f'fetch {ECOSHARD_ROOT}')
    fetch_task.join()

    country_vector = gdal.OpenEx(COUNTRY_VECTOR_PATH, gdal.OF_VECTOR)
    country_layer = country_vector.GetLayer()
    # try just on costa ricas
    country_layer.SetAttributeFilter("iso3='CRI'")
    for country_feature in country_layer:
        country_iso = country_feature.GetField('ISO3')
        LOGGER.info(f'processing {country_iso}')
        country_workspace = os.path.join(COUNTRY_WORKSPACE_DIR, country_iso)
        try:
            os.makedirs(country_workspace)
        except OSError:
            pass
        country_geom = country_feature.GetGeometryRef()
        envelope = country_geom.GetEnvelope()
        # swizzle so we get xmin, ymin, xmax, ymax order
        bounding_box = [envelope[x] for x in [0, 2, 1, 3]]
        LOGGER.debug(bounding_box)
        clip_raster_stack(
            RASTER_DATA_STACK, bounding_box, country_workspace, task_graph)

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
