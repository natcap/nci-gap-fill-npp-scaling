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
import taskgraph

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

ECOSHARD_ROOT = 'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/Data-20201207T190350Z-001_md5_63b09dd643d3e686a5d83a9e340c90d2.zip'
WORKSPACE_DIR = 'workspace'
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')

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

RASTER_DATA_STACK = [
    ESA_LULC_RASTER_PATH,
    NPP_RASTER_PATH,
    GRAZING_ZONE_RASTER_PATH,
    TIMBER_RASTER_PATH,
    CURRENT_MEAT_PROD_RASTER_PATH,
    POTENTIAL_MEAT_PROD_RASTER_PATH,
    CURRENT_METHANE_PROD_RASTER_PATH,
    POTENTIAL_MEAT_PROD_RASTER_PATH,
]

PRODUCTION_ZONE_RASTER_STACK = [
    GRAZING_ZONE_RASTER_PATH,
    TIMBER_RASTER_PATH,
]


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
    pass


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
        target_raster_path = os.path.join(
            target_dir, os.path.basename(raster_path))
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                raster_path, raster_info['pixel_size'], target_raster_path,
                'near'),
            kwargs={'target_bb': bounding_box},
            target_path_list=[target_raster_path],
            task_name=f'warp {raster_path} to {target_raster_path}')
        task_list.append(task)
    return task


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, DATA_DIR]:
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
    for country_feature in country_layer:
        country_iso = country_feature.GetField('ISO3')
        LOGGER.info(f'processing {country_iso}')
        country_workspace = os.path.join(WORKSPACE_DIR, country_iso)
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
        break

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
