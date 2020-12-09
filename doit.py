"""
NCI Gap Fill/NPP Scaling Script.

Objectives: To take global maps of timber and grazing production functions
and (1) fill in gaps in those maps using convolution, (2) scale the
production of each timber or grazing production function by NPP based on
current ESA land use, and (3) fill the gaps in between current production of
grazing of forestry to create continuous maps of potential production of
grazing or timber lands if future land use conditions allowed for those
activities.
"""
import collections
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

ECOSHARD_ROOT = (
    'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/'
    'Data-20201207T190350Z-001_md5_63b09dd643d3e686a5d83a9e340c90d2.zip')
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
    DATA_DIR, 'Data', 'grazingLayers', 'potential_meat_gap_filled_cur.tif')
CURRENT_METHANE_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'current_grass_methane.tif')
POTENTIAL_METHANE_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'potential_methane_gap_filled_cur.tif')
MAX_FILL_DIST_DEGREES = 0.5

RASTER_ID_PATH_STACK = [
    ('esa', ESA_LULC_RASTER_PATH),
    ('npp', NPP_RASTER_PATH),
    ('grazing_zone', GRAZING_ZONE_RASTER_PATH),
    ('timber', TIMBER_RASTER_PATH),
    ('cur_meat_prod', CURRENT_MEAT_PROD_RASTER_PATH),
    ('potential_meat_prod', POTENTIAL_MEAT_PROD_RASTER_PATH),
    ('cur_methane_prod', CURRENT_METHANE_PROD_RASTER_PATH),
    ('potential_methane_prod', POTENTIAL_METHANE_PROD_RASTER_PATH)
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


def clip_and_fill_by_convolution(
        base_raster_path, bounding_box, convolve_radius,
        target_filled_raster_path):
    """Clip and fill.

    Clip the base raster data to the bounding box then fill any noodata
    holes with a weighted distance convolution.

    Args:
        base_raster_path (str): path to base raster
        bounding_box (tuple): a 4-tuple indicating xmin, ymin, xmax, ymax
            to clip base data to.
        convolve_radius (float): maximum convolution distance kernel in
            projected units of base.
        target_filled_raster_path (str): raster created by convolution fill,
            if holes are too far from valid pixels resulting fill will be
            nonsensical, perhaps NaN.

    Return:
        None
    """
    target_dir = os.path.dirname(target_filled_raster_path)
    basename = os.path.basename(target_filled_raster_path)
    clip_raster_path = os.path.join(target_dir, f'clip_{basename}')
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    pygeoprocessing.warp_raster(
        base_raster_path, base_raster_info['pixel_size'],
        clip_raster_path, 'near', target_bb=bounding_box)

    # this ensures a minimum of 5 pixels in case the pixel size is too
    # chunky
    n = max(5, int(convolve_radius / base_raster_info['pixel_size'][0]))
    base = numpy.zeros((n, n))
    base[n//2, n//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(base, n/5)
    kernel_raster_path = os.path.join(target_dir, f'kernel_{basename}')
    geotransform = base_raster_info['geotransform']
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, None, base_raster_info['pixel_size'],
        (geotransform[0], geotransform[3]),
        base_raster_info['projection_wkt'], kernel_raster_path)
    backfill_raster_path = os.path.join(target_dir, f'backfill_{basename}')
    pygeoprocessing.convolve_2d(
        (clip_raster_path, 1), (kernel_raster_path, 1),
        backfill_raster_path, ignore_nodata_and_edges=True,
        mask_nodata=False, normalize_kernel=True,
        working_dir=target_dir)

    base_nodata = base_raster_info['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(clip_raster_path, 1), (backfill_raster_path, 1),
         (base_nodata, 'raw')], _fill_nodata_op, target_filled_raster_path,
        gdal.GDT_Float32, base_nodata)
    os.remove(clip_raster_path)
    os.remove(kernel_raster_path)
    os.remove(backfill_raster_path)


def clip_fill_scale(
        class_raster_path, value_raster_path, scale_raster_path,
        lulc_raster_path, valid_lulc_code_list,
        mask_vector_path, mask_vector_where_filter,
        target_dir, task_graph):
    """Clip, fill, scale value raster.

    This ensures base data are filled in, clipped and masked to the given
    feature, and scaled by the scale/scale_median in the scale raster
    per each class in the class raster. This is further filtered by only
    operating on pixels defined in lulc_raster_path that have codes in
    valid_lulc_code list.

    Args:
        class_raster_path (str): path to raster that uniquely identifies
            classes that should be used to calculate median scale.
        value_raster_path (str): primary value raster that should be filled
            and scaled
        scale_raster_path (str): path to raster used to scale per unique
            class in class_raster_path
        lulc_raster_path (str): landcover code class used to further filter
            what should be processed
        valid_lulc_code_list (list): list of landcover codes in
            lulc_raster_path that limit which pixels are processed.
        mask_vector_path (str): path to a vector used to mask raster stack.
        mask_vector_where_filter (str): where filter to indicate which
            feature to filter by.
        target_dir (str): desired path to place clipped rasters from
            ``raster_path_list``, they will have the same basename.
        task_graph (TaskGraph): taskgraph object that can be used for
            scheduling.

    Return:
        (scaled_raster_path, task for that raster)
    """
    task_path_map = dict()
    mask_vector = gdal.OpenEx(mask_vector_path, gdal.OF_VECTOR)
    mask_layer = mask_vector.GetLayer()
    mask_layer.SetAttributeFilter(mask_vector_where_filter)
    mask_feature = next(iter(mask_layer))
    mask_geom = mask_feature.GetGeometryRef()
    envelope = mask_geom.GetEnvelope()
    # swizzle so we get xmin, ymin, xmax, ymax order
    bounding_box = [envelope[i] for i in [0, 2, 1, 3]]

    for raster_id, raster_path in [
            ('value', value_raster_path),
            ('scale', scale_raster_path)]:
        fill_raster_path = os.path.join(
            target_dir, f'fill_{os.path.basename(raster_path)}')
        fill_task = task_graph.add_task(
            func=clip_and_fill_by_convolution,
            args=(
                raster_path, bounding_box, MAX_FILL_DIST_DEGREES,
                fill_raster_path),
            target_path_list=[fill_raster_path],
            task_name=f'clip and fill {raster_path} to {fill_raster_path}')
        task_path_map[raster_id] = {
            'task': fill_task,
            'path': fill_raster_path}

    scaled_biomass_raster_path = os.path.join(
        target_dir,
        f"scaled_{os.path.basename(task_path_map['timber']['path'])}")

    scale_task = task_graph.add_task(
        func=scale_value,
        args=(
            task_path_map['value']['path'],
            task_path_map['scale']['path'],
            class_raster_path,
            lulc_raster_path,
            valid_lulc_code_list,
            mask_vector_path, mask_vector_where_filter,
            scaled_biomass_raster_path),
        target_path_list=[scaled_biomass_raster_path],
        dependent_task_list=[
            task_path_map[raster_id]['task']
            for raster_id in ['timber', 'npp', 'esa']],
        task_name=(
            f"scale {task_path_map['timber']['path']} to "
            f"{scaled_biomass_raster_path}"))
    return (scaled_biomass_raster_path, scale_task)


def get_unique_raster_values(raster_path):
    """Return list of unique raster values in ``raster_path``."""
    unique_vals = set()
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    for _, array in pygeoprocessing.iterblocks((raster_path, 1)):
        unique_vals |= set(array[~numpy.isclose(array, nodata)])
    return unique_vals


def scale_value(
        value_raster_path, scale_raster_path, class_raster_path,
        lulc_raster_path, valid_lulc_code_list, mask_vector_path,
        mask_vector_where_filter, target_scaled_raster_path):
    """Scale biomass by NPP.

    Scale biomass by mean NPP in unique zones in regions that are allowed
    by the landcover type.

    Args:
        value_raster_path (str): path to base value raster to scale.
        scale_raster_path (str): Raster to scale value by weighted normalized
            values.
        class_raster_path (str): path to raster with unique values, used to
            normalize scale per class.
        lulc_raster_path (str): path to landcover raster, used to
            guide where changes can occur.
        valid_lulc_code_list (str): list of landcover codes to filter
            scale raster by -- only allow values through where these
            codes are defined.
        mask_vector_path (str): path to a vector used to mask raster stack.
        mask_vector_where_filter (str): where filter to indicate which
            feature to filter by.
        target_scaled_raster_path (str): created by this function,
            values from biomass raster scaled by NPP in unique functional
            zones.

    Return:
        None
    """
    # the biomass zones are uniquely identified by their unique floating
    # point values. Jeff said this was okay.
    unique_class_vals = get_unique_raster_values(class_raster_path)

    # Align raster stack
    working_dir = os.path.dirname(target_scaled_raster_path)
    base_raster_path_list = [
        biomass_raster_path, esa_lulc_raster_path, npp_raster_path]
    aligned_raster_path_list = [
        os.path.join(working_dir, f'aligned_{os.path.basename(path)}')
        for path in base_raster_path_list]
    esa_info = pygeoprocessing.get_raster_info(esa_lulc_raster_path)
    pygeoprocessing.align_and_resize_raster_stack(
        base_raster_path_list, aligned_raster_path_list,
        ['average', 'average', 'near'], esa_info['pixel_size'],
        'intersection', vector_mask_options={
            'mask_vector_path': mask_vector_path,
            'mask_vector_where_filter': mask_vector_where_filter})

    # create output raster
    target_nodata = -1
    pygeoprocessing.new_raster_from_base(
        aligned_raster_path_list[0], target_scaled_raster_path,
        gdal.GDT_Float32, [target_nodata])
    target_scaled_biomass_raster = gdal.OpenEx(
        target_scaled_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    target_scaled_biomass_band = (
        target_scaled_biomass_raster.GetRasterBand(1))

    # open these for fast reading
    biomass_raster = gdal.OpenEx(aligned_raster_path_list[0], gdal.OF_RASTER)
    biomass_band = biomass_raster.GetRasterBand(1)
    esa_raster = gdal.OpenEx(aligned_raster_path_list[1], gdal.OF_RASTER)
    esa_band = esa_raster.GetRasterBand(1)
    npp_raster = gdal.OpenEx(aligned_raster_path_list[2], gdal.OF_RASTER)
    npp_band = npp_raster.GetRasterBand(1)

    # calculate the mean per biomass value per biomass zone
    npp_sum = collections.defaultdict(float)
    npp_count = collections.defaultdict(int)
    for offset_dict in pygeoprocessing.iterblocks(
            (aligned_raster_path_list[0], 1), offset_only=True):
        npp_array = npp_band.ReadAsArray(**offset_dict)
        biomass_array = biomass_band.ReadAsArray(**offset_dict)
        esa_array = esa_band.ReadAsArray(**offset_dict).astype(numpy.int32)
        esa_timber_mask = numpy.in1d(
            esa_array, lulc_timber_allowed_codes).reshape(esa_array.shape)
        for biomass_zone_val in unique_class_vals:
            valid_biomass_mask = numpy.isclose(
                biomass_array, biomass_zone_val)
            valid_mask = valid_biomass_mask & esa_timber_mask
            npp_count[biomass_zone_val] += numpy.count_nonzero(valid_mask)
            npp_sum[biomass_zone_val] += numpy.sum(npp_array[valid_mask])

    # adjust the biomass value by the npp mean
    for offset_dict in pygeoprocessing.iterblocks(
            (aligned_raster_path_list[0], 1), offset_only=True):
        npp_array = npp_band.ReadAsArray(**offset_dict)
        biomass_array = biomass_band.ReadAsArray(**offset_dict)
        esa_array = esa_band.ReadAsArray(**offset_dict).astype(numpy.int32)
        target_scaled_biomass_array = numpy.empty(npp_array.shape)
        target_scaled_biomass_array[:] = target_nodata
        esa_timber_mask = numpy.in1d(
            esa_array, lulc_timber_allowed_codes).reshape(esa_array.shape)
        for biomass_zone_val in unique_class_vals:
            if npp_count[biomass_zone_val] == 0:
                continue
            npp_mean = (
                npp_sum[biomass_zone_val] / npp_count[biomass_zone_val])
            valid_biomass_mask = numpy.isclose(
                biomass_array, biomass_zone_val)
            valid_mask = valid_biomass_mask & esa_timber_mask
            target_scaled_biomass_array[valid_mask] = (
                biomass_array[valid_mask] * npp_array[valid_mask] / npp_mean)
        target_scaled_biomass_band.WriteArray(
            target_scaled_biomass_array,
            xoff=offset_dict['xoff'], yoff=offset_dict['yoff'])


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
    scaled_timber_raster_path_list = []
    for country_feature in country_layer:
        country_iso = country_feature.GetField('ISO3')
        LOGGER.info(f'processing {country_iso}')
        country_workspace = os.path.join(COUNTRY_WORKSPACE_DIR, country_iso)
        try:
            os.makedirs(country_workspace)
        except OSError:
            pass
        scaled_raster_task_tuple = process_timber_stack(
            RASTER_ID_PATH_STACK, COUNTRY_VECTOR_PATH,
            f"iso3='{country_iso}'", country_workspace, task_graph)
        scaled_timber_raster_path_list.append(scaled_raster_task_tuple)

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
