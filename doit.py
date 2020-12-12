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
import multiprocessing
import os
import queue
import sys
import warnings
warnings.filterwarnings('error')

from osgeo import gdal
import ecoshard
import pygeoprocessing
import numpy
import scipy.ndimage
import taskgraph

gdal.SetCacheMax(2**26)

N_CPUS = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.DEBUG,
    filename='log.txt',
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('pygeoprocessing').setLevel(logging.DEBUG)

ECOSHARD_ROOT = (
    'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/'
    'Data-20201210-2-timber-layer_md5_dbf4ef0857bbe7f8fafda800a9eae099.zip')

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
ANNUAL_BIOMASS_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'timberLayers', 'global_annual_biomass_per_ha.tif')
PLT_AN_BIO_PROJ_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'timberLayers', 'plt_an_bio_proj.tif')


CURRENT_MEAT_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'current_grass_meat.tif')
POTENTIAL_MEAT_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'potential_meat_gap_filled_cur.tif')
CURRENT_METHANE_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'current_grass_methane.tif')
POTENTIAL_METHANE_PROD_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'grazingLayers', 'potential_methane_gap_filled_cur.tif')
MAX_FILL_DIST_DEGREES = 0.5

FORESTRY_VALID_LULC_LIST = [
    50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170]
GRAZING_VALID_LULC_LIST = [
    30, 40, 100, 110, 120, 121, 122, 130, 140, 150, 152, 153, 180, 200, 201,
    202, 203]

def _fill_nodata_op(base, fill, nodata):
    result = numpy.copy(base)
    if nodata is not None:
        nodata_mask = numpy.isclose(base, nodata)
        result[nodata_mask] = fill[nodata_mask]
    return result


def _scrub_nan(base_array, base_nodata):
    """Scrub NaNs to nodata as arbitrary nodata."""
    nan_mask = numpy.isnan(base_array)
    if numpy.any(nan_mask):
        base_array[nan_mask] = base_nodata
    return base_array


def fill_by_convolution(
        base_raster_path, convolve_radius, target_filled_raster_path):
    """Clip and fill.

    Clip the base raster data to the bounding box then fill any noodata
    holes with a weighted distance convolution.

    Args:
        base_raster_path (str): path to base raster
        convolve_radius (float): maximum convolution distance kernel in
            projected units of base.
        target_filled_raster_path (str): raster created by convolution fill,
            if holes are too far from valid pixels resulting fill will be
            nonsensical, perhaps NaN.

    Return:
        None
    """
    try:
        LOGGER.info(f'filling {base_raster_path}')
        target_dir = os.path.dirname(target_filled_raster_path)
        basename = os.path.basename(target_filled_raster_path)
        base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)

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
        base_nodata = base_raster_info['nodata'][0]
        if base_nodata is None:
            target_datatype = gdal.GDT_Float64
        else:
            target_datatype = base_raster_info['datatype']
        LOGGER.info(f'convolve 2d on {base_raster_path} {backfill_raster_path}')
        pygeoprocessing.convolve_2d(
            (base_raster_path, 1), (kernel_raster_path, 1),
            backfill_raster_path, ignore_nodata_and_edges=True,
            mask_nodata=False, normalize_kernel=True,
            target_nodata=base_nodata,
            target_datatype=target_datatype,
            working_dir=target_dir)

        LOGGER.info(
            f'fill nodata of {base_raster_path} to {backfill_raster_path}')
        pygeoprocessing.raster_calculator(
            [(base_raster_path, 1), (backfill_raster_path, 1),
             (base_nodata, 'raw')], _fill_nodata_op, target_filled_raster_path,
            base_raster_info['datatype'], base_nodata)
        os.remove(kernel_raster_path)
        os.remove(backfill_raster_path)
    except Exception:
        LOGGER.exception(
            f'error on fill by convolution {target_filled_raster_path}')
        raise


def _clip_raster(raster_path, bounding_box, target_clip_raster_path):
    """Clip raster to target.

    Args:
        raster_path (str): path to base raster
        bounding_box (tuple): path to 4 element lat/lng bounding box
        target_clip_raster_path (str): raster clipped to the bounding
            box.

    """
    try:
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        nan_scrub_raster_path = os.path.join(
            os.path.dirname(target_clip_raster_path),
            f'nan_scrub_{os.path.basename(raster_path)}')
        if bounding_box[0] <= -180:
            bounding_box[0] = -179.99
        if bounding_box[2] >= 180:
            bounding_box[2] = 179.99

        pygeoprocessing.warp_raster(
            raster_path, raster_info['pixel_size'],
            nan_scrub_raster_path, 'near', target_bb=bounding_box)
        pygeoprocessing.raster_calculator(
            [(nan_scrub_raster_path, 1), (raster_info['nodata'][0], 'raw')],
            _scrub_nan, target_clip_raster_path, raster_info['datatype'],
            raster_info['nodata'][0])
    except Exception:
        LOGGER.exception(f'error on clip {target_clip_raster_path}')
        raise


def clip_fill_scale(
        value_raster_path, class_raster_path, scale_raster_path,
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
        value_raster_path (str): primary value raster that should be filled
            and scaled
        class_raster_path (str): path to raster that uniquely identifies
            classes that should be used to calculate median scale.
        scale_raster_path (str): path to raster used to scale per unique
            class in class_raster_path
        lulc_raster_path (str): landcover code class used to further filter
            what should be processed
        valid_lulc_code_list (list): list of landcover codes in
            lulc_raster_path that limit which pixels are processed.
        mask_vector_path (str): path to a vector used to mask raster stack.
        mask_vector_where_filter (str): where filter to indicate which
            feature to filter by.
        f (str): desired path to place clipped rasters from
            ``raster_path_list``, they will have the same basename.
        task_graph (TaskGraph): taskgraph object that can be used for
            scheduling.

    Return:
        (scaled_raster_path, task for that raster)
    """
    try:
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
                ('scale', scale_raster_path),
                ('class', class_raster_path)]:
            # this can happen because biomass uses value as its class, skip it
            if raster_id == 'class' and class_raster_path == value_raster_path:
                task_path_map['class'] = task_path_map['value']
                continue
            clip_raster_path = os.path.join(
                target_dir, f'clip_{os.path.basename(raster_path)}')
            clip_task = task_graph.add_task(
                func=_clip_raster,
                args=(raster_path, bounding_box, clip_raster_path),
                target_path_list=[clip_raster_path],
                task_name=f'clip {raster_path} to {clip_raster_path}')

            fill_raster_path = os.path.join(
                target_dir, f'fill_{os.path.basename(raster_path)}')
            fill_task = task_graph.add_task(
                func=fill_by_convolution,
                args=(
                    clip_raster_path, MAX_FILL_DIST_DEGREES,
                    fill_raster_path),
                target_path_list=[fill_raster_path],
                dependent_task_list=[clip_task],
                task_name=f'clip and fill {raster_path} to {fill_raster_path}')
            task_path_map[raster_id] = {
                'task': fill_task,
                'path': fill_raster_path}

        # the biomass zones are uniquely identified by their unique floating
        # point values. Jeff said this was okay.
        no_fill_scale_raster_path = os.path.join(
            target_dir, f'''no_fill_scale_{os.path.basename(
                task_path_map['value']['path'])}''')
        scale_task = task_graph.add_task(
            func=scale_value,
            args=(
                task_path_map['value']['path'],
                task_path_map['scale']['path'],
                task_path_map['class']['path'],
                lulc_raster_path,
                valid_lulc_code_list,
                mask_vector_path, mask_vector_where_filter,
                no_fill_scale_raster_path),
            target_path_list=[no_fill_scale_raster_path],
            dependent_task_list=[
                task_path_map[raster_id]['task']
                for raster_id in ['value', 'scale', 'class']],
            task_name=(
                f"scale {task_path_map['value']['path']} to "
                f"{no_fill_scale_raster_path}"))

        # fill the scale
        no_mask_fill_scale_raster_path = os.path.join(
            target_dir, f'no_mask_fill_scale_{os.path.basename(raster_path)}')
        fill_scale_task = task_graph.add_task(
            func=fill_by_convolution,
            args=(
                no_fill_scale_raster_path, MAX_FILL_DIST_DEGREES,
                no_mask_fill_scale_raster_path),
            target_path_list=[no_mask_fill_scale_raster_path],
            dependent_task_list=[scale_task],
            task_name=f'clip and fill {raster_path} to {fill_raster_path}')

        # mask the result to the feature
        scaled_value_raster_path = os.path.join(
            target_dir,
            f"scaled_{os.path.basename(task_path_map['value']['path'])}")
        mask_scale_task = task_graph.add_task(
            func=pygeoprocessing.mask_raster,
            args=(
                (no_mask_fill_scale_raster_path, 1), mask_vector_path,
                scaled_value_raster_path),
            kwargs={'where_clause': mask_vector_where_filter},
            dependent_task_list=[fill_scale_task],
            target_path_list=[scaled_value_raster_path],
            task_name=f'mask final result of {scaled_value_raster_path}')

        # clip and mask by convolution
        task_path_map[raster_id] = {
            'task': mask_scale_task,
            'path': scaled_value_raster_path}

        return (scaled_value_raster_path, mask_scale_task)
    except Exception:
        LOGGER.exception(
            f'error on clip fill scale {target_dir} {value_raster_path}')
        raise


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
        mask_vector_where_filter, target_scaled_value_raster_path):
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
            feature to filter by.dd
        target_scaled_value_raster_path (str): created by this function,
            values from biomass raster scaled by NPP in unique functional
            zones.

    Return:
        None
    """
    try:
        unique_class_vals = get_unique_raster_values(class_raster_path)
        LOGGER.info(f'scaling {value_raster_path}')

        # Align raster stack
        working_dir = os.path.dirname(target_scaled_value_raster_path)
        base_raster_path_list = [
            value_raster_path, lulc_raster_path, scale_raster_path,
            class_raster_path]
        aligned_raster_path_list = [
            os.path.join(working_dir, f'aligned_{os.path.basename(path)}')
            for path in base_raster_path_list]
        no_duplicates_raster_path_list = sorted(set(base_raster_path_list))
        no_duplicates_aligned_raster_path_list = sorted(set(
            aligned_raster_path_list))
        lulc_info = pygeoprocessing.get_raster_info(lulc_raster_path)
        LOGGER.info(f'align raster stack for {no_duplicates_raster_path_list}')
        pygeoprocessing.align_and_resize_raster_stack(
            no_duplicates_raster_path_list,
            no_duplicates_aligned_raster_path_list,
            ['near'] * len(no_duplicates_aligned_raster_path_list),
            lulc_info['pixel_size'],
            'intersection', vector_mask_options={
                'mask_vector_path': mask_vector_path,
                'mask_vector_where_filter': mask_vector_where_filter})

        # create output raster
        target_nodata = -1
        pygeoprocessing.new_raster_from_base(
            aligned_raster_path_list[0], target_scaled_value_raster_path,
            gdal.GDT_Float32, [target_nodata])
        target_scaled_value_raster = gdal.OpenEx(
            target_scaled_value_raster_path, gdal.OF_RASTER | gdal.GA_Update)
        target_scaled_biomass_band = (
            target_scaled_value_raster.GetRasterBand(1))

        # open these for fast reading
        base_raster = gdal.OpenEx(aligned_raster_path_list[0], gdal.OF_RASTER)
        base_band = base_raster.GetRasterBand(1)
        lulc_raster = gdal.OpenEx(aligned_raster_path_list[1], gdal.OF_RASTER)
        lulc_band = lulc_raster.GetRasterBand(1)
        scale_raster = gdal.OpenEx(aligned_raster_path_list[2], gdal.OF_RASTER)
        scale_band = scale_raster.GetRasterBand(1)
        class_raster = gdal.OpenEx(aligned_raster_path_list[3], gdal.OF_RASTER)
        class_band = class_raster.GetRasterBand(1)

        LOGGER.info(
            f'calculate the mean value per class for value '
            f'{aligned_raster_path_list[2]} class {aligned_raster_path_list[3]}')
        scale_sum = collections.defaultdict(float)
        scale_count = collections.defaultdict(int)
        for offset_dict in pygeoprocessing.iterblocks(
                (aligned_raster_path_list[0], 1), offset_only=True):
            scale_array = scale_band.ReadAsArray(**offset_dict)
            class_array = class_band.ReadAsArray(**offset_dict)
            lulc_array = lulc_band.ReadAsArray(**offset_dict).astype(numpy.int32)
            lulc_mask = numpy.in1d(
                lulc_array, valid_lulc_code_list).reshape(lulc_array.shape)
            for class_val in unique_class_vals:
                valid_class_mask = numpy.isclose(class_array, class_val)
                valid_mask = valid_class_mask & lulc_mask
                scale_count[class_val] += numpy.count_nonzero(valid_mask)
                scale_sum[class_val] += numpy.sum(scale_array[valid_mask])

        # adjust the biomass value by the npp mean
        LOGGER.info(
            f'adjust value by mean for value '
            f'{aligned_raster_path_list[0]} class {aligned_raster_path_list[3]}')
        for offset_dict in pygeoprocessing.iterblocks(
                (aligned_raster_path_list[0], 1), offset_only=True):
            scale_array = scale_band.ReadAsArray(**offset_dict)
            base_array = base_band.ReadAsArray(**offset_dict)
            class_array = class_band.ReadAsArray(**offset_dict)
            lulc_array = lulc_band.ReadAsArray(**offset_dict).astype(numpy.int32)
            target_scaled_base_array = numpy.empty(scale_array.shape)
            target_scaled_base_array[:] = target_nodata
            lulc_mask = numpy.in1d(
                lulc_array, valid_lulc_code_list).reshape(lulc_array.shape)
            for class_val in unique_class_vals:
                if scale_count[class_val] == 0:
                    continue
                npp_mean = (
                    scale_sum[class_val] / scale_count[class_val])
                if numpy.isclose(npp_mean, 0.0):
                    npp_mean = 1.0
                valid_class_mask = numpy.isclose(class_array, class_val)
                valid_mask = valid_class_mask & lulc_mask & numpy.isfinite(
                    scale_array)
                try:
                    target_scaled_base_array[valid_mask] = (
                        base_array[valid_mask] * scale_array[valid_mask] /
                        npp_mean)
                except:
                    LOGGER.exception(
                        f'invalid value in divide: {npp_mean}\n'
                        f'scale_sum[class_val]: {scale_sum[class_val]}\n'
                        f'scale_array[valid_mask]: {scale_array[valid_mask]}\n'
                        f'on this raster: {target_scaled_value_raster_path}')
                    raise
            target_scaled_biomass_band.WriteArray(
                target_scaled_base_array,
                xoff=offset_dict['xoff'], yoff=offset_dict['yoff'])
    except Exception:
        LOGGER.exception(f'error on scale {target_scaled_base_array}')
        raise


def stitch_worker(work_queue, target_global_raster_path):
    """Stitch base, a smaller raster, into target, a global one."""
    try:
        LOGGER.info(f'starting up stitching for {target_global_raster_path}')
        global_raster = gdal.OpenEx(
            target_global_raster_path, gdal.OF_RASTER | gdal.GA_Update)
        global_band = global_raster.GetRasterBand(1)
        global_info = pygeoprocessing.get_raster_info(target_global_raster_path)
        global_inv_gt = gdal.InvGeoTransform(global_info['geotransform'])
        global_nodata = global_info['nodata'][0]
        n_cols, n_rows = global_info['raster_size']

        while True:
            try:
                payload = work_queue.get(timeout=60)
            except queue.Empty:
                LOGGER.info(
                    f'work queue {work_queue} empty on {target_global_raster_path}, '
                    'waiting for more')
                continue
            LOGGER.debug(f'stitching: got payload {payload}')
            if payload == 'STOP':
                LOGGER.debug(
                    f'got stop, stopping on {target_global_raster_path}')
                break
            scenario_id, base_raster_path = payload
            base_info = pygeoprocessing.get_raster_info(base_raster_path)
            base_nodata = base_info['nodata'][0]
            base_gt = base_info['geotransform']
            global_xoff, global_yoff = gdal.ApplyGeoTransform(
                global_inv_gt, base_gt[0], base_gt[3])
            for offset_dict, base_array in pygeoprocessing.iterblocks(
                    (base_raster_path, 1)):
                xoff = int(global_xoff)+offset_dict['xoff']
                if xoff >= n_cols:
                    LOGGER.warn(f'xoff >= n_cols ({xoff} >= {n_cols}) (global_xoff: {global_xoff}) for {base_raster_path} with gt {base_gt} {offset_dict}')
                    continue
                yoff = int(global_yoff)+offset_dict['yoff']
                if yoff >= n_rows:
                    continue
                win_xsize = offset_dict['win_xsize']
                win_ysize = offset_dict['win_ysize']
                if xoff+win_xsize > n_cols:
                    LOGGER.debug(f'xoff+win_xsize > n_cols: {xoff}+{win_xsize} > {n_cols} {target_global_raster_path}')
                    win_xsize += n_cols - (xoff+win_xsize)
                    LOGGER.debug(f'new win_xsize {win_xsize}')

                if yoff+win_ysize > n_rows:
                    LOGGER.debug(f'yoff+win_ysize > n_rows: {yoff}+{win_ysize} > {n_rows} {target_global_raster_path}')
                    win_ysize += n_rows - (yoff+win_ysize)
                    LOGGER.debug(f'new win_ysize {win_ysize}')
                # change the size of the array if needed
                base_array = base_array[0:win_ysize, 0:win_xsize]
                global_array = global_band.ReadAsArray(
                    xoff=xoff, yoff=yoff,
                    win_xsize=win_xsize, win_ysize=win_ysize)
                base_array[numpy.isclose(base_array, base_nodata)] = global_nodata
                valid_mask = numpy.isclose(global_array, global_nodata)
                global_array[valid_mask] = base_array[valid_mask]

                global_band.WriteArray(
                    global_array, xoff=xoff, yoff=yoff)

            LOGGER.debug(f'flush cache {payload}')
            global_band.FlushCache()
            LOGGER.debug(f'done stitching {payload}')

        global_band = None
        global_raster = None
        LOGGER.info(
            f'all done stitching, building overviews '
            f'for {target_global_raster_path}')
        ecoshard.build_overviews(
            target_global_raster_path, interpolation_method='average')
    except Exception:
        LOGGER.exception(
            f'exception on stitching {target_global_raster_path}')
        raise


def main():
    """Entry point."""
    try:
        for dir_path in [WORKSPACE_DIR, DATA_DIR, COUNTRY_WORKSPACE_DIR]:
            try:
                os.makedirs(dir_path)
            except OSError:
                pass
        task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, N_CPUS, 60)
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

        manager = multiprocessing.Manager()
        worker_queue_list = []
        for (scenario_id, value_raster_path, class_raster_path,
             valid_lulc_code_list) in [
                ('annual_biomass', ANNUAL_BIOMASS_RASTER_PATH,
                 ANNUAL_BIOMASS_RASTER_PATH, FORESTRY_VALID_LULC_LIST,),
                # ('plt_an_bio_proj', PLT_AN_BIO_PROJ_RASTER_PATH,
                #  PLT_AN_BIO_PROJ_RASTER_PATH, FORESTRY_VALID_LULC_LIST,),
                ('current_meat_prod', CURRENT_MEAT_PROD_RASTER_PATH,
                 GRAZING_ZONE_RASTER_PATH, GRAZING_VALID_LULC_LIST,),
                ('potential_meat_prod', POTENTIAL_MEAT_PROD_RASTER_PATH,
                 GRAZING_ZONE_RASTER_PATH, GRAZING_VALID_LULC_LIST,),
                ('potential_methane_prod', POTENTIAL_METHANE_PROD_RASTER_PATH,
                 GRAZING_ZONE_RASTER_PATH, GRAZING_VALID_LULC_LIST,),
                ('current_methane_prod', CURRENT_METHANE_PROD_RASTER_PATH,
                 GRAZING_ZONE_RASTER_PATH, GRAZING_VALID_LULC_LIST,),
                ]:
            global_stitch_raster_path = os.path.join(
                WORKSPACE_DIR, f'global_{os.path.basename(value_raster_path)}')
            value_info = pygeoprocessing.get_raster_info(value_raster_path)
            pygeoprocessing.new_raster_from_base(
                ESA_LULC_RASTER_PATH, global_stitch_raster_path,
                value_info['datatype'], [-9999])

            work_queue = manager.Queue()

            stitch_worker_process = multiprocessing.Process(
                target=stitch_worker,
                args=(work_queue, global_stitch_raster_path))
            stitch_worker_process.start()
            worker_queue_list.append((stitch_worker_process, work_queue))

            # create global stitch raster
            for country_feature in country_layer:
                country_iso = country_feature.GetField('ISO3')
                # skip Antarctica and anything so small it's not named
                if country_iso in [None, 'ATA']:
                    continue

                LOGGER.info(f'processing {country_iso}')
                country_workspace = os.path.join(
                    COUNTRY_WORKSPACE_DIR, f'{country_iso}_{scenario_id}')
                try:
                    os.makedirs(country_workspace)
                except OSError:
                    pass

                scaled_raster_path, scaled_task = clip_fill_scale(
                    value_raster_path, class_raster_path, NPP_RASTER_PATH,
                    ESA_LULC_RASTER_PATH, valid_lulc_code_list,
                    COUNTRY_VECTOR_PATH, f"iso3='{country_iso}'",
                    country_workspace, task_graph)

                task_graph.add_task(
                    func=work_queue.put,
                    args=((scenario_id, scaled_raster_path),),
                    dependent_task_list=[scaled_task],
                    transient_run=True,
                    task_name=(f'''stitch callback {
                        scaled_raster_path} into {global_stitch_raster_path}'''))

        LOGGER.info(
            'done with everything, waiting for task graph to join')
        task_graph.join()
        task_graph.close()
        LOGGER.info(
            'done scheduling, now waiting for stitch workers to stop')
        for stitch_worker_process, work_queue in worker_queue_list:
            work_queue.put('STOP')
            LOGGER.info(
                f'stopping stitch work queue {work_queue}')
            stitch_worker_process.join()
        LOGGER.info('all stitch workers joined')

    except Exception:
        LOGGER.exception('something bad happened in the the main scheduler')
        raise


def _work_callback(work_queue, scaled_raster_path):
    """Pass raster to work queue callback."""
    work_queue.put(scaled_raster_path)


if __name__ == '__main__':
    main()
