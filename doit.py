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
import signal
import threading
import time

from osgeo import gdal
import ecoshard
import pygeoprocessing
import numpy
import scipy.ndimage
import taskgraph

gdal.SetCacheMax(2**26)

N_CPUS = multiprocessing.cpu_count()


def signal_catcher(info_string):
    for index, signal_type in enumerate([
            signal.SIGSEGV,
            signal.SIGBUS,
            signal.SIGCHLD,
            signal.SIGFPE,
            signal.SIGHUP,
            signal.SIGILL,
            signal.SIGINT,
            ]):
        try:
            def sig_handler(signum, frame):
                print(f"*** {info_string} signal index caught: {index} ")
            signal.signal(signal_type, sig_handler)
        except:
            LOGGER.exception(f'bad signal {index}')


logging.basicConfig(
    level=logging.DEBUG,
    filename='log.txt',
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.DEBUG)
logging.getLogger('pygeoprocessing').setLevel(logging.DEBUG)

ECOSHARD_ROOT = (
    'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/'
    'Data-20201210-2-timber-layer_md5_dbf4ef0857bbe7f8fafda800a9eae099.zip')

WORKSPACE_DIR = 'workspace'
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
COUNTRY_WORKSPACE_DIR = os.path.join(WORKSPACE_DIR, 'country_workspaces')

COUNTRY_VECTOR_URL = (
    'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/'
    'gtm_regions_md5_56c7618d8ec458fc13cf2e9cd692c640.gpkg')
COUNTRY_VECTOR_PATH = os.path.join(
    DATA_DIR, 'gtm_regions_md5_56c7618d8ec458fc13cf2e9cd692c640.gpkg')

ESA_LULC_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'supportingLayers', 'lulc.tif')
NPP_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'supportingLayers',
    'MOD17A3_Science_NPP_mean_00_15.tif')
GRAZING_ZONE_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'supportingLayers', 'aez.tif')
ANNUAL_BIOMASS_RASTER_URL = (
    'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/'
    'global_annual_biomass_per_ha_v3_md5_8abced970e9faa1c79cc0a13cf36ea63.tif')

ANNUAL_BIOMASS_RASTER_PATH = os.path.join(
    DATA_DIR, os.path.basename(ANNUAL_BIOMASS_RASTER_URL))
PLT_AN_BIO_PROJ_RASTER_PATH = os.path.join(
    DATA_DIR, 'Data', 'timberLayers', 'plt_an_bio_proj.tif')

PLT_AN_BIO_PROJ_CLASS_RASTER_PATH = os.path.join(
    WORKSPACE_DIR, 'derived_data', 'class_plt_an_bio_proj.tif')

MAX_FILL_DIST_DEGREES = 0.5

FORESTRY_VALID_LULC_LIST = [
    50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170]
GRAZING_VALID_LULC_LIST = [
    30, 40, 100, 110, 120, 121, 122, 130, 140, 150, 152, 153, 180, 200, 201,
    202, 203]


def _set_nodata_value(raster_path, nodata):
    """Set the nodata value of the raster to `nodata`."""
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER | gdal.GA_Update)
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(nodata)


def _fill_nodata_op(base, fill, nodata):
    result = numpy.copy(base)
    if nodata is not None:
        nodata_mask = numpy.isclose(base, nodata)
        result[nodata_mask] = fill[nodata_mask]
        non_nodata_mask = ~numpy.isclose(result, nodata)
    else:
        non_nodata_mask = numpy.ones(result.shape, dtype=bool)
    # zero out any negative values, this was to fix an issue where I was
    # getting negative values on methane rasters.
    result[non_nodata_mask & (result < 0.0)] = 0.0
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

        non_square_block = False
        signal_raster_path = base_raster_path
        if base_raster_info['block_size'][0] != base_raster_info['block_size'][1]:
            non_square_block = True
            square_block_path = os.path.join(
                target_dir, f'{base_raster_path}_square_block.tif')
            ecoshard.compress_raster(
                base_raster_path, square_block_path,
                compression_algorithm='LZW',
                compression_predictor=None)
            signal_raster_path = square_block_path

        LOGGER.info(f'convolve 2d on {signal_raster_path} {backfill_raster_path}')
        pygeoprocessing.convolve_2d(
            (signal_raster_path, 1), (kernel_raster_path, 1),
            backfill_raster_path, ignore_nodata_and_edges=True,
            mask_nodata=False, normalize_kernel=True,
            target_nodata=base_nodata,
            target_datatype=target_datatype,
            working_dir=target_dir)

        LOGGER.info(
            f'fill nodata of {signal_raster_path} to {backfill_raster_path}')
        pygeoprocessing.raster_calculator(
            [(signal_raster_path, 1), (backfill_raster_path, 1),
             (base_nodata, 'raw')], _fill_nodata_op, target_filled_raster_path,
            base_raster_info['datatype'], base_nodata)

        os.remove(kernel_raster_path)
        os.remove(backfill_raster_path)
        if non_square_block:
            os.remove(square_block_path)
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
            nan_scrub_raster_path, 'near', target_bb=bounding_box,
            gdal_warp_options=['warpMemoryLimit=2e20'])
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
            class in class_raster_path. Scale can be None in which case the
            final result should not be scaled.
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
        path_map = dict()
        mask_vector = gdal.OpenEx(mask_vector_path, gdal.OF_VECTOR)
        mask_layer = mask_vector.GetLayer()
        mask_layer.SetAttributeFilter(mask_vector_where_filter)
        mask_feature = next(iter(mask_layer))
        mask_geom = mask_feature.GetGeometryRef()
        envelope = mask_geom.GetEnvelope()
        # swizzle so we get xmin, ymin, xmax, ymax order
        bounding_box = [envelope[i] for i in [0, 2, 1, 3]]

        fill_task_list = []
        for raster_id, raster_path in [
                ('value', value_raster_path),
                ('scale', scale_raster_path),
                ('class', class_raster_path)]:
            # this can happen because biomass uses value as its class, skip it
            if raster_id == 'scale' and raster_path is None:
                path_map['scale'] = None
                continue

            if raster_id == 'class' and class_raster_path == value_raster_path:
                path_map['class'] = path_map['value']
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
            fill_task_list.append(fill_task)
            path_map[raster_id] = fill_raster_path

        # the biomass zones are uniquely identified by their unique floating
        # point values. Jeff said this was okay.
        no_fill_scale_raster_path = os.path.join(
            target_dir, f'''no_fill_scale_{os.path.basename(
                path_map['value'])}''')

        scale_task = task_graph.add_task(
            func=mask_and_scale_value,
            args=(
                path_map['value'],
                path_map['scale'],
                path_map['class'],
                lulc_raster_path,
                valid_lulc_code_list,
                mask_vector_path, mask_vector_where_filter,
                no_fill_scale_raster_path),
            target_path_list=[no_fill_scale_raster_path],
            dependent_task_list=fill_task_list,
            task_name=(
                f"scale {path_map['value']} to "
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
            f"scaled_{os.path.basename(path_map['value'])}")
        mask_scale_task = task_graph.add_task(
            func=pygeoprocessing.mask_raster,
            args=(
                (no_mask_fill_scale_raster_path, 1), mask_vector_path,
                scaled_value_raster_path),
            kwargs={'where_clause': mask_vector_where_filter},
            dependent_task_list=[fill_scale_task],
            target_path_list=[scaled_value_raster_path],
            task_name=f'mask final result of {scaled_value_raster_path}')

        return (scaled_value_raster_path, mask_scale_task)
    except Exception:
        LOGGER.exception(
            f'error on clip fill scale {target_dir} {value_raster_path}')
        raise


def get_unique_raster_values(raster_path):
    """Return list of unique raster values in ``raster_path``."""
    unique_vals = set()
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    last_time = time.time()
    for index, (_, array) in enumerate(
            pygeoprocessing.iterblocks((raster_path, 1))):
        if time.time()-last_time > 5.0:
            last_time = time.time()
            LOGGER.debug(
                f'unique raster values on block {index} there are '
                f'{len(unique_vals)} unique vals so far')
        if nodata is not None:
            valid_mask = ~numpy.isclose(array, nodata)
        else:
            valid_mask = slice(None)
        unique_vals |= set(
            numpy.unique(array[valid_mask]))
    clean_unique_vals = [x for x in unique_vals if numpy.isfinite(x)]
    return clean_unique_vals


def mask_and_scale_value(
        value_raster_path, scale_raster_path, class_raster_path,
        lulc_raster_path, valid_lulc_code_list, mask_vector_path,
        mask_vector_where_filter, target_scaled_value_raster_path):
    """Mask by valid lulc and scale if present.

    Scale biomass by mean NPP in unique zones in regions that are allowed
    by the landcover type.

    Args:
        value_raster_path (str): path to base value raster to scale.
        scale_raster_path (str): Raster to scale value by weighted normalized
            values. It's possible for this value to be None in which case
            no scaling occurs, only masking.
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
        LOGGER.info(
            f'scaling {value_raster_path}\n'
            f'unique class vals for {class_raster_path}: {unique_class_vals}')

        # Align raster stack
        working_dir = os.path.dirname(target_scaled_value_raster_path)
        base_raster_path_list = [
            x for x in (
                value_raster_path, lulc_raster_path, class_raster_path,
                scale_raster_path) if x is not None]
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
                'mask_vector_where_filter': mask_vector_where_filter},
            gdal_warp_options=['warpMemoryLimit=2e20'])

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
        class_raster = gdal.OpenEx(aligned_raster_path_list[2], gdal.OF_RASTER)
        class_band = class_raster.GetRasterBand(1)
        if scale_raster_path is not None:
            scale_raster = gdal.OpenEx(
                aligned_raster_path_list[3], gdal.OF_RASTER)
            scale_band = scale_raster.GetRasterBand(1)

            LOGGER.info(
                f'calculate the mean value per class for value '
                f'{aligned_raster_path_list[1]} class '
                f'{aligned_raster_path_list[2]}')
            scale_sum = collections.defaultdict(float)
            scale_count = collections.defaultdict(int)
            for offset_dict in pygeoprocessing.iterblocks(
                    (aligned_raster_path_list[0], 1), offset_only=True):
                scale_array = scale_band.ReadAsArray(**offset_dict)
                class_array = class_band.ReadAsArray(**offset_dict)
                lulc_array = lulc_band.ReadAsArray(
                    **offset_dict).astype(numpy.int32)
                lulc_mask = numpy.in1d(
                    lulc_array, valid_lulc_code_list).reshape(lulc_array.shape)
                for class_val in unique_class_vals:
                    valid_class_mask = numpy.isclose(class_array, class_val)
                    valid_mask = valid_class_mask & lulc_mask
                    scale_count[class_val] += numpy.count_nonzero(valid_mask)
                    scale_sum[class_val] += numpy.sum(scale_array[valid_mask])

        # adjust the biomass value by the npp mean or just mask if scale is 0
        LOGGER.info(
            f'mask and adjust value by mean for value '
            f'{aligned_raster_path_list[0]} class {aligned_raster_path_list[2]}')
        for offset_dict in pygeoprocessing.iterblocks(
                (aligned_raster_path_list[0], 1), offset_only=True):
            base_array = base_band.ReadAsArray(**offset_dict)
            class_array = class_band.ReadAsArray(**offset_dict)
            lulc_array = lulc_band.ReadAsArray(**offset_dict).astype(numpy.int32)
            target_scaled_base_array = numpy.empty(lulc_array.shape)
            target_scaled_base_array[:] = target_nodata
            lulc_mask = numpy.in1d(
                lulc_array, valid_lulc_code_list).reshape(lulc_array.shape)
            if scale_raster_path is not None:
                scale_array = scale_band.ReadAsArray(**offset_dict)
            for class_val in unique_class_vals:
                if scale_raster_path is None:
                    npp_mean = 1.0
                elif scale_count[class_val] == 0:
                    continue
                else:
                    npp_mean = (
                        scale_sum[class_val] / scale_count[class_val])
                    if numpy.isclose(npp_mean, 0.0):
                        npp_mean = 1.0
                valid_class_mask = numpy.isclose(class_array, class_val)
                valid_mask = valid_class_mask & lulc_mask
                if scale_raster_path is not None:
                    valid_mask = valid_mask & numpy.isfinite(scale_array)
                try:
                    target_scaled_base_array[valid_mask] = (
                        base_array[valid_mask])
                    if scale_raster_path is not None:
                        target_scaled_base_array[valid_mask] *= (
                            scale_array[valid_mask] / npp_mean)
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
        target_scaled_biomass_band.FlushCache()
        target_scaled_biomass_band = None
        target_scaled_value_raster = None
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
        task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)
        token_path = os.path.join(DATA_DIR, 'download.token')
        fetch_task = task_graph.add_task(
            func=ecoshard.download_and_unzip,
            args=(ECOSHARD_ROOT, DATA_DIR),
            kwargs={'target_token_path': token_path},
            target_path_list=[token_path],
            task_name=f'fetch {ECOSHARD_ROOT}')
        download_task_list = [fetch_task]
        for url, download_path in [
                (COUNTRY_VECTOR_URL, COUNTRY_VECTOR_PATH),
                (ANNUAL_BIOMASS_RASTER_URL, ANNUAL_BIOMASS_RASTER_PATH),
                ]:
            fetch_task = task_graph.add_task(
                func=ecoshard.download_url,
                args=(url, download_path),
                target_path_list=[download_path],
                task_name=f'fetch {url}')
            download_task_list.append(fetch_task)

        task_graph.join()

        nodata_npp_value_task = task_graph.add_task(
            func=_set_nodata_value,
            args=(NPP_RASTER_PATH, 65535),
            ignore_path_list=[NPP_RASTER_PATH],
            dependent_task_list=download_task_list,
            task_name=f'add 65536 nodata value to NPP')
        nodata_npp_value_task.join()

        nodata_biomass_value_task = task_graph.add_task(
            func=_set_nodata_value,
            args=(ANNUAL_BIOMASS_RASTER_PATH, 0),
            ignore_path_list=[ANNUAL_BIOMASS_RASTER_PATH],
            dependent_task_list=download_task_list,
            task_name=f'set 0 nodata value on biomass')
        nodata_biomass_value_task.join()

        country_vector = gdal.OpenEx(COUNTRY_VECTOR_PATH, gdal.OF_VECTOR)
        country_layer = country_vector.GetLayer()

        plt_ann_bio_class_task = task_graph.add_task(
            func=create_class_raster,
            args=(
                PLT_AN_BIO_PROJ_RASTER_PATH,
                PLT_AN_BIO_PROJ_CLASS_RASTER_PATH),
            target_path_list=[PLT_AN_BIO_PROJ_CLASS_RASTER_PATH],
            task_name='create class raster for PLT_AN_BIO')
        plt_ann_bio_class_task.join()

        manager = multiprocessing.Manager()
        worker_queue_list = []
        for (scenario_id, value_raster_path, class_raster_path,
             scale_raster_path, valid_lulc_code_list) in [
                ('annual_biomass', ANNUAL_BIOMASS_RASTER_PATH,
                 ANNUAL_BIOMASS_RASTER_PATH, NPP_RASTER_PATH,
                 FORESTRY_VALID_LULC_LIST,),
                ('plt_an_bio_proj', PLT_AN_BIO_PROJ_RASTER_PATH,
                 PLT_AN_BIO_PROJ_CLASS_RASTER_PATH,
                 NPP_RASTER_PATH, FORESTRY_VALID_LULC_LIST,),
                ]:
            global_stitch_raster_path = os.path.join(
                WORKSPACE_DIR, f'global_{os.path.basename(value_raster_path)}')
            value_info = pygeoprocessing.get_raster_info(value_raster_path)
            pygeoprocessing.new_raster_from_base(
                ESA_LULC_RASTER_PATH, global_stitch_raster_path,
                value_info['datatype'], [-9999])

            work_queue = manager.Queue()

            stitch_worker_process = threading.Thread(
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
                    value_raster_path, class_raster_path, scale_raster_path,
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
            LOGGER.info(
                f'sending stop to stitch work queue {work_queue}')
            work_queue.put('STOP')
        for stitch_worker_process, work_queue in worker_queue_list:
            LOGGER.info(f'joining process  {stitch_worker_process}')
            stitch_worker_process.join()
        LOGGER.info('all stitch workers joined')

    except Exception:
        LOGGER.exception('something bad happened in the the main scheduler')
        raise


def _work_callback(work_queue, scaled_raster_path):
    """Pass raster to work queue callback."""
    work_queue.put(scaled_raster_path)


def _map_op(base_array, class_values):
    """Map base array to index of class values."""
    result = numpy.zeros(base_array.shape, dtype=numpy.uint8)
    for index, class_value in enumerate(class_values):
        result[numpy.isclose(base_array, class_value)] = index
    return result


def create_class_raster(base_raster_path, target_class_raster_path):
    """Create an integer class raster from base."""
    unique_values = get_unique_raster_values(base_raster_path)
    LOGGER.debug(f'there are {len(unique_values)} unique values: {unique_values}')
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (unique_values, 'raw')], _map_op,
        target_class_raster_path, gdal.GDT_Byte, None)


if __name__ == '__main__':
    main()
