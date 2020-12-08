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

import ecoshard
import pygeoprocessing
import taskgraph

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

ECOSHARD_ROOT = 'https://storage.googleapis.com/ecoshard-root/nci-gap-fill-npp-scaling/Data-20201207T190350Z-001_md5_63b09dd643d3e686a5d83a9e340c90d2.zip'
WORKSPACE_DIR = 'workspace'
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, DATA_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)
    token_path = os.path.join(DATA_DIR, 'download.token')
    task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(ECOSHARD_ROOT, DATA_DIR),
        kwargs={'target_token_path': token_path},
        target_path_list=[token_path],
        task_name=f'fetch {ECOSHARD_ROOT}')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
