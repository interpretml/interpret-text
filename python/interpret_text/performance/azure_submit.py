# # ---------------------------------------------------------
# # Copyright (c) Microsoft Corporation. All rights reserved.
# # ---------------------------------------------------------

# import os
# import sys
# import logging

# from azureml.core import Workspace, Experiment
# from azureml.core.runconfig import RunConfiguration, DEFAULT_CPU_IMAGE
# from azureml.core.conda_dependencies import CondaDependencies

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
# from utilities.helpers.resource_provisioning_helpers import get_shared_resource_group, get_shared_workspace  # noqa
# from utilities.helpers import time_helpers  # noqa
# from utilities import constants  # noqa

# test_logger = logging.getLogger(__name__)
# test_logger.setLevel(logging.INFO)


# def experiment_run_setup(name, compute_target, local_queuing=False, docker=True):
#     """Creates an experiment using either the local workspace or using configuration from environment variables.
#     :param name: The name of the experiment
#     :type name: str
#     :param compute_target: The name of the compute target
#     :type compute_target: str
#     :param local_queuing: Whether the experiment is being queued locally
#     :type local_queuing: bool
#     :param docker: Whether to use docker
#     :type docker: bool
#     :returns: experiment, run config
#     """
#     # Enable developers to override with a local configuration (see scenarios/conftest.py)
#     subscription_id = os.getenv('SUBSCRIPTION_ID', default='15ae9cb6-95c1-483d-a0e3-b1a1a3b06324')
#     resource_group = os.getenv('RESOURCE_GROUP', default='abishkarg')
#     workspace_name = os.getenv('WORKSPACE_NAME', default='abishkarw')
#     workspace_region = os.getenv('WORKSPACE_REGION', default='eastus2')

#     common_packages = [
#         'psutil', 'hdbscan', 'lightgbm', 'memory_profiler', 'lime', 'matplotlib', 'xgboost'
#     ]

#     azureml_pip_packages = [
#         'azureml-defaults', 'azureml-contrib-interpret', 'azureml-core', 'azureml-telemetry',
#         'azureml-explain-model'
#     ]

#     conda_packages = [
#         'numpy', 'scipy', 'pandas', 'scikit-learn', 'pytorch-cpu', 'tensorflow'
#     ]

#     if local_queuing:
#         versioned_pip_packages = [package + '<0.1.50' for package in azureml_pip_packages]
#         pip_packages = versioned_pip_packages + common_packages
#         index_url = 'https://azuremlsdktestpypi.azureedge.net/AzureML-Explain-Model-Gated/4403235/'
#         conda_dependencies = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages,
#                                                       pin_sdk_version=False, pip_indexurl=index_url)
#     else:
#         pip_packages = azureml_pip_packages + common_packages
#         conda_dependencies = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)

#     conda_dependencies.add_channel('pytorch')
#     conda_dependencies.add_channel('conda-forge')

#     # create a new runconfig object
#     run_config = RunConfiguration(conda_dependencies=conda_dependencies)

#     # signal that you want to use AmlCompute to execute script.
#     run_config.target = compute_target

#     # AmlCompute will be created in the same region as workspace
#     # Set vm size for AmlCompute
#     run_config.amlcompute.vm_size = 'STANDARD_D2_V2'

#     # enable Docker
#     run_config.environment.docker.enabled = docker

#     # set Docker base image to the default CPU-based image
#     run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE

#     # use conda_dependencies.yml to create a conda environment in the Docker image for execution
#     run_config.environment.python.user_managed_dependencies = not docker

#     # specify CondaDependencies obj
#     run_config.environment.python.conda_dependencies = conda_dependencies

#     ws = None
#     try:
#         if local_queuing:
#             test_logger.info('Trying to load developer workspace.')
#             ws = Workspace.from_config()
#         else:
#             test_logger.info('Trying default workspace.')
#             from azureml.core.authentication import AzureCliAuthentication
#             auth = AzureCliAuthentication()
#             ws = Workspace(subscription_id=subscription_id, resource_group=resource_group,
#                            workspace_name=workspace_name, _location=workspace_region, auth=auth)
#             ws.write_config()
#     except Exception as exception:
#         test_logger.info('Failed to load workspace. Exception {}'.format(exception))
#         test_logger.info('Workspace not accessible. Using shared workspace instead')
#         shared_resource_group = get_shared_resource_group(time_helpers._get_next_rg_name())
#         perf_prefix = 'perf'
#         ws = get_shared_workspace(shared_resource_group, perf_prefix,
#                                   '{}-{}'.format(perf_prefix, constants.default_location))

#     return Experiment(workspace=ws, name=name), run_config





from azureml.core import Workspace
ws = Workspace.create(name='myworkspace',
            subscription_id='15ae9cb6-95c1-483d-a0e3-b1a1a3b06324',
            resource_group='interprettext',
            create_resource_group=False,
            location='eastus2'
            )

experiment = Experiment(ws, "MyExperiment")


from azureml.core import ScriptRunConfig

# run a trial from the train.py code in your current directory
config = ScriptRunConfig(source_directory='.', script='script.py',
    run_config=RunConfiguration())
run = experiment.submit(config)

# get the url to view the progress of the experiment and then wait
# until the trial is complete
print(run.get_portal_url())
run.wait_for_completion()