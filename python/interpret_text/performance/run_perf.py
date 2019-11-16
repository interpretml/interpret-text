import os
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies


subscription_id = os.getenv("SUBSCRIPTION_ID", default="<my-subscription-id>")
resource_group = os.getenv("RESOURCE_GROUP", default="<my-resource-group>")
workspace_name = os.getenv("WORKSPACE_NAME", default="<my-workspace-name>")
workspace_region = os.getenv("WORKSPACE_REGION", default="eastus2")

#Create Workspace

# try:
#     from azureml.core.authentication import AzureCliAuthentication
#     auth = AzureCliAuthentication()

#     # Create the workspace using the specified parameters
#     ws = Workspace.create(name = workspace_name,
#                         subscription_id = subscription_id,
#                         resource_group = resource_group, 
#                         location = workspace_region,
#                         create_resource_group = True,
#                         auth=auth)
#     ws.get_details()
#     # write the details of the workspace to a configuration file to the notebook library
#     ws.write_config()
# except Exception as exception:
#     print("Exception occured: ", exception)

#Load Workspace
try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")

#Load Experiment
experiment_name = 'performance-benchmarking-remote-vm'
exp = Experiment(workspace=ws, name=experiment_name)

interpret_text_env = Environment("interpret_text")

interpret_text_env.docker.enabled = True
interpret_text_env.python.conda_dependencies = CondaDependencies.create(conda_packages=['python==3.6.8', 'pip>=19.1.1', 'ipykernel>=4.6.1', 'jupyter>=1.0.0', 'matplotlib>=2.2.2', 'numpy>=1.13.3', 'pandas>=0.24.2', 'pytest>=3.6.4', 'pytorch>=1.0.0', 'scipy>=1.0.0', 'tensorflow-gpu==1.14.0', 'h5py>=2.8.0', 'tensorflow-hub==0.5.0', 'py-xgboost<=0.80', 'dask[dataframe]==1.2.2', 'numba>=0.38.1', 'cudatoolkit==9.2'], pip_packages=['azureml-sdk[automl,notebooks,contrib,explain]==1.0.57', 'cached-property==1.5.1', 'papermill>=1.0.1', 'nteract-scrapbook>=0.2.1', 'pytorch-pretrained-bert>=0.6', 'tqdm==4.31.1', 'scikit-learn>=0.19.0,<=0.20.3', 'nltk>=3.4', 'interpret-community>=0.1.0.2', 'git+https://github.com/microsoft/nlp.git', 'interpret', 'shap>=0.20.0, <=0.29.3'])
interpret_text_env.register(workspace=ws)

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cpu_cluster_name = "cpu-cluster"
gpu_cluster_name = "gpu-cluster"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print("Found existing gpu-cluster")
except ComputeTargetException:
    print("GPU CLUSTER NOT FOUND")
    # print("Creating new cpu-cluster")
    
    # # Specify the configuration for the new cluster
    # compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_NC6",
    #                                                        min_nodes=0,
    #                                                        max_nodes=4)

    # # Create the cluster with the specified name and configuration
    # gpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    
    # # Wait for the cluster to complete, show the output log
    # gpu_cluster.wait_for_completion(show_output=True)


script_folder = './sample_script'
os.makedirs(script_folder, exist_ok=True)
script_path = "test_run.py"

src = ScriptRunConfig(source_directory=script_folder, 
                      script=script_path)

src.run_config.framework = "python"
src.run_config.target = gpu_cluster.name
# Set up environment
src.run_config.environment = interpret_text_env

run = exp.submit(config=src)

# from azureml.widgets import RunDetails
# RunDetails(run).show()

# #Don't have this to submit multiple experiments
# run.wait_for_completion(show_output=True)