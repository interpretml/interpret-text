from azureml.core import Workspace
subscription_id = "15ae9cb6-95c1-483d-a0e3-b1a1a3b06324"
resource_group = "nlpinterpret"
workspace_name = "performanceBenchmarking"
workspace_region = "eastus2"

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
#     print("Exception happened: ", exception)

#Load Workspace
try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")


experiment_name = 'performance-benchmarking-remote-vm'

from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)


import os
script_folder = './vm-run'
os.makedirs(script_folder, exist_ok=True)



from azureml.core.compute import ComputeTarget, RemoteCompute
from azureml.core.compute_target import ComputeTargetException

username = "abishkar"
address = "40.71.179.131"

compute_target_name = 'perfBench'
# if you want to connect using SSH key instead of username/password you can provide parameters private_key_file and private_key_passphrase 
try:
    attached_dsvm_compute = RemoteCompute(workspace=ws, name=compute_target_name)
    print('found existing:', attached_dsvm_compute.name)
except ComputeTargetException:
    attach_config = RemoteCompute.attach_configuration(address=address,
                                                       ssh_port=22,
                                                       username=username,
                                                       private_key_file='C:/Users/abchhetr/.ssh/id_rsa')
    attached_dsvm_compute = ComputeTarget.attach(workspace=ws,
                                                 name=compute_target_name,
                                                 attach_configuration=attach_config)
    attached_dsvm_compute.wait_for_completion(show_output=True)

from azureml.core import ScriptRunConfig
from uuid import uuid4

src = ScriptRunConfig(source_directory=script_folder, 
                      script='../python/interpret_text/performance/scripts/test_run.py', 
                      # pass the dataset as a parameter to the training script
                    #   arguments=['--data-folder',  
                    #              dataset.as_named_input('diabetes').as_mount('/tmp/{}'.format(uuid4()))]
                     ) 

src.run_config.framework = "python"
src.run_config.target = attached_dsvm_compute.name


run = exp.submit(config=src)

from azureml.widgets import RunDetails
RunDetails(run).show()

run.wait_for_completion(show_output=True)