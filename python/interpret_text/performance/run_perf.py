import os

azure = False

#Do NGCD
if azure:
    from azureml.core import Workspace
    ws = Workspace.create(name='myworkspace',
                subscription_id='15ae9cb6-95c1-483d-a0e3-b1a1a3b06324',
                resource_group='nlpinterpret',
                create_resource_group=False,
                location='eastus2'
                )

    experiment = Experiment(ws, "MyExperiment")

    from azureml.core import ScriptRunConfig

    # run a trial from the train.py code in your current directory
    config = ScriptRunConfig(source_directory='.', script='scripts/msra_run.py',
        run_config=RunConfiguration())
    run = experiment.submit(config)

    # get the url to view the progress of the experiment and then wait
    # until the trial is complete
    print(run.get_portal_url())
    run.wait_for_completion()
else:
    os.system('python python/interpret_text/performance/scripts/msra_run.py')
