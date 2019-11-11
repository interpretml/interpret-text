import os

azure = False

if azure:
    from azureml.core import Workspace
    ws = Workspace.create(name='myworkspace',
                subscription_id='5f08d643-1910-4a38-a7c7-84a39d4f42e0',
                resource_group='interprettext',
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
    os.system('python scripts/msra_run.py')