import json
import torch
import logging
import time
from urllib import request
from azureml.core.run import Run
from interpret_text.performance.utils.memory import get_peak_memory
from pytorch_pretrained_bert import BertModel, BertTokenizer

from interpret_text.msra.MSRAExplainer import MSRAExplainer

logging.basicConfig(filename='benchmark.log')
test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)


azure_run = False

if azure_run:
    run = Run.get_context()
start = time.time()

test_memory = get_peak_memory()

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
text = "rare bird has more than enough charm to make it memorable."
model = BertModel.from_pretrained("bert-base-uncased").to(device)
interpreter_simple = MSRAExplainer(model_name = "BERT", input_text = text, device=device)
test_logger.info('Running explain model global for MSRA')
explanation = interpreter_simple.explain_local(model=model)
print(explanation.local_importance_values)

# Get elapsed execution time
end = time.time()
test_logger.info('elapsed time: ' + str(end - start))
peak_memory = get_peak_memory()
test_logger.info('peak memory usage: ' + str(peak_memory))
#log accuracy of scenario model (bert or logistic regression)

if azure_run:
    # log execution time and peak memory usage to the Azure Run
    run.log('execution time', end - start)
    run.log('peak memory usage', peak_memory)