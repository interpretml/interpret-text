# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from urllib import request
# import json
# import torch
# from interpret_text.msra.MSRAExplainer import MSRAExplainer

# device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
# text = "rare bird has more than enough charm to make it memorable."
# model = BertModel.from_pretrained("bert-base-uncased").to(device)
# interpreter_simple = MSRAExplainer(model_name = "BERT", input_text = text, device=device)
# explanation = interpreter_simple.explain_local(model=model)
# print(explanation.local_importance_values)

a = [i*2 for i in range(100000000)]
print(a)