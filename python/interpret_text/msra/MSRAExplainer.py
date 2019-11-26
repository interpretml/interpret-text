import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import json
import html
import matplotlib.pyplot as plt
from urllib import request
from IPython.core.display import display, HTML
from pytorch_pretrained_bert import BertTokenizer

from interpret_text.common.structured_model_mixin import PureStructuredModelMixin
from interpret_text.explanation.explanation import _create_local_explanation


class MSRAExplainer(PureStructuredModelMixin, nn.Module):
    """The MSRAExplainer for returning explanations for pytorch NN models.
    """

    def __init__(self, device):
        """ Initialize the MSRAExplainer
        :param device: A pytorch device
        :type device: torch.device
        """
        super(MSRAExplainer, self).__init__()
        self.device = device
        # Constant paramters for now, will modify based on the model later
        # Scale: The maximum size of sigma. The recommended value is 10 * Std[word_embedding_weight], where word_embedding_weight is the word embedding weight in the model interpreted. Larger scale will give more salient result, Default: 0.5.
        self.scale = 0.5
        # Rate: A hyper-parameter that balance the MLE Loss and Maximum Entropy Loss. Larger rate will result in larger information loss. Default: 0.1.
        self.rate = 0.1
        # Phi: A function whose input is x (element in the first parameter) and returns a hidden state (of type ``torch.FloatTensor``, of any shape
        self.Phi = None
        self.regular = None

        assert self.scale >= 0, "the value for scale cannot be less than zero"
        assert 1 >= self.rate >= 0, "the value for rate has to be between 0 and 1"

    def explain_local(self, model, embedded_input, regularization, dataset=None):
        """Explain the model by using MSRA's interpretor
        :param model: a pytorch model
        :type: torch.nn
        :param embedded_input: The preprocessed text
        :type embedded_input: numpy ndarray
        :param regularization: The regularization term
        :type regularization: numpy ndarray
        :param dataset: dataset used while training the model
        :type dataset: numpy ndarray
        :return: A model explanation object. It is guaranteed to be a LocalExplanation
        :rtype: DynamicLocalExplanation
        """
        self.input_embeddings = embedded_input
        self.input_size = self.input_embeddings.size(0)
        self.input_dimension = self.input_embeddings.size(1)
        self.ratio = nn.Parameter(torch.randn(self.input_size, 1), requires_grad=True)
        self.regular = regularization
        assert not isinstance(self.input_embeddings, type(None)), "input embeddings are required to generate explanation"

        if self.regular is not None:
            self.regular = nn.Parameter(torch.tensor(self.regular).to(self.input_embeddings), requires_grad=False)
            self.Phi = self.generate_Phi(model, 3)
        else:
            assert dataset is not None, "Dataset is required if explainer not initialized with regularization parameter"
            self.Phi = self.generate_Phi(model, 3)
            self.regular = self._calculate_regularization(dataset, self.Phi)

        # values below are arbitarily set for now
        self.optimize(iteration=5000, lr=0.01, show_progress=True)
        local_importance_values = self.get_sigma()
        self.local_importance_values = local_importance_values
        return _create_local_explanation(local_importance_values=np.array(local_importance_values), method="neural network", model_task="classification")

    def getRegularizationBERT(self, model, explain_layer=3):
        """
        :param model: A pytorch model
        :type model: torch.nn
        :param explain_layer: layer to explain
        :type explain_layer: int
        :return: the regularization value for BERT model
        :rtype: list[float]
        """
        no_tries = 0
        while True:
            try:
                data = request.urlopen("https://nlpbp.blob.core.windows.net/data/regular.json").read()
                break
            except:
                no_tries += 1
                if no_tries > 10:
                    raise Exception("Too many failed HTTP Requests")
                print("Request Failed, trying again...")
        regularization_bert = json.loads(data)
        self.Phi = self.generate_Phi(model, layer=explain_layer)
        return regularization_bert

    def calculate_regularization(self, sampled_x, model, device, explain_layer=3, reduced_axes=None):
        """ Calculate the variance of the state generated from the perturbed inputs that is used for Interpreter
        :param sampled_x: A list of sampled input embeddings $x$, each $x$ is of shape ``[length, dimension]``. All the $x$s can have different length, but should have the same dimension. Sampled number should be higher to get a good estimation.
        :type sampled_x: list[torch.Tensor]
        :param reduced_axes: The axes that is variable in Phi (e.g., the sentence length axis). We will reduce these axes by mean along them.
        :type reduced_axes: list[int]
        :param model: A pytorch model
        :type model: torch.model
        :param explain_layer: The layer that needs to be explained. Defaults to the last layer
        :type explain_layer: int
        :param device: A pytorch device
        :type device: torch.device
        :param Phi: A function whose input is x (element in the first parameter) and returns a hidden state (of type ``torch.FloatTensor``, of any shape
        :type Phi: function
        :return: The regularization term calculated
        :rtype: torch.FloatTensor
        """
        sample_num = len(sampled_x)
        sample_s = []
        self.Phi = self.generate_Phi(model, layer=explain_layer)
        for n in range(sample_num):
            x = sampled_x[n]
            if device is not None:
                x = x.to(device)
            s = self.Phi(x)
            if reduced_axes is not None:
                for axis in reduced_axes:
                    assert axis < len(s.shape)
                    s = s.mean(dim=axis, keepdim=True)
            sample_s.append(s.tolist())
        sample_s = np.array(sample_s)
        return np.std(sample_s, axis=0)

    def forward(self):
        """ Calculate loss:
            $L(sigma) = (||Phi(embed + epsilon) - Phi(embed)||_2^2)
            // (regularization^2) - rate * log(sigma)$
        :return: a scalar, the target loss.
        :rtype: torch.FloatTensor
        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        x = self.input_embeddings + 0.0  # S * D
        x_tilde = x + ratios * torch.randn(self.input_size, self.input_dimension).to(x.device) * self.scale  # S * D
        s = self.Phi(x)  # D or S * D
        s_tilde = self.Phi(x_tilde)
        loss = (s_tilde - s) ** 2
        if self.regular is not None:
            loss = torch.mean(loss / self.regular ** 2)
        else:
            loss = torch.mean(loss) / torch.mean(s ** 2)

        return loss - torch.mean(torch.log(ratios)) * self.rate

    def optimize(self, iteration=5000, lr=0.01, show_progress=False):
        """ Optimize the loss function
        :param iteration: Total optimizing iteration
        :type iteration: int
        :param lr: Learning rate
        :type lr: float
        :param show_progress: Whether to show the learn progress
        :type show_progress: bool
        :return: The regularization term calculated
        :rtype: torch.FloatTensor
        """
        minLoss = None
        state_dict = None
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        func = (lambda x: x) if not show_progress else tqdm
        for _ in func(range(iteration)):
            optimizer.zero_grad()
            loss = self()
            loss.backward(retain_graph=True)
            optimizer.step()
            if minLoss is None or minLoss > loss:
                state_dict = {k: self.state_dict()[k] + 0.0 for k in self.state_dict().keys()}
                minLoss = loss
        self.eval()
        self.load_state_dict(state_dict)

    def get_sigma(self):
        """ Calculate and return the sigma
        :return: Sigma values of shape ``[seqLen]``, the ``sigma``.
        :rtype: np.ndarray
        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        return ratios.detach().cpu().numpy()[:, 0] * self.scale

    def visualize(self, text, max_alpha=0.5):
        """ Currently a placeholder visualize function until python widget is in a working state
        :param text: sample text
        :type text: str
        :param max_alpha: max alpha value
        :type max_alpha: float
        """

        CLS_TOKEN = ["[CLS]"]
        SEP_TOKEN = ["[SEP]"]

        def html_escape(text):
            return html.escape(text)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        parsed_sentence = CLS_TOKEN + tokenizer.tokenize(text) + SEP_TOKEN

        self.plot_global_imp(parsed_sentence[1:-2], [.4 - i for i in self.local_importance_values[1:-2]], "positive" )

        max_alpha = 0.5
        highlighted_text = []
        for i, word in enumerate(parsed_sentence):
            # since this is a placeholder function, ignore the magic numbers below
            weight = .55 - (self.local_importance_values[i] * 2)

            # make it blue if weight positive
            if weight > 0:
                highlighted_text.append('<span style="background-color:rgba(135,206,250,' + str(abs(weight) / max_alpha) +
                                        ');">' + html_escape(word) + '</span>')
            # red if it's negative
            elif weight < 0:
                highlighted_text.append('<span style="background-color:rgba(250,0,0,' + str(abs(weight) / max_alpha) +
                                        ');">' + html_escape(word) + '</span>')
            else:
                highlighted_text.append(word)
        highlighted_text = highlighted_text[1:-2]
        highlighted_text = ' '.join(highlighted_text)
        display(HTML(highlighted_text))

    def generate_Phi(self, model, layer):
        """Generate the Phi/hidden state that needs to be interpreted
        /Generates a function that returns the new hidden state given a new perturbed input is passed
        :param model: a pytorch model
        :type: torch.nn
        :param layer: the layer to generate Phi for
        :type layer: int
        :return: The phi function
        :rtype: function
        """
        # Assuming below that model has only 12 layers, will change this later
        assert (
            1 <= layer <= 12
        ), "model only has 12 layers, while you want to access layer %d" % (layer)

        def Phi(x):
            x = x.unsqueeze(0)
            attention_mask = torch.ones(x.shape[:2]).to(x.device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # extract the 3rd layer
            model_list = model.encoder.layer[:layer]
            hidden_states = x
            for layer_module in model_list:
                hidden_states = layer_module(hidden_states, extended_attention_mask)
            return hidden_states[0]
        return Phi

    def plot_global_imp(self, top_words, top_importances, label_name):
        plt.figure(figsize=(8,4))
        plt.title("most important words for class label: " + str(label_name), fontsize=18)
        plt.bar(range(len(top_importances)), top_importances, 
                color="b", align="center")
        plt.xticks(range(len(top_importances)), top_words, rotation=60, fontsize=18)
        plt.show()
