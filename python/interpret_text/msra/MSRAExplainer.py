import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import html
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from pytorch_pretrained_bert import BertTokenizer

from interpret_text.common.structured_model_mixin import PureStructuredModelMixin
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_text.common.utils_msra import get_single_embedding, make_bert_embeddings


class MSRAExplainer(PureStructuredModelMixin, nn.Module):
    """The MSRAExplainer for returning explanations for pytorch NN models.
    """

    def __init__(self, model, train_dataset, device, target_layer=14, total_layers=12):
        """ Initialize the MSRAExplainer
        :param model: a pytorch model
        :type: torch.nn
        :param train_dataset: dataset used while training the model
        :type train_dataset: list
        :param device: A pytorch device
        :type device: torch.device
        :param target_layer: The target layer to explain. Default is 14, which is the classification layer.
        If set to -1, all layers will be explained
        :type target_layer: int
        :param total_layer: The total number of bert hidden layers
        :type total_layer: int
        """
        super(MSRAExplainer, self).__init__()
        self.device = device
        # Constant paramters for now, will modify based on the model later
        # Scale: The maximum size of sigma. The recommended value is 10 * Std[word_embedding_weight], where
        # word_embedding_weight is the word embedding weight in the model interpreted. Larger scale will give
        # more salient result, Default: 0.5.
        self.scale = 0.5
        # Rate: A hyper-parameter that balance the MLE Loss and Maximum Entropy Loss. Larger rate will result in
        # larger information loss. Default: 0.1.
        self.rate = 0.1
        # Phi: A function whose input is x (element in the first parameter) and returns a hidden state (of type
        # ``torch.FloatTensor``, of any shape
        self.Phi = None
        self.regular = None
        self.target_layer = target_layer
        self.total_layers = total_layers
        self.model = model
        self.train_dataset = train_dataset

        assert (
            self.train_dataset is not None
        ), "Training dataset is required. please pass in the data used to train\
            the model"
        assert self.model is not None, "You have to pass in a trained model."
        assert self.scale >= 0, "The value for scale cannot be less than zero"
        assert 1 >= self.rate >= 0, "The value for rate has to be between 0 and 1"
        assert type(self.target_layer) == int and (
            self.target_layer == -1 or 1 <= self.target_layer <= 14
        ), (
            "the\
            value of the target layer has to be an interger and either -1 (all bert layers), or 1 to 12 (specific bert\
            layer) or 13 (pooler layer), or 14 (final classification layer).\
            You want to access layer %d"
            % (self.target_layer)
        )

    def explain_local(self, text):
        """Explain the model by using MSRA's interpretor
        :param text: The text
        :type text: string
        :return: A model explanation object. It is guaranteed to be a LocalExplanation
        :rtype: DynamicLocalExplanation
        """
        assert text is not None, "input text is required to generate explanation"

        embedded_input = get_single_embedding(self.model, text, self.device)
        self.input_embeddings = embedded_input

        self.input_size = self.input_embeddings.size(0)
        self.input_dimension = self.input_embeddings.size(1)
        self.ratio = nn.Parameter(torch.randn(self.input_size, 1), requires_grad=True)
        self.input_embeddings.to(self.device)

        if self.regular is None:
            assert self.train_dataset is not None, "Training dataset is required"
            training_embeddings = make_bert_embeddings(
                self.train_dataset, self.model, self.device
            )
            regularization = self.calculate_regularization(
                training_embeddings, self.model
            ).tolist()
            self.regular = nn.Parameter(
                torch.tensor(regularization).to(self.input_embeddings),
                requires_grad=False,
            )
            self.Phi = self.generate_Phi(self.target_layer, self.total_layers)

        # values below are arbitarily set for now
        self.optimize(iteration=50, lr=0.01, show_progress=True)
        local_importance_values = self.get_sigma()
        self.local_importance_values = local_importance_values
        return _create_local_explanation(
            local_importance_values=np.array(local_importance_values),
            method="neural network",
            model_task="classification",
        )

    def calculate_regularization(self, sampled_x, model, reduced_axes=None):
        """ Calculate the variance of the state generated from the perturbed inputs that is used for Interpreter
        :param sampled_x: A list of sampled input embeddings $x$, each $x$ is of shape ``[length, dimension]``.
        All the $x$s can have different length, but should have the same dimension. Sampled number should be higher
        to get a good estimation.
        :type sampled_x: list[torch.Tensor]
        :param reduced_axes: The axes that is variable in Phi (e.g., the sentence length axis). We will reduce
        these axes by mean along them.
        :type reduced_axes: list[int]
        :param model: A pytorch model
        :type model: torch.model
        :param explain_layer: The layer that needs to be explained. Defaults to the last layer
        :type explain_layer: int
        :param device: A pytorch device
        :type device: torch.device
        :param Phi: A function whose input is x (element in the first parameter) and returns a hidden
        state (of type ``torch.FloatTensor``, of any shape
        :type Phi: function
        :return: The regularization term calculated
        :rtype: torch.FloatTensor
        """
        sample_num = len(sampled_x)
        sample_s = []
        self.Phi = self.generate_Phi(
            layer=self.target_layer, total_layers=self.total_layers
        )
        for n in range(sample_num):
            x = sampled_x[n]
            if self.device is not None:
                x = x.to(self.device)

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
        x = self.input_embeddings + 0.0
        x_tilde = (
            x
            + ratios
            * torch.randn(self.input_size, self.input_dimension).to(x.device)
            * self.scale
        )  # S * D
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
            if self.device is not None:
                self.to(self.device)
            loss = self()
            loss.backward(retain_graph=True)
            optimizer.step()
            if minLoss is None or minLoss > loss:
                state_dict = {
                    k: self.state_dict()[k] + 0.0 for k in self.state_dict().keys()
                }
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

    def generate_Phi(self, layer, total_layers):
        """Generate the Phi/hidden state that needs to be interpreted
        /Generates a function that returns the new hidden state given a new perturbed input is passed
        :param model: a pytorch model
        :type: torch.nn
        :param layer: the layer to generate Phi for
        :type layer: int
        :param total_layers: the total number of bert layers
        :return: The phi function
        :rtype: function
        """

        def Phi(x):
            x = x.unsqueeze(0)
            attention_mask = torch.ones(x.shape[:2]).to(x.device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            hidden_states = x
            if layer == 14:
                # explain the classification layer
                encoder = self.model.bert.encoder
                pooler = self.model.bert.pooler
                classifier = self.model.classifier
                return classifier(
                    pooler(encoder(hidden_states, attention_mask, False)[-1])
                )[0]

            else:
                # extract one of the bert layers
                model_list = self.model.bert.encoder.layer[:layer]
                for layer_module in model_list:
                    hidden_states = layer_module(hidden_states, extended_attention_mask)
                return hidden_states[0][layer]

        return Phi

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

        self.plot_global_imp(
            parsed_sentence[1:-2],
            [0.4 - i for i in self.local_importance_values[1:-2]],
            "positive",
        )

        max_alpha = 0.5
        highlighted_text = []
        for i, word in enumerate(parsed_sentence):
            # since this is a placeholder function, ignore the magic numbers below
            weight = 0.55 - (self.local_importance_values[i] * 2)

            # make it blue if weight positive
            if weight > 0:
                highlighted_text.append(
                    '<span style="background-color:rgba(135,206,250,'
                    + str(abs(weight) / max_alpha)
                    + ');">'
                    + html_escape(word)
                    + "</span>"
                )
            # red if it's negative
            elif weight < 0:
                highlighted_text.append(
                    '<span style="background-color:rgba(250,0,0,'
                    + str(abs(weight) / max_alpha)
                    + ');">'
                    + html_escape(word)
                    + "</span>"
                )
            else:
                highlighted_text.append(word)
        highlighted_text = highlighted_text[1:-2]
        highlighted_text = " ".join(highlighted_text)
        display(HTML(highlighted_text))

    def plot_global_imp(self, top_words, top_importances, label_name):
        plt.figure(figsize=(8, 4))
        plt.title(
            "most important words for class label: " + str(label_name), fontsize=18
        )
        plt.bar(range(len(top_importances)), top_importances, color="b", align="center")
        plt.xticks(range(len(top_importances)), top_words, rotation=60, fontsize=18)
        plt.show()
