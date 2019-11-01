import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from interpret_text.common.structured_model_mixin import PureStructuredModelMixin
from interpret_community.common.base_explainer import LocalExplainer
from interpret_text.explanation.explanation import _create_local_explanation

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

class MSRAExplainer(PureStructuredModelMixin, nn.Module):
    """The MSRAExplainer for returning explanations for deep neural network models.

    :param model: The tree model to explain.
    :type model: bert, xlnet or pytorch NN model
    """

    def __init__(self, x, scale=0.5, rate=0.1, Phi=None, regularization=None, words=None):
        """ Initialize an interpreter class.
        :param input_words: The input sentence, used for visualizing.
        :type input_words: Array[String]
        :param input_embeddings: The input word embeddings. A FloatTensor of shape ``[length, dimension]``.
        "type input_embeddings: torch.FloatTensor
        :param model: A pytorch model
        :type model: torch.nn
        :param scale: The maximum size of sigma. The recommended value is 10 * Std[word_embedding_weight], where word_embedding_weight is the word embedding weight in the model interpreted. Larger scale will give more salient result, Default: 0.5.
        :type scale: float
        :param rate: A hyper-parameter that balance the MLE Loss and Maximum Entropy Loss. Larger rate will result in larger information loss. Default: 0.1.
        :type rate: float
        :param regularization: The regularization of the hidden state. Default: None
        :type regularization: np.ndarray
        """
        super(MSRAExplainer, self).__init__()
        self.s = x.size(0)
        self.d = x.size(1)
        self.ratio = nn.Parameter(torch.randn(self.s, 1), requires_grad=True)

        self.scale = scale
        self.rate = rate
        self.x = x
        self.Phi = Phi

        self.regular = regularization
        if self.regular is not None:
            self.regular = nn.Parameter(torch.tensor(self.regular).to(x), requires_grad=False)
        self.words = words
        if self.words is not None:
            assert self.s == len(
                words
            ), "the length of x should be of the same with the lengh of words"

    def explain_local(self,model, evaluation_examples, device, dataset=None):
        """Explain the model by using msra's interpretor

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: DatasetWrapper
        :param sampled_x: A list of sampled input embeddings $x$, each $x$ is of shape ``[length, dimension]``. All the $x$s can have different length, but should have the same dimension. Sampled number should be higher to get a good estimation.
        :type sampled_x: List[torch.Tensor]
        :return: A model explanation object. It is guaranteed to be a LocalExplanation
        """
        self.Phi = self._generate_Phi(model, layer=3)
        if self.regular is None:
            self.regular = self._calculate_regularization(dataset, self.Phi)
        self.optimize(iteration=5000, lr=0.01, show_progress=True)
        local_importance_values = self.get_sigma()
        self.local_importance_values = local_importance_values
        return _create_local_explanation(local_importance_values=np.array(local_importance_values), method="neural network", model_task="classification")

    def _generate_Phi(self, model, layer):
        """Generate the Phi/hidden state that needs to be interpreted

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to explain the model's output.
        :type evaluation_examples: DatasetWrapper
        :return: A model explanation object. It is guaranteed to be a LocalExplanation
        """
        assert (
            1 <= layer <= 12
        ), "model only have 12 layers, while you want to access layer %d" % (layer)

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

    def _calculate_regularization(self, sampled_x, Phi, reduced_axes=None, device=None):
        """ Calculate the variance that is used for Interpreter
        :param sampled_x: A list of sampled input embeddings $x$, each $x$ is of shape ``[length, dimension]``. All the $x$s can have different length, but should have the same dimension. Sampled number should be higher to get a good estimation.
        :type sampled_x: List[torch.Tensor]
        :param reduced_axes: The axes that is variable in Phi (e.g., the sentence length axis). We will reduce these axes by mean along them.
        :type reduced_axes: List[int]
        :return: torch.FloatTensor: The regularization term calculated
        """
        sample_num = len(sampled_x)
        sample_s = []
        for n in range(sample_num):
            x = sampled_x[n]
            if device is not None:
                x = x.to(device)
            s = Phi(x)
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
        Returns:
            torch.FloatTensor: a scalar, the target loss.
        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        x = self.x + 0.0  # S * D
        x_tilde = x + ratios * torch.randn(self.s, self.d).to(x.device) * self.scale  # S * D
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
        Args:
            iteration (int): Total optimizing iteration
            lr (float): Learning rate
            show_progress (bool): Whether to show the learn progress
        """
        minLoss = None
        state_dict = None
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        func = (lambda x: x) if not show_progress else tqdm
        for _ in func(range(iteration)):
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            optimizer.step()
            if minLoss is None or minLoss > loss:
                state_dict = {k: self.state_dict()[k] + 0.0 for k in self.state_dict().keys()}
                minLoss = loss
        self.eval()
        self.load_state_dict(state_dict)

    def get_sigma(self):
        """ Calculate and return the sigma
        Returns:
            np.ndarray: of shape ``[seqLen]``, the ``sigma``.
        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        return ratios.detach().cpu().numpy()[:, 0] * self.scale