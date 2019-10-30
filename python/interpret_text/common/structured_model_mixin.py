from interpret_community.common.chained_identity import ChainedIdentity


class PureStructuredModelMixin(ChainedIdentity):
    """The base PureStructuredModelMixin API for explainers used on specific models.

    :param model: The grey box model to explain.
    :type model: A grey box model.
    """

    def __init__(self, model=None, **kwargs):
        """Initialize the PureStructuredModelExplainer.

        :param model: The white box model to explain.
        :type model: A white box model.
        """
        super(PureStructuredModelMixin, self).__init__(**kwargs)
        self._logger.debug('Initializing PureStructuredModelMixin')
        self.model = model