"""
Log-ruimte wrapper rond een sklearn-achtige regressor.

Staat bewust in een eigen module: het getrainde model wordt gepickled, en een
klasse die in `train_surrogate.py` zelf gedefinieerd staat wordt vastgelegd als
`__main__.LogTargetModel` zodra dat script direct gedraaid wordt. Elk ander
script (optimize_layout.py, evaluate.py) kan de pickle dan niet meer laden.
"""

import numpy as np


class LogTargetModel:
    """
    Wrapt een regressor die op log(waiterDist) traint maar pixels moet teruggeven.

    waiterDist loopt van ~253k tot ~1,3M px; in log-ruimte zijn de fouten veel
    gelijkmatiger verdeeld. De optimizer roept alleen `.predict()` aan en
    verwacht pixels, dus de terugtransformatie zit hier.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, np.log(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.exp(self.model.predict(X))

    @property
    def feature_importances_(self):
        return getattr(self.model, "feature_importances_", None)
