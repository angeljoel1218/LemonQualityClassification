"""
compat_mpl.py
-------------
matplotlib 3.9 renombro `boxplot(labels=...)` a `tick_labels=` y en versiones recientes `labels` ya no existe.
Los notebooks y modulos del proyecto usan `labels=`; este parche lo traduce solo cuando hace falta, de modo que el
mismo codigo funcione con matplotlib >= 3.8 (requirements.txt) y con las versiones nuevas.
"""
import inspect

import matplotlib.axes as _axes

_orig = _axes.Axes.boxplot
if "labels" not in inspect.signature(_orig).parameters and "tick_labels" in inspect.signature(_orig).parameters:
    def _boxplot(self, x, *args, labels=None, **kwargs):
        if labels is not None:
            kwargs["tick_labels"] = labels
        return _orig(self, x, *args, **kwargs)
    _boxplot.__doc__ = _orig.__doc__
    _axes.Axes.boxplot = _boxplot
