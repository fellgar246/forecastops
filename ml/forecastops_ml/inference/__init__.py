"""Batch inference over stored models.

Gradient boosting forecasts are ``GradientBoostingForecaster.predict``. That
path calls :func:`forecastops_ml.features.build_features` with the training
cutoff, then scores the fitted tree model.
"""
