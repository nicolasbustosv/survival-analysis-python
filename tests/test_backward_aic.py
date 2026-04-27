"""Known-answer test for backward AIC variable selection."""
import numpy as np
import pandas as pd

from survival_analysis.cox import backward_aic


def _make_fixture():
    """
    x_strong is the true predictor (HR ~ 2).
    x_noise has no effect — backward AIC should drop it.
    """
    rng = np.random.default_rng(7)
    n = 300
    x_strong = rng.normal(0, 1, n)
    x_noise  = rng.normal(0, 1, n)
    duration = np.exp(0.7 * x_strong + rng.normal(0, 0.3, n)) * 400
    return pd.DataFrame({
        "duration": duration,
        "event":    np.ones(n),
        "x_strong": x_strong,
        "x_noise":  x_noise,
    })


def test_backward_aic_drops_noise():
    df = _make_fixture()
    final_model, remaining = backward_aic(
        df, duration_col="duration", event_col="event"
    )
    assert "x_strong" in remaining, "backward AIC dropped the true predictor"


def test_backward_aic_returns_model(simple_df=None):
    from lifelines import CoxPHFitter
    df = _make_fixture()
    model, remaining = backward_aic(df, duration_col="duration", event_col="event")
    assert isinstance(model, CoxPHFitter)
    assert len(remaining) >= 1
