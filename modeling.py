#modeling.py
import h2o
from h2o.automl import H2OAutoML
import pandas as pd

def run_h2o_automl(df, target, max_models=20, max_runtime_secs=300):
    # Start H2O (reuse cluster if already running)
    h2o.init(strict_version_check=False)
    hf = h2o.H2OFrame(df)
    x = [c for c in hf.columns if c != target]

    aml = H2OAutoML(max_models=max_models,
                   max_runtime_secs=max_runtime_secs,
                   seed=42,
                   verbosity="info")
    aml.train(x=x, y=target, training_frame=hf)
    print(f"🏆 Best model: {aml.leader.model_id}")
    return aml.leader