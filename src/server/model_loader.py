import joblib
import torch

from src.training_player_data.mininet import MiniNet

def load_model_and_preproc(model_path: str, preproc_path: str):
    preproc = joblib.load(preproc_path)

    # infer model input size from the preprocessor output
    try:
        input_size = len(preproc.get_feature_names_out())
    except Exception:
        # fallback: transform a 1-row dummy with correct columns is hard here,
        # so we fail loudly (better than silently wrong size)
        raise ValueError("Could not infer input size from preproc; sklearn may be too old or preproc is unexpected.")

    model = MiniNet(input_size=int(input_size))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, preproc