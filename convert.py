import glob
import os

import numpy as np
from tqdm import tqdm
import shutil


def convert_opt_weights(model_folder, path, model_name):
    import torch

    print(f"Load the pre-trained pytorch weights of f{model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")


    bin_files = glob.glob(os.path.join(model_folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))
                
if __name__ == "__main__":
    #model_folder = "/home/tangexing/tgx/models/OPT-1.3B"
    #path = "/home/tangexing/tgx/models/OPT-1.3B-flex "
    #model_name = "OPT-1.3B"
    model_folder = "/home/tangexing/tgx/models/Llama-2-7b-hf"
    path = "/home/tangexing/tgx/models/Llama-2-7b-hf-offload"
    model_name = "Llama-2-7b-hf"
    convert_opt_weights(model_folder, path, model_name)