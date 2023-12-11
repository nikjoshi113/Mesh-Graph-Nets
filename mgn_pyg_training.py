# %% [markdown]
# # PyG Implementation of MeshGraphNets

# %%
import json
import os
import time

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset

import MGN
from helpers import (Struct, PyJSON)
from MGN import (MeshGraphNet, NodeType, control_randomness,
                               train_mgn, get_stats)

# %% [markdown]
# # Preparing and Loading the Dataset

# %%
# load config file
with open("config/config_additve.json", "r") as f:
    # CONFIG = Struct(json.load(f))
    CONFIG = PyJSON(json.load(f))
f.close()

# control_randomness
control_randomness(seed=5)

# %%
# source dataset filename
CONFIG.dataset.settings.source_file = "{}_{}_{}traj_{}ts_vis.pt".format(
    str(CONFIG.model.settings.model_type),
    str(CONFIG.model.settings.model_type_prefix),
    str(CONFIG.dataset.specs.number_trajectories),
    str(CONFIG.dataset.specs.number_timesteps)
)
CONFIG.dataset.settings.target_folder_name = "{}_dataset-{}traj-{}ts".format(
    str(time.strftime("%y-%m-%d_%H-%M")),
    str(CONFIG.dataset.specs.number_trajectories),
    str(CONFIG.dataset.specs.number_timesteps)
)

# dataset file name
CONFIG.model.settings.name = "model-tr{}_te{}_shuff{}-nl{}_bs{}_hd{}_ep{}_wd{}_lr{}".format(
    str(CONFIG.model.data.trajectories_train),
    str(CONFIG.model.data.trajectories_test),
    str(1 if CONFIG.model.data.shuffle_trajectories else 0),
    str(CONFIG.model.architecture.num_layers),
    str(CONFIG.model.architecture.batch_size),
    str(CONFIG.model.architecture.hidden_dim),
    str(CONFIG.model.architecture.epochs),
    str(CONFIG.model.architecture.weight_decay),
    str(CONFIG.model.architecture.lr),
)

# set path
CONFIG.dataset.dir.root = str(Path().resolve().absolute())
CONFIG.dataset.dir.dataset = str(Path(Path(CONFIG.dataset.dir.root)/"datasets"))
CONFIG.dataset.dir.model = str(Path(Path(CONFIG.dataset.dir.root)/"models"/CONFIG.dataset.settings.target_folder_name))
CONFIG.dataset.dir.checkpoint = str(Path(Path(CONFIG.dataset.dir.model)/"checkpoint"))
CONFIG.dataset.dir.postprocess = str(Path(Path(CONFIG.dataset.dir.model)/"postprocess"))

# control randomness
control_randomness(CONFIG.model.data.seed)
if (int(torch.initial_seed()) != CONFIG.model.data.seed) or (int(np.random.get_state()[1][0]) != CONFIG.model.data.seed):
    print("initial seed does not match with pytorch and/or numpy!")


# %% [markdown]
# **Creating folder structure**

# %%
# creating folder structure
for folder in [
    CONFIG.dataset.dir.model,
    CONFIG.dataset.dir.checkpoint,
    CONFIG.dataset.dir.postprocess
]:
    os.makedirs(folder) if not os.path.exists(folder) else None

# save json
def save_json():
    file = "{}/config_{}.json".format(CONFIG.dataset.dir.model,CONFIG.dataset.settings.target_folder_name)
    os.remove(file) if os.path.exists(file) else None
    with open(file, "w") as f:
        json.dump(CONFIG.to_dict(), f, indent=2)
    f.close()
save_json()

# %% [markdown]
# **Loading a Pre-Processed Dataset**

# %%
# load dataset
DATASET = torch.load(os.path.join(
    CONFIG.dataset.dir.dataset, CONFIG.dataset.settings.source_file))


# # select traintest
# DATASET_TRAINTEST = DATASET[:(CONFIG.model.architecture.train_size+CONFIG.model.architecture.test_size)]
# print("shape of 1. trayectory: {}".format(DATASET[:1]))

# stats_list = get_stats(DATASET_TRAINTEST)


# %% [markdown]
# **Select device**

# %%
CONFIG.dataset.settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Selected device: {}".format(CONFIG.dataset.settings.device))

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# %% [markdown]
# # Training

# %%
test_losses, losses, feature_val_losses, best_model, best_test_loss, test_loader = train_mgn(
    DATASET, CONFIG.dataset.settings.device, CONFIG)

print("Min test set loss: {0}".format(min(test_losses)))
print("Minimum loss: {0}".format(min(losses)))
if (CONFIG.model.postprocessing.save_feature_val):
    print("Minimum temperature validation loss: {0}".format(
        min(feature_val_losses)))
save_json()

# %% [markdown]
# Let's visualize the results!

# %%
from MGN import save_plots
save_plots(CONFIG, losses, test_losses, feature_val_losses)



