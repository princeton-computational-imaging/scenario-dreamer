import hydra
import pickle
import random
from tqdm import tqdm
from datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
from cfgs.config import CONFIG_PATH
import multiprocessing as mp
from pathlib import Path
from omegaconf import OmegaConf

# ───────────────────────────────────────────────────────────
#  helper so the Pool can pickle it
# ───────────────────────────────────────────────────────────
def _work_one_chunk(idx, cfg_dict):
    cfg = OmegaConf.create(cfg_dict)
    cfg.preprocess_waymo.chunk_idx = idx          # set my own chunk
    _run_one_cfg(cfg)


# ───────────────────────────────────────────────────────────
#  the old body → turned into a function we can reuse
# ───────────────────────────────────────────────────────────
def _run_one_cfg(cfg):
    random.seed(42)

    cfg.dataset_root = cfg.scratch_root
    cfg.dataset.waymo.preprocess = False
    dset = WaymoDatasetAutoEncoder(cfg, split_name=cfg.preprocess_waymo.mode)

    start = cfg.preprocess_waymo.chunk_idx * cfg.preprocess_waymo.chunk_size
    end   = start + cfg.preprocess_waymo.chunk_size
    chunk = [i for i in range(start, min(end, len(dset.files)))]
    if not chunk:
        return

    for idx in tqdm(chunk, position=0, leave=False):
        with open(dset.files[idx], "rb") as f:
            data = pickle.load(f)
        dset.get_data(data, idx)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):

    # case 1 — behave exactly like before (single-chunk run)
    if cfg.preprocess_waymo.chunk_idx >= 0:
        _run_one_cfg(cfg)
        print("Done!")
        return

    # case 2 — chunk_idx == -1  ⇒  run *all* chunks in parallel
    if cfg.preprocess_waymo.mode == 'train':
        total_chunks = 10
    else:
        total_chunks = 1

    n_workers = min(
        cfg.preprocess_waymo.get("num_workers", mp.cpu_count()),
        total_chunks
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f"[preprocess-waymo]  Launching {total_chunks} chunks on {n_workers} workers …")
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(
            _work_one_chunk,
            [(i, cfg_dict) for i in range(total_chunks)]
        )
    print("Done!")

main()