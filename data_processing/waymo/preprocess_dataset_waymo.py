import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import hydra
import random
from tqdm import tqdm
from datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
from datasets.waymo.dataset_ctrl_sim import CtRLSimDataset
from cfgs.config import CONFIG_PATH, NUM_WAYMO_TRAIN_SCENARIOS
import multiprocessing as mp
from omegaconf import OmegaConf

def _mp_init():
    # Ensure per-process thread caps
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

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
    if cfg.preprocess_waymo.stage == 'scenario_dreamer':
        cfg.ae.dataset.preprocess = False
        dset = WaymoDatasetAutoEncoder(cfg.ae.dataset, split_name=cfg.preprocess_waymo.mode)
    else:
        cfg.ctrl_sim.dataset.preprocess = False
        dset = CtRLSimDataset(cfg.ctrl_sim.dataset, split_name=cfg.preprocess_waymo.mode)

    start = cfg.preprocess_waymo.chunk_idx * cfg.preprocess_waymo.chunk_size
    end   = start + cfg.preprocess_waymo.chunk_size
    chunk = [i for i in range(start, min(end, len(dset.files)))]
    if not chunk:
        return

    for idx in tqdm(chunk, position=0, leave=False):
        d = dset.get(idx)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):

    # case 1 — behave exactly like before (single-chunk run)
    if cfg.preprocess_waymo.chunk_idx >= 0:
        _run_one_cfg(cfg)
        print("Done!")
        return

    # case 2 — chunk_idx == -1  ⇒  run *all* chunks in parallel
    if cfg.preprocess_waymo.mode == 'train':
        total_chunks = math.ceil(NUM_WAYMO_TRAIN_SCENARIOS / cfg.preprocess_waymo.chunk_size)
    else:
        if cfg.preprocess_waymo.stage == 'scenario_dreamer':
            total_chunks = 1 # chunk size is 50000
        else:
            total_chunks = 4 # chunk size is 12000
    
    if cfg.preprocess_waymo.mode == 'test':
        cfg.preprocess_waymo.chunk_size = 61000 # test have 10000 scenes resampled from same scenarios
    n_workers = min(
        cfg.preprocess_waymo.get("num_workers", mp.cpu_count()),
        total_chunks
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f"[preprocess-waymo]  Launching {total_chunks} chunks on {n_workers} workers …")
    with mp.Pool(processes=n_workers, initializer=_mp_init) as pool:
        pool.starmap(
            _work_one_chunk,
            [(i, cfg_dict) for i in range(total_chunks)]
        )
    print("Done!")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()