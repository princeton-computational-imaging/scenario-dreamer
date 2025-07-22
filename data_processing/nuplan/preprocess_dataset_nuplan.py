import hydra
import random
from tqdm import tqdm
from datasets.nuplan.dataset_autoencoder_nuplan import NuplanDatasetAutoEncoder
from cfgs.config import CONFIG_PATH
import multiprocessing as mp
from pathlib import Path
from omegaconf import OmegaConf

# ───────────────────────────────────────────────────────────
#  helper so the Pool can pickle it
# ───────────────────────────────────────────────────────────
def _work_one_chunk(idx, cfg_dict):
    cfg = OmegaConf.create(cfg_dict)
    cfg.preprocess_nuplan.chunk_idx = idx          # set my own chunk
    _run_one_cfg(cfg)


# ───────────────────────────────────────────────────────────
#  the old body → turned into a function we can reuse
# ───────────────────────────────────────────────────────────
def _run_one_cfg(cfg):
    random.seed(42)

    cfg.dataset_root = cfg.scratch_root
    cfg.ae.dataset.preprocess = False
    dset = NuplanDatasetAutoEncoder(cfg.ae.dataset, split_name=cfg.preprocess_nuplan.mode)

    start = cfg.preprocess_nuplan.chunk_idx * cfg.preprocess_nuplan.chunk_size
    end   = start + cfg.preprocess_nuplan.chunk_size
    chunk = [i for i in range(start, min(end, len(dset.files)))]
    if not chunk:
        return

    for idx in tqdm(chunk, position=0, leave=False):
        dset.get(idx)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):

    # case 1 — behave exactly like before (single-chunk run)
    if cfg.preprocess_nuplan.chunk_idx >= 0:
        _run_one_cfg(cfg)
        print("Done!")
        return

    # case 2 — chunk_idx == -1  ⇒  run *all* chunks in parallel
    if cfg.preprocess_nuplan.mode == 'train':
        total_chunks = 10
    else:
        total_chunks = 1
    
    if cfg.preprocess_nuplan.mode == 'test':
        cfg.preprocess_nuplan.chunk_size = 67000
        

    n_workers = min(
        cfg.preprocess_nuplan.get("num_workers", mp.cpu_count()),
        total_chunks
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f"[preprocess-nuplan]  Launching {total_chunks} chunks on {n_workers} workers …")
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(
            _work_one_chunk,
            [(i, cfg_dict) for i in range(total_chunks)]
        )
    print("Done!")

main()