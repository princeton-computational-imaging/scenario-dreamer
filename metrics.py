import os 
from tqdm import tqdm 
import pickle 
import gzip
from utils.metrics_helpers import convert_data_to_unified_format, compute_lane_metrics, compute_agent_metrics

class Metrics():

    def __init__(self, cfg):
        self.cfg = cfg 
        self.dataset = cfg.dataset_name
        self.samples_path = cfg.eval.metrics.samples_path
        self.eval_set = cfg.eval.metrics.eval_set


    def compute_metrics(self):
        """Compute metrics given the generated samples and the ground truth samples."""
        sample_paths = [os.path.join(self.samples_path, file) for file in os.listdir(self.samples_path)]

        with open(self.eval_set, 'rb') as f:
            gt_sample_filenames = pickle.load(f)['files']

        if self.cfg.dataset_name == 'nuplan':
            gt_sample_ids = [os.path.splitext(file)[0] for file in gt_sample_filenames]
        
        num_samples = len(sample_paths)
        num_gt_samples = len(gt_sample_filenames)
        assert num_samples == num_gt_samples, "Number of samples and ground truth samples do not match."

        print("Number of evaluated samples (real/generated): ", num_samples)
        samples = []
        gt_samples = []
        print("Converting samples to unified format for metrics computation...")
        for i in tqdm(range(num_samples)):
            with open(sample_paths[i], 'rb') as f:
                data = pickle.load(f)
            sample = convert_data_to_unified_format(data, dataset_name=f"{self.cfg.dataset_name}")

            if self.cfg.dataset_name == 'waymo':
                # agent and lane gt data are loaded from the preprocessed scenario dreamer waymo data
                with open(os.path.join(self.cfg.eval.metrics.gt_test_dir, gt_sample_filenames[i]), 'rb') as f:
                    gt_data = pickle.load(f)
            else:
                # the gt agent data comes from the preprocessed scenario dreamer nuplan data
                sample_id = gt_sample_ids[i]
                with open(os.path.join(self.cfg.eval.metrics.gt_agent_test_dir, f'{sample_id}_0.pkl'), 'rb') as f:
                    gt_agent_data = pickle.load(f)

                # As the lane graph is preprocessed slightly differently between SLEDGE and scenario dreamer,
                # for fairest comparison with SLEDGE we process the gt lane graphs following the SLEDGE preprocessing scheme (this requires 
                # loading from the SLEDGE preprocessed nuplan data)
                # We could preprocess the gt lane graphs using the scenario dreamer preprocessing scheme,
                # but then we wouldn't know if performance improvement compared to SLEDGE is attributed to the GT lane graph preprocessing
                # being more aligned with scenario dreamer. 
                # In practice, we find both preprocessing schemes yield very similar performance.
                with gzip.open(os.path.join(self.cfg.eval.metrics.gt_lane_test_dir, gt_sample_filenames[i]), 'rb') as f:
                    gt_lane_data = pickle.load(f)
                
                gt_data = gt_lane_data 
                # add agent data to the gt lane data
                gt_data['agent_states'] = gt_agent_data['agent_states']
                gt_data['agent_types'] = gt_agent_data['agent_types']
                gt_data['lg_type'] = gt_agent_data['lg_type']
                
            gt_sample = convert_data_to_unified_format(gt_data, dataset_name=f'{self.cfg.dataset_name}_gt')
            
            if len(sample['G']) > 0:
                samples.append(sample)
            gt_samples.append(gt_sample)

        lane_metrics = compute_lane_metrics(samples=samples, gt_samples=gt_samples)
        agent_metrics = compute_agent_metrics(samples=samples, gt_samples=gt_samples)

        print("--------------------------------------------------------------------------")
        print("Lane metrics: ", ["{}: {:.2f}".format(k,v) for (k,v) in lane_metrics.items()])
        print("Agent metrics: ", ["{}: {:.2f}".format(k,v) for (k,v) in agent_metrics.items()])
        print("--------------------------------------------------------------------------")

        metrics = {
            'lane_metrics': lane_metrics,
            'agent_metrics': agent_metrics
        }
        # save metrics to file
        metrics_path = os.path.join(self.cfg.eval.metrics.metrics_save_path, 'metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved to {metrics_path}")