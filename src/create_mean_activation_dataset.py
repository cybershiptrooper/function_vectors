import os, json
import torch, numpy as np
import argparse

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from compute_indirect_effect import compute_indirect_effect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    models = {
        "gptj" : "EleutherAI/gpt-j-6b",
        "pythia" : "EleutherAI/pythia-2.8b-deduped"
    }
    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--n_top_heads', help='Number of attenion head outputs used to compute function vector', required=False, type=int, default=10)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=-1) 

    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/pythia-2.8b-deduped')

    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='../results')
    parser.add_argument('--ie_path_root', help='File path to load indirect effects from', type=str, required=False, default=None)
    parser.add_argument('--mean_activations_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--indirect_effect_path', help='Path to file containing indirect_effect scores for the specified task', required=False, type=str, default=None)   

    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') 
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=25)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})  

    parser.add_argument('--compute_baseline', help='Whether to compute the model baseline 0-shot -> n-shot performance', type=bool, required=False, default=True)
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument("--metric", help="Metric to use when evaluating generated strings", type=str, required=False, default="f1_score")
        
    args = parser.parse_args()  