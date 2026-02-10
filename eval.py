import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import json
import clip
from transformers import AutoProcessor, AutoModel
import ImageReward as RM

from utils import *
from pipelines.sd3 import sd3_timestep_pipe, update, decode_latent

def main(args):
    prompts = parse_prompts(args.prompt_path)

    # Load IR model
    ir_model = RM.load("ImageReward-v1.0")
    
    # Load PickScore model
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    pick_processor = AutoProcessor.from_pretrained(processor_name_or_path)
    pick_model = AutoModel.from_pretrained(model_pretrained_name_or_path).to('cuda')

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-L/14", device='cuda')

    result_path = os.path.join("results", args.exp_name)
    results_dict = {}

    ir_score_list = []
    pick_score_list = []
    clip_score_list = []
    hpsv2_score_list = []
    for prompt_idx, prompt in enumerate(tqdm(prompts)):

        # get ir
        ir_score = get_ir(ir_model, prompt, os.path.join(result_path, f"{prompt_idx:05d}.png"), device='cuda')
        # get pickscore
        pick_score = get_pickscore(pick_model, pick_processor, prompt, os.path.join(result_path, f"{prompt_idx:05d}.png"), device='cuda')
        # get clipscore
        clip_score = get_clipscore(clip_model, clip_preprocess, prompt, os.path.join(result_path, f"{prompt_idx:05d}.png"), device='cuda')
        # get hpsv2 score
        hpsv2_score = get_hpsv2(prompt, os.path.join(result_path, f"{prompt_idx:05d}.png"), device='cuda')

        results_dict[prompt_idx] = {
            "prompt": prompt,
            "prompt_idx": prompt_idx,   
            "ir_score": ir_score,
            "pick_score": pick_score,
            "clip_score": clip_score,
            "hpsv2_score": hpsv2_score
        }
        ir_score_list.append(ir_score)
        pick_score_list.append(pick_score)
        clip_score_list.append(clip_score)
        hpsv2_score_list.append(hpsv2_score)
    results_dict["average"] = {
        "ir_score": np.mean(ir_score_list),
        "pick_score": np.mean(pick_score_list),
        "clip_score": np.mean(clip_score_list),
        "hpsv2_score": np.mean(hpsv2_score_list)
    }
    

    with open(os.path.join(result_path, "eval_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with argparse")
    parser.add_argument("--exp_name", type=str, default="debug", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--prompt_path", type=str, default="data.csv", help="Path to the prompt file")

    args = parser.parse_args()
    main(args)