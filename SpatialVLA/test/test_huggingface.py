import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

parser = argparse.ArgumentParser("Huggingface AutoModel Tesing")
parser.add_argument("--model_name_or_path", default="", help="pretrained model name or path.")
parser.add_argument("--num_images", type=int, default=1, help="num_images for testing.")

args = parser.parse_args()
if __name__ == "__main__":
    model_name_or_path = Path(args.model_name_or_path)
    
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    print(processor.statistics)

    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
    image = Image.open("test/example.png").convert("RGB")
    images = [image] * args.num_images
    prompt = "What action should the robot take to pick the cup?"
    inputs = processor(images=images, text=prompt, unnorm_key="bridge_orig/1.0.0", return_tensors="pt")
    print(inputs)
    
    generation_outputs = model.predict_action(inputs)
    print(generation_outputs, processor.batch_decode(generation_outputs))

    actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
    print(actions)
    
    print("DONE!")
