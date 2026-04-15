---
license: apache-2.0
pipeline_tag: robotics
library_name: transformers
---

# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions

The model was presented in the paper [UniVLA: Learning to Act Anywhere with Task-centric Latent Actions](https://huggingface.co/papers/2505.06111).

## UniVLA-7b for SimplerEnv-Bridge

Code can be found at [https://github.com/OpenDriveLab/UniVLA](https://github.com/OpenDriveLab/UniVLA).

**🚀 Run the following script to start an evaluation on SimplerEnv-Bridge "Put Spoon on Table Cloth":** 

> Please visit our official repo for detailed instruction.

```bash
ckpt_path="/path/to/your/univla-7b-224-sft-simpler-bridge"
action_decoder_path="/path/to/your/univla-7b-224-sft-simpler-bridge/action_decoder.pt"

python experiments/robot/r2r/real2sim_eval_maniskill3.py \
    --model="univla" -e "PutSpoonOnTableClothInScene-v1" -s 0 --num-episodes 24 --num-envs 1 \
    --action_decoder_path ${action_decoder_path} \
    --ckpt_path ${ckpt_path} \
```

## 📝 Citation
If you find our models useful in your work, please cite [our paper](https://arxiv.org/pdf/2505.06111):

```bibtex
@article{bu2025univla,
  title={Univla: Learning to act anywhere with task-centric latent actions},
  author={Bu, Qingwen and Yang, Yanting and Cai, Jisong and Gao, Shenyuan and Ren, Guanghui and Yao, Maoqing and Luo, Ping and Li, Hongyang},
  journal={arXiv preprint arXiv:2505.06111},
  year={2025}
}
```