# pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning

<a href="https://pyspatial.github.io/"><img src='https://img.shields.io/badge/Website-pySpatial-green' alt='Website'></a>
<a href="https://arxiv.org/pdf/2603.00905"><img src='https://img.shields.io/badge/PDF-pySpatial-yellow' alt='PDF'></a>
<a href="#citation"><img src='https://img.shields.io/badge/BibTex-pySpatial-blue' alt='Paper BibTex'></a>

## Installation

Clone the repository with the submodules by using:

```shell
git clone --recursive git@github.com:Zhanpeng1202/pySpatial.git
```

## Environment

Update requirements.txt with correct CUDA version for PyTorch and cuUML, i.e., replacing cu126 and cu12 with your CUDA version.

```shell
conda create -n pySpatial python=3.10
conda activate  pySpatial
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # change to your CUDA version 
pip3 install -r requirement.txt
```

Add API key to enable VLM query
```shell
# save your LLM credential in the bash environment variable
nano ~/.bashrc
export OPENAI_API_KEY=YOUR_CHATGPT_KEY
export GEMINI_API_KEY=YOUR_GEMINI_KEY
export OPEN_ROUTER_KEY=OTHER_MODELS
```

## Datasets & Benchmarks
Download the Mindcube Benchmark, please refer to 
```shell
mkdir datasets
cd datasets
git clone git@github.com:mll-lab-nu/MindCube.git
cd MindCube
bash scripts/bash_scripts/download_data.bash
```

## Quick Evaluation on the MindCube Benchmark


```python
  python -m reconstruct_pipe \
        --mode batch \
        --jsonl_path "datasets/MindCube/data/raw/MindCube_tinybench.jsonl" \
        --base_data_path "datasets/MindCube/data" \
        --gpu_ids "0,1"
  # preprocess the Mindcube Tinybench, change the argument to the Mindcube dataset downloaded


  # run evaluation on the mindcube dataset
  python -m mindcube \
        --jsonl_path "datasets/MindCube/data/raw/MindCube_tinybench.jsonl" \
        --processed_dir "output/preprocessed" \
        --num_threads 1 \
  # noted that if you are using high-tier API account, there is a larger threshold, set num_thread larger
    
```


## Acknowledgement

This work is built on other amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [VGGT](https://github.com/facebookresearch/vggt) and [CUT3R](https://github.com/CUT3R/CUT3R)
- [MindCube](https://github.com/mll-lab-nu/MindCube) and [VPGen](https://github.com/j-min/VPGen)





## Citation
If you find this paper and code useful, please kindly cite our work:
---
```bibtex
  @inproceedings{luo2026pySpatial,
    title={pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning},
    author={Luo, Zhanpeng and Zhang, Ce and Yong, Silong and Dai, Cunxi and
    Wang, Qianwei and Ran, Haoxi and Shi, Guanya and Sycara, Katia and Xie, Yaqi},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=yv15C8ql24}
  }
```
