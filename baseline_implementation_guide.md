# Baseline Implementation

## Important Details

For RoG, we use LLaMA2-Chat-7B.

## Important Note

The packages being used by Reasoning-on-graphs are mostly quite well-supported.

So if you want to avoid most of the below issues, simply install the newest version of stable python (3.12.8 in January 2024), and carry out the whole installation and inference steps, and use the following as reference in case there are any issues.

## Implementation Details

Used the Python version appropriate for January 2024: 3.11.8

pybind11 gave some problems. Simple google search helped on this one.

```xml
pip install wheel setuptools pip --upgrade
```

then separately install pybind11. 

graph-walker gave problems for installation. Removed it temporarily from requirements.txt, installed the rest, then installed graph-walker. 

### Graph-Walker inference

graph-walker again gave problems while trying to run inference:

ImportError: /home/ubuntu/miniconda3/envs/rog/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/ubuntu/miniconda3/envs/rog/lib/python3.11/site-packages/_walker.cpython-311-x86_64-linux-gnu.so)

Seems like there is some cpython library that is required. 

Tried the following fixes given in:

https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0

Solution 1:

```xml
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install --only-upgrade libstdc++6
```

This installed the required package, but didn’t finally solve it. 

Solution 2:

```xml
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/miniconda3/envs/rog/lib/
```

This finally solved it.

Solution 3:

```xml
conda install -c conda-forge libstdcxx-ng
```

### Transformers Issue

```xml
cannot import name 'log' from 'torch.distributed.elastic.agent.server.api'
```

Upgrade deepspeed version to 0.14.4

RuntimeError: Failed to import transformers.modeling_utils because of the following error (look up to see its traceback):
`np.complex_` was removed in the NumPy 2.0 release. Use `np.complex128` instead.

Do the following:

```xml
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
```

### Token Issue

```xml
huggingface_hub.errors.LocalTokenNotFoundError: Token is required (`token=True`), but no token found. You need to provide a token or be logged in to Hugging Face with `huggingface-cli login` or `huggingface_hub.login`. See https://huggingface.co/settings/tokens.
```

Solution:

https://stackoverflow.com/questions/74593644/how-to-fix-no-token-found-error-while-downloading-hugging-face

```xml
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('YOUR_TOKEN_HERE')"
```

### NVIDIA driver issues

```xml
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```

Auto-installing an NVIDIA driver

https://askubuntu.com/questions/1258904/how-do-i-know-which-nvidia-driver-i-need

### While running predict answer

Libraries required by ChatGPT will be asked for, because the llms.language_models module imports all the relevant files for language models. 

One can either remove them from imports or continue to install them. 

I am going to ignore them. 

## Running Running Running

```xml
python src/qa_prediction/gen_rule_path.py \
        --model_name RoG \
        --model_path rmanluo/RoG \
        -d RoG-webqsp \
        --split test \
        --n_beam 3
```

```xml
python src/qa_prediction/gen_rule_path.py \
        --model_name RoG \
        --model_path rmanluo/RoG \
        -d RoG-cwq \
        --split test \
        --n_beam 3
```

```xml
python src/qa_prediction/predict_answer.py \
        --model_name RoG \
        --model_path rmanluo/RoG \
        -d RoG-webqsp \
        --prompt_path prompts/llama2_predict.txt \
        --add_rule \
        --rule_path results/gen_rule_path/RoG-webqsp/RoG/test/predictions_3_False.jsonl \
        -n 1
```

```xml
python src/qa_prediction/predict_answer.py \
        --model_name RoG \
        --model_path rmanluo/RoG \
        -d RoG-cwq \
        --prompt_path prompts/llama2_predict.txt \
        --add_rule \
        --rule_path results/gen_rule_path/RoG-cwq/RoG/test/predictions_3_False.jsonl
```

## Conclusion for ClaimBenchKG baseline

Its definitely been fine-tuned, on both the KGQA datasets, using Freebase as the KG.

In theory, we could set this up entirely ourselves as well. The repo is decently well-documented.

But I assume this won't be a priority until the others work as well.