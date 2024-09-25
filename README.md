# PRP: Attacking LLM Guard-Rails

Code for the ACL 2024 paper: PRP: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails

![image info](./assets/prp.png)

## Setup
### Installation
To run PRP, you can clone this repository and install the package using conda. To install the package, run the following commands:
```bash
$ git clone https://github.com/AshishHoodaIITD/prp-llm-guard-rail-attack.git
$ cd prp-llm-guard-rail-attack
$ conda env create -f environment.yml
```

### Download Models
To download models from hugging face, there are two options:
1. [Recommended] Install git-lfs. Then, git clone the huggingface url of the model at the desired path.
2. Specify the desired path via the ```HF_HOME``` env variable.

### Model Configs
Setup the model config file in the ```configs``` directory. Refer to the existing configs.

## How to Run Attack

### Generate Universal Adversarial Prefix
Specify the attack settings in the config file ```universal_adversarial_prefix.json```, and then run the following
```bash
python attack_uap.py --config universal_adversarial_prefix.json
```

Then, use the ```eval.py``` script to execute the PRP attack. For example, the following command attacks the setting when ```vicuna``` is both the response and the guard model.
```bash
python eval.py --response_model VICUNA_33B --guard_model VICUNA_33B --few_shot 3 --adversarial_prefix results/vicuna_33b_universal_adversarial_prefix.json --num_samples 10
```

## Cite our work
If you find this work useful, please cite our paper:
```
@inproceedings{mangaokar-etal-2024-prp,
    title = "{PRP}: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails",
    author = "Mangaokar, Neal  and
      Hooda, Ashish  and
      Choi, Jihye  and
      Chandrashekaran, Shreyas  and
      Fawaz, Kassem  and
      Jha, Somesh  and
      Prakash, Atul",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.591",
    pages = "10960--10976",
}
```


