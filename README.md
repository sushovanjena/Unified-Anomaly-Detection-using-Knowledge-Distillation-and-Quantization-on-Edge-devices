# Unified-Anomaly-Detection-using-Knowledge-Distillation-and-Quantization-on-Edge-devices

## Unified Student-Teacher Feature Pyramid Matching (STFPM)

Student-Teacher Feature Pyramid Matching - Directory for Training and Testing
# Dataset
Download dataset from [MvTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

# Checkpoint
Download checkpoint from (https://drive.google.com/file/d/1s4g-ymPedorhmPmlfOFvp_Ls6tJgghEs/view?usp=sharing).

# Environment
gym_env.yaml

# Training in Pytorch
Train a model:
```
Dataset path is set as deafult in the code as 'STAD/data/' in the below way, so downloaded data has to be stored in this heirarchy.

parser.add_argument("--mvtec-ad", type=str, default='../STAD/data', help="MvTec-AD dataset path")
``` 
python main.py train --epochs 400
```
After running this command, a directory `snapshots/` should be created, inside which checkpoint will be saved.
```
# Testing
Evaluate a model:
```
python main.py test --category carpet --checkpoint snapshots/best_394_83.pt
```
This command will evaluate the model specified by --checkpoint argument. 

# Post-Training Quantization (INT-8) in PyTorch
```
python main_ptq.py test --category carpet --checkpoint snapshots/best_394_83.pt
```
Note - PyTorch Quantization is not supported in CUDA, so it runs in CPU.

# Post-Training Quantization (INT-8) in TensorRT
```
python TRT_main_STFPM.py test --category carpet --checkpoint snapshots/best_394_83.pt
```
Note - TensorRT Quantization is not supported in CPU, so it tested in NVIDIA Jetson Xavier NX.

# Quantization-aware Training (INT-8) in PyTorch
```
python main_qat.py train --epochs 400
```

Note - PyTorch Quantization is not supported in CUDA, so it runs in CPU.

# Citation

If you find the work useful in your research, please cite our paper.
```
@article{jena2024unified,
  title={Unified Anomaly Detection methods on Edge Device using Knowledge Distillation and Quantization},
  author={Jena, Sushovan and Pulkit, Arya and Singh, Kajal and Banerjee, Anoushka and Joshi, Sharad and Ganesh, Ananth and Singh, Dinesh and Bhavsar, Arnav},
  journal={arXiv preprint arXiv:2407.02968},
  year={2024}
}
```
