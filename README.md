# Unified-Anomaly-Detection-using-Knowledge-Distillation-and-Quantization

## Unified Student-Teacher Feature Pyramid Matching 
Student-Teacher Feature Pyramid Matching - Directory for Training and Testing
# Dataset
Download dataset from [MvTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

# Training
Train a model:
```
python main.py train --epochs 400
```
After running this command, a directory `snapshots/` should be created, inside which checkpoint will be saved.

# Testing
Evaluate a model:
```
python main.py test --checkpoint snapshots/best_394_83.pt

This command will evaluate the model specified by --checkpoint argument. 


# Citation

If you find the work useful in your research, please cite our papar.
```
@article{jena2024unified,
  title={Unified Anomaly Detection methods on Edge Device using Knowledge Distillation and Quantization},
  author={Jena, Sushovan and Pulkit, Arya and Singh, Kajal and Banerjee, Anoushka and Joshi, Sharad and Ganesh, Ananth and Singh, Dinesh and Bhavsar, Arnav},
  journal={arXiv preprint arXiv:2407.02968},
  year={2024}
}
```
