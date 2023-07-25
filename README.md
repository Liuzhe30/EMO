# EMO
Predicting the effect of non-coding mutations on quantitative gene expression using deep learning

<p align="center"><img width="100%" src="model/EMO_structure.png" /></p>

## Environment
- Python == 3.9
- Tensorflow-gpu == 2.7
- Protobuf == 3.20
- Scikit-learn == 1.1

## Model weights
|model|trained weights|
|:---:|:---:|
|Small|[Download]()|
|Middle|[Download]()|
|Large|[Download]()|
|Huge|[Download]()|

## Training and fine-tuning
You can specify the model size through the command interface:
```
cd [work_path]
python training.py -m small --epoch 100 --lr 0.005 --save_dir model/weights/
python finetune.py -m small -t Pancreas
```