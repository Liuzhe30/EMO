# EMO
Predicting the effect of non-coding mutations on quantitative gene expression using deep learning

<p align="center"><img width="100%" src="model/EMO_structure.png" /></p>

## Environment
- Python == 3.9
- Tensorflow-gpu == 2.7
- Protobuf == 3.20
- Scikit-learn == 1.1

## Training and fine-tuning
You can specify the model size through the command:
```
cd [work_path]
python training.py -m small --epoch 100 --lr 0.005 --save_dir model/weights/
python finetune.py -m small -t Pancreas
```

## Processed training dataset and model weights
|model|trained weights|training data|test data|
|:---:|:---:|:---:|:---:|
|Small|[Download]()|[Download]()|[Download]()|
|Middle|[Download]()|[Download]()|[Download]()|
|Large|[Download]()|[Download]()|[Download]()|
|Huge|[Download]()|[Download]()|[Download]()|

## Raw data
|Data|resource|
|:---:|:---:|
|fine-mapping eQTL|[GTEx v8](https://gtexportal.org/home/datasets)|
|tissue ATAC-seq|[ENCODE](https://www.encodeproject.org/atac-seq/)|
|GRCh38/hg38 genome|[UCSC Genome Browser](https://genome.ucsc.edu/cgi-bin/hgGateway)|