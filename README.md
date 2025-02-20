# DDL: Dynamic Direction Learning for Semi-Supervised Facial Expression Recognition
## Abstract
Abstract—Most semi-supervised facial expression recognition (FER) algorithms leverage pseudo-labeling to mine additional in- formation from unlabeled samples. Despite its good performance, two critical issues persist: class imbalance and domain shift. The former is a typical challenge due to the significant variation in sample numbers across different FER classes, resulting in highly imbalanced pseudo labels in existing semi-supervised methods. For the latter, given that labeled and unlabeled data usually come from different sources, a considerable domain gap might exist, leading the model to generate low-quality pseudo labels. To tackle these issues, we introduce a novel semi-supervised FER algorithm called Dynamic Direction Learning (DDL), which consists of adaptive balance learning (ABL) and adaptive align- ment learning (AAL). ABL allows a balanced training process by dynamically adjusting the constraints of self-training based on the performance of a balanced validation dataset. Moreover, AAL adaptively aligns the feature distribution of labeled and unlabeled data by minimizing their distance in feature space. Additionally, a role rotation mechanism (RRM) is proposed to avoid confirmation bias, which further improves self-training. Extensive experiments demonstrate that DDL achieves state-of- the-art performance on different FER datasets.
## Framework
![img](DDL_network.png)
## Model Weights
| Model | Weights | Pwd |
|-------|-------|-------|
| DDL-ResNet18 RAF_DB | [DDL-ResNet18 RAF_DB](https://pan.baidu.com/s/17QP_LEz7XvohowUkC-8jGg)  | pevk |
| DDL-ResNet50 RAF_DB | [DDL-ResNet50 RAF_DB](https://pan.baidu.com/s/1cp6SWwjSrnk8068q7Eoq8g) | 7v59 |
## Command
Enter the target directory:
```python
cd src
```
The training command for RAF-DB is:
```python
python train.py --name exp_raf_db --mu 1 --total-steps 250000 --seed 5 --num-classes 7 --finetune-lr 1e-4 --batch-size 32 --seed 5  --finetune-epochs 600  --teacher_lr 0.0018 --student_lr 0.0018 --amp --resize 224 --world-size 2 --workers 4 --randaug 2 16 --warmup-steps 1500 --uda-steps 1500 --nesterov --ema 0.995 --lambda-u 0.5 --label-smoothing 0.15 --w_mmd2 0.5 --lambda-mmd 0.2 --pretrain_model pretrain_model/resnet18_msceleb.pth --log log 
```
