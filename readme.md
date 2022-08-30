# Introduction

The CANMD repository is the PyTorch Implementation of CIKM 2022 Paper [Contrastive Domain Adaptation for Early Misinformation Detection: A Case Study on COVID-19](https://arxiv.org/abs/2208.09578)

<img src=pics/intro.png>

We propose contrastive adaptation network for early misinformation detection (CANMD). Specifically, we leverage pseudo labeling to generate high-confidence target examples for joint training with source data. We additionally design a label correction component to estimate and correct the label shifts (i.e., class priors) between the source and target domains. Moreover, a contrastive adaptation loss is integrated in the objective function to reduce the intra-class discrepancy and enlarge the inter-class discrepancy. As such, the adapted model learns corrected class priors and an invariant conditional distribution across both domains for improved estimation of the target data distribution. To demonstrate the effectiveness of the proposed CANMD, we study the case of COVID-19 early misinformation detection and perform extensive experiments using multiple real-world datasets. The results suggest that CANMD can effectively adapt misinformation detection systems to the unseen COVID-19 target domain with significant improvements compared to the state-of-the-art baselines.


## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@inproceedings{yue2022contrastive,
  title={Contrastive Domain Adaptation for Early Misinformation Detection: A Case Study on COVID-19},
  author={Yue, Zhenrui and Zeng, Huimin and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 31th ACM International Conference on Information & Knowledge Management},
  year={2022}
}
```


## Data & Requirements

Download datasets: [FEVER](https://fever.ai/dataset/fever.html), [GettingReal](https://www.kaggle.com/datasets/mrisdal/fake-news), [GossipCop](https://github.com/KaiDMML/FakeNewsNet), [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip), [PHEME](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078), [CoAID](https://github.com/cuilimeng/CoAID), [Constraint](https://competitions.codalab.org/competitions/26655) and [ANTiVax](https://github.com/SakibShahriar95/ANTiVax).

Required packages: PyTorch, pandas, numpy etc. For our running environment see requirements.txt


## Train Source Misinformation Detection Models

```bash
python src/train.py --output_dir=SOURCE_FOLDER --source_data_type=constraint --source_data_path=./data/Constraint;
```
Excecute the above command (with arguments) to train a source misinformation detection model, select dataset and path (i.e., source_data_type and source_data_path) from FEVER, GettingReal, GossipCop, LIAR, PHEME, CoAID, Constraint and ANTiVax. Trained source models could be found under ./experiments/SOURCE_FOLDER.


## Adapt Source Misinformation Detection Models to Target Domain

```bash
python src/adapt.py --load_model_path=SOURCE_FOLDER --output_dir=TARGET_FOLDER --source_data_type=constraint --source_data_path=./data/Constraint --target_data_type=coaid --target_data_path=./data/CoAID --alpha=0.001 --conf_threshold=0.6;
```
Excecute the above command (with arguments) to adapt the source model with CANMD. Specify load_model_path to load the source trained model, select both source and target datasets and paths (i.e., source_data_type, source_data_path, target_data_type and target_data_path) from FEVER, GettingReal, GossipCop, LIAR, PHEME, CoAID, Constraint and ANTiVax. Adapted models could be found under ./experiments/TARGET_FOLDER. Alpha (i.e., lambda in the contrastive loss) and conf_threshold (i.e., tau in confidence thresholding) are hyperparameters for CANMD and can be chosen empirically.


## Performance

<img src=pics/performance.png>

Note that the performance may vary slightly from the reported results as we adopt an updated version of transformers in this repo.


## Acknowledgement

During the implementation we base our code mostly on [Transformers](https://github.com/huggingface/transformers) from Hugging Face and [Abstention](https://github.com/kundajelab/abstention) by Shrikumar et al. Many thanks to these authors for their great work!
