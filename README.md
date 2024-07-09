# 2021290209
MataAdapt
# Introduction

MetaAdapt存储库是ACL 2023 Paper的PyTorch实现 [MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning](https://arxiv.org/abs/2305.12692)

![image](https://github.com/Clound-Computing/2021290209/assets/133028358/ee3dd61b-abd1-4f78-af20-a7a5b1869b62)


我们提出了一种基于元学习的领域自适应少镜头错误信息检测方法。MetaAdapt利用有限的目标示例来提供反馈，并指导知识从源领域转移到目标领域(即，学习适应)。特别是，我们用多个源任务训练初始模型，并计算它们与元任务的相似度分数。基于相似性分数，我们重新调整元梯度以自适应地从源任务中学习。因此，MetaAdapt可以学习如何调整错误信息检测模型，并利用源数据来提高目标域中的性能。为了证明我们的方法的效率和有效性，我们进行了大量的实验，将MetaAdapt与最先进的基线和大型语言模型(llm)(如LLaMA)进行比较，其中MetaAdapt在实际数据集上以大幅减少的参数在域自适应少镜头错误信息检测方面取得了更好的性能。


## Citing 

如果您在研究中使用我们的方法，请考虑引用以下论文:
```
@inproceedings{yue2023metaadapt,
  title={MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning},
  author={Yue, Zhenrui and Zeng, Huimin and Zhang, Yang and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}

@inproceedings{yue2022contrastive,
  title={Contrastive Domain Adaptation for Early Misinformation Detection: A Case Study on COVID-19},
  author={Yue, Zhenrui and Zeng, Huimin and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 31th ACM International Conference on Information & Knowledge Management},
  year={2022}
}
```


## Data & Requirements

采用的数据集是公开的，如果您在获取数据集方面有困难，请与我们联系。要运行我们的代码，您需要PyTorch和Transformers，请参阅requirements.txt以了解我们的运行环境


## Run MetaAdapt

```bash
python src/metaadapt.py --source_data_path=PATH/TO/SOURCE --source_data_type=SOURCE_DATASET --target_data_path=PATH/TO/TARGET --target_data_type=TARGET_DATASET --output_dir=OUTPUT_DIR;
```
执行上述命令(带参数)以适应错误信息检测模型，从FEVER, GettingReal, GossipCop, LIAR和PHEME中选择源数据集，从CoAID, Constraint和ANTiVax中选择目标数据集。采用的模型是RoBERTa，元学习的功能版本写在roberta_utils.py中。经过训练的模型和评估指标可以在OUTPUT_DIR中找到。我们提供了一个从FEVER到ANTiVax的示例命令，其学习率和温度参数如下:
```bash
python src/metaadapt.py --source_data_path=PATH/TO/FEVER --source_data_type=fever --target_data_path=PATH/TO/ANTiVax --target_data_type=antivax --learning_rate_meta=1e-5 --learning_rate_learner=1e-5 --softmax_temp=0.1 --output_dir=fever2antivax;
```


## Performance
![image](https://github.com/Clound-Computing/2021290209/assets/133028358/9d39c141-f185-4d84-818f-14d5a8756e68)

## Acknowledgement

在实现过程中，我们的代码主要基于 [Transformers](https://github.com/huggingface/transformers) from Hugging Face and [MetaST](https://github.com/microsoft/MetaST) by Wang et al. 非常感谢这些作者的伟大工作!
