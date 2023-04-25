# Sequential Recommendation with Probabilistic Logical Reasoning (SR-PLR)

This is the implementation for IJCAI 2023 [paper](https://arxiv.org/abs/2304.11383).

This implementation contains the datasets, model source code and config used in SR-PLR. Our codes refer to the libraries used in [RecBole](https://github.com/RUCAIBox/RecBole).


## Environments

Python 3.7.3


## Example to run the codes

-   Running commands can be found in [`./command.sh`]
-   For example:

```
# SR-PLR on Amazon_Sports_and_Outdoors dataset based on SASRec
> python run_recbole.py --dataset='Amazon_Sports_and_Outdoors' --reg_weight=1e-2 --neg_sam_num=3 --learning_rate=0.001 --base_model='SASRec'
```

## References

This paper mainly referes RecBole and BetaE. Thanks for your nice works! 
```bibtex
@inproceedings{ren_beta_2020,
  author    = {Hongyu Ren and
               Jure Leskovec},
  title     = {Beta embeddings for multi-hop logical reasoning in knowledge graphs},
  booktitle = {NIPS 2020},
  pages = {19716--19726},
  year      = {2020}
}

@inproceedings{recbole,
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM} 2021},
  pages     = {4653--4664},
  year      = {2021}
}
```
