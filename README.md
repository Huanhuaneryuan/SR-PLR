# SR-PLR
# Sequential Recommendation with Probabilistic Logical Reasoning (SR-PLR)

This is the implementation of our experiments in the paper.

This implementation contains the datasets, model source code and config used in SR-PLR. Our codes refer to the libraries used in RecBole.


## Environments

Python 3.7.3


## Example to run the codes

-   Running commands can be found in [`./command.sh`]
-   For example:

```
# SR-PLR on Amazon_Sports_and_Outdoors dataset based on SASRec
> python run_recbole.py --dataset='Amazon_Sports_and_Outdoors' --reg_weight=1e-2 --neg_sam_num=3 --learning_rate=0.001 --base_model='SASRec'
```
