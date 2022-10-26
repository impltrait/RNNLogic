# RNNLogic+

在此文件夹中，我们提供了RNNLogic+的重构代码，这是本文第3.4节中介绍的RNNLogic的改进版本。

RNNLogic+ 的理念是首先通过运行 RNNLogic（不带 emb）来学习有用的逻辑规则，然后使用这些逻辑规则来训练更强大的推理预测器。通过这种方式，即使没有使用知识图谱嵌入，RNNLogic+也能通过emb实现与RNNLogic接近的结果。

若要运行 RNNLogic+，您可以执行以下步骤。

## Step 1: Mine logic rules

In the first step, we mine some low-quality logic rules, which are used to pre-train the rule generator in RNNLogic+ to speed up training.

To do that, go to the folder `miner`, and compile the codes by running the following command:

`g++ -O3 rnnlogic.h rnnlogic.cpp main.cpp -o rnnlogic -lpthread`

Afterwards, run the following command to mine logic rules:

`./rnnlogic -data-path ../data/FB15k-237 -max-length 3 -threads 40 -lr 0.01 -wd 0.0005 -temp 100 -iterations 1 -top-n 0 -top-k 0 -top-n-out 0 -output-file mined_rules.txt`

The codes run on CPUs. Thus it is better to use a server with many CPUs and use more threads by adjusing the option `-thread`. The program will output a file called `mined_rules.txt`, and you can move the file to your dataset folder.

**In `data/FB15k-237` and `data/wn18rr`, we have provided these mined rules, so you can skip this step.**

## Step 2: Run RNNLogic+

接下来，我们准备运行 RNNLogic。为此，请先编辑文件夹“config”中的配置文件，然后转到文件夹“src”。

如果您想使用single-GPU训练，请编辑第39行和第60行，然后进一步运行:

`python run_rnnlogic.py --config ../config/FB15k-237.yaml` 

`python run_rnnlogic.py --config ../config/wn18rr.yaml` 

If you would like to use multi-GPU training, please run:

`python -m torch.distributed.launch --nproc_per_node=4 run_rnnlogic.py --config ../config/FB15k-237.yaml`

`python -m torch.distributed.launch --nproc_per_node=4 run_rnnlogic.py --config ../config/wn18rr.yaml`

## Results and Discussion

Using the defaul configuration files, we are able to achieve the following results without using knowledge graph embeddings:

**FB15k-237:**

```
Hit1 : 0.242949
Hit3 : 0.358812
Hit10: 0.494145
MR   : 384.201315
MRR  : 0.327182
```

**WN18RR:**

```
Hit1 : 0.439614
Hit3 : 0.483718
Hit10: 0.537939
MR   : 6377.744942
MRR  : 0.471933
```

**Discussion:**

Note that for efficiency consideration, the default configurations are quite conservative, and it is easy to further improve the results.

For example:

- Current configuration files only consider logic rules which are not longer than 3. You might consider longer logic rules for better reasoning results.
- Current configuration files specify the training iterations to 5. You might increase the value for better results.
- Current configuration files specify the hidden dimension in the reasoning predictor to 16. You might also increase the value for better results.
