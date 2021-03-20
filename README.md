# 基于Pytorch的中文文本摘要生成

1. 开这个仓库的主要目的是记录一下自己实验过程和数据。

2. 参考文本摘要领域大佬写的两篇论文： [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/pdf/1705.04304.pdf) and [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)，然后参考另一位大佬修改的代码[Text-Summarizer-Pytorch-Chinese](https://github.com/LowinLi/Text-Summarizer-Pytorch-Chinese).

另外，在这里还是要感谢一下[@LowinLi](https://github.com/LowinLi)。这里的所有内容基本上没做什么修改（python读取文件的时候出现编码问题，我的猜想是大佬用的mac系统，类linux，所以对编码不敏感，我用windows的话就报错了。），最多修改了一下超参数，刚开始在自己windows笔记本上跑的话，确实有点吃力，设置的batch_size=10,好像后来还直接报cuda错误，我的猜想就是设置过大了，显存承受不了。说多了。直接看训练和测试效果吧。

## 实验结果

| 指标    | 验证集 | 测试集 |
| ------- | ------ | ------ |
| ROUGE-1 | 34.06  | 31.87  |
| ROUGE-2 | 16.46  | 15.47  |
| ROUGE-L | 33.83  | 30.93  |

## 0. 数据预处理

下载[数据集](https://pan.baidu.com/s/1tYP4Xch7uZ2SE_XUkmKofA )(提取码：g8c6 )，下载完之后放在根目录下的Pre LCSTS，有三个文件，train.csv，eval.csv,test.csv。

我直接使用的是大佬的词表（data/vocab），当然了你也可以使用自己跑出来的词表(data/all_vocab.txt)，我自己也跑了一份，具体的操作是遍历[LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html)所有的字，然后排序计数。

如果你要是使用自己的词表的话，记得替换并且更改名字。

```shell
python make_data_files.py
```

然后在data目录下的chunked里面有很多bin的文件，到这里数据基本上都导入完成了。

## 1.  超参数

```python
train_data_path = "data/chunked/train/train_*" # 100条数据
valid_data_path = "data/chunked/valid/valid_*"
test_data_path = "data/chunked/test/test_*"
vocab_path = "data/vocab"
demo_vocab_path = "vocab"
demo_vocab_size = 40000

# Hyperparameters
hidden_dim = 512
emb_dim = 256
batch_size = 30
max_enc_steps = 100  #99% of the articles are within length 55
max_dec_steps = 20  #99% of the titles are within length 15
beam_size = 4
min_dec_steps = 3
vocab_size = 40000

lr = 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 5000000

save_model_path = "data/saved_models"
demo_model_path = "data/saved_models"

intra_encoder = True
intra_decoder = True

cuda = False
```

你需要改的就是max_iterations = 5000000、batch_size=30这两个吧，如果有别的想法可以改别的。第一个参数是循环多少次会停止，第二个参数是一次性训练多少条数据，也就是给GPU送入多少数据，batch_size=30在我们实验室的服务器上大概占用3G的显存吧，当然这会跟显存的带宽有关系，请多试几次。

## 训练模型

训练模型总共分为两个阶段，训练MLE，然后接着使用RL接着训练。

### 训练MLE

```shell
sh train.sh
```

或者

```shell
python train.py --train_mle=yes --train_rl=no --mle_weight=1.0
```

训练开始之后会出现这样的字符：

```shell
2021-03-15 10:55:30,345 - data_util.log - INFO - log启动
2021-03-15 10:55:31,851 - data_util.log - INFO - Bucket queue size: 0, Input queue size: 0
2021-03-15 10:56:31,862 - data_util.log - ERROR - Found batch queue thread dead. Restarting.
2021-03-15 10:56:31,867 - data_util.log - ERROR - Found batch queue thread dead. Restarting.
2021-03-15 10:56:31,869 - data_util.log - ERROR - Found batch queue thread dead. Restarting.
2021-03-15 10:56:31,870 - data_util.log - ERROR - Found batch queue thread dead. Restarting.
2021-03-15 10:56:31,883 - data_util.log - INFO - Bucket queue size: 0, Input queue size: 28825
2021-03-15 10:56:45,162 - data_util.log - INFO - iter:50  mle_loss:7.056  reward:0.0000
2021-03-15 10:56:53,157 - data_util.log - INFO - iter:100  mle_loss:6.302  reward:0.0000
2021-03-15 10:57:00,994 - data_util.log - INFO - iter:150  mle_loss:6.196  reward:0.0000
2021-03-15 10:57:08,355 - data_util.log - INFO - iter:200  mle_loss:6.108  reward:0.0000
2021-03-15 10:57:15,955 - data_util.log - INFO - iter:250  mle_loss:6.126  reward:0.0000
2021-03-15 10:57:23,537 - data_util.log - INFO - iter:300  mle_loss:6.033  reward:0.0000
2021-03-15 10:57:30,982 - data_util.log - INFO - iter:350  mle_loss:5.934  reward:0.0000
2021-03-15 10:57:31,946 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 29936
2021-03-15 10:57:38,275 - data_util.log - INFO - iter:400  mle_loss:5.885  reward:0.0000
2021-03-15 10:57:46,289 - data_util.log - INFO - iter:450  mle_loss:5.862  reward:0.0000
2021-03-15 10:57:53,543 - data_util.log - INFO - iter:500  mle_loss:5.873  reward:0.0000
2021-03-15 10:58:01,334 - data_util.log - INFO - iter:550  mle_loss:5.828  reward:0.0000
2021-03-15 10:58:08,818 - data_util.log - INFO - iter:600  mle_loss:5.862  reward:0.0000
2021-03-15 10:58:16,338 - data_util.log - INFO - iter:650  mle_loss:5.840  reward:0.0000
2021-03-15 10:58:23,746 - data_util.log - INFO - iter:700  mle_loss:5.766  reward:0.0000
2021-03-15 10:58:31,165 - data_util.log - INFO - iter:750  mle_loss:5.867  reward:0.0000
2021-03-15 10:58:32,003 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-15 10:58:39,490 - data_util.log - INFO - iter:800  mle_loss:5.765  reward:0.0000
2021-03-15 10:58:47,942 - data_util.log - INFO - iter:850  mle_loss:5.769  reward:0.0000
2021-03-15 10:58:54,740 - data_util.log - INFO - iter:900  mle_loss:5.715  reward:0.0000
2021-03-15 10:59:02,514 - data_util.log - INFO - iter:950  mle_loss:5.766  reward:0.0000
2021-03-15 10:59:10,067 - data_util.log - INFO - iter:1000  mle_loss:5.741  reward:0.0000
2021-03-15 10:59:17,518 - data_util.log - INFO - iter:1050  mle_loss:5.679  reward:0.0000
2021-03-15 10:59:24,629 - data_util.log - INFO - iter:1100  mle_loss:5.714  reward:0.0000
2021-03-15 10:59:32,057 - data_util.log - INFO - iter:1150  mle_loss:5.666  reward:0.0000
2021-03-15 10:59:32,064 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-15 10:59:40,150 - data_util.log - INFO - iter:1200  mle_loss:5.660  reward:0.0000
2021-03-15 10:59:49,003 - data_util.log - INFO - iter:1250  mle_loss:5.636  reward:0.0000
2021-03-15 10:59:56,388 - data_util.log - INFO - iter:1300  mle_loss:5.597  reward:0.0000
2021-03-15 11:00:03,390 - data_util.log - INFO - iter:1350  mle_loss:5.579  reward:0.0000
2021-03-15 11:00:10,570 - data_util.log - INFO - iter:1400  mle_loss:5.706  reward:0.0000
2021-03-15 11:00:17,938 - data_util.log - INFO - iter:1450  mle_loss:5.644  reward:0.0000
2021-03-15 11:00:24,644 - data_util.log - INFO - iter:1500  mle_loss:5.618  reward:0.0000
2021-03-15 11:00:32,124 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-15 11:00:32,585 - data_util.log - INFO - iter:1550  mle_loss:5.634  reward:0.0000
2021-03-15 11:00:40,778 - data_util.log - INFO - iter:1600  mle_loss:5.525  reward:0.0000
2021-03-15 11:00:48,666 - data_util.log - INFO - iter:1650  mle_loss:5.581  reward:0.0000
2021-03-15 11:00:54,864 - data_util.log - INFO - iter:1700  mle_loss:5.554  reward:0.0000
2021-03-15 11:01:01,017 - data_util.log - INFO - iter:1750  mle_loss:5.611  reward:0.0000
2021-03-15 11:01:07,166 - data_util.log - INFO - iter:1800  mle_loss:5.484  reward:0.0000
2021-03-15 11:01:13,867 - data_util.log - INFO - iter:1850  mle_loss:5.494  reward:0.0000
2021-03-15 11:01:21,360 - data_util.log - INFO - iter:1900  mle_loss:5.545  reward:0.0000
2021-03-15 11:01:29,120 - data_util.log - INFO - iter:1950  mle_loss:5.588  reward:0.0000
2021-03-15 11:01:32,152 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
```

当然了，如果你出现了`2021-03-15 11:01:32,152 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000`一直是这个，证明你没有开始训练，这种问题我也出现过，修改超参数比如最大迭代次数问题就应该可以解决了，代码是没有问题的。

### 训练过程中eval

当你觉得loss值不再变的时候，可以进行强化学习：

```shell
2021-03-16 08:23:41,143 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-16 08:23:47,014 - data_util.log - INFO - iter:516300  mle_loss:3.463  reward:0.0000
2021-03-16 08:23:54,694 - data_util.log - INFO - iter:516350  mle_loss:3.434  reward:0.0000
2021-03-16 08:24:03,135 - data_util.log - INFO - iter:516400  mle_loss:3.403  reward:0.0000
2021-03-16 08:24:10,869 - data_util.log - INFO - iter:516450  mle_loss:3.437  reward:0.0000
2021-03-16 08:24:18,269 - data_util.log - INFO - iter:516500  mle_loss:3.393  reward:0.0000
2021-03-16 08:24:26,026 - data_util.log - INFO - iter:516550  mle_loss:3.414  reward:0.0000
2021-03-16 08:24:33,511 - data_util.log - INFO - iter:516600  mle_loss:3.504  reward:0.0000
2021-03-16 08:24:41,203 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-16 08:24:41,309 - data_util.log - INFO - iter:516650  mle_loss:3.472  reward:0.0000
2021-03-16 08:24:49,410 - data_util.log - INFO - iter:516700  mle_loss:3.479  reward:0.0000
2021-03-16 08:24:57,243 - data_util.log - INFO - iter:516750  mle_loss:3.563  reward:0.0000
2021-03-16 08:25:06,059 - data_util.log - INFO - iter:516800  mle_loss:3.540  reward:0.0000
2021-03-16 08:25:13,879 - data_util.log - INFO - iter:516850  mle_loss:3.551  reward:0.0000
2021-03-16 08:25:20,023 - data_util.log - INFO - iter:516900  mle_loss:3.595  reward:0.0000
2021-03-16 08:25:27,078 - data_util.log - INFO - iter:516950  mle_loss:3.717  reward:0.0000
2021-03-16 08:25:33,924 - data_util.log - INFO - iter:517000  mle_loss:3.560  reward:0.0000
2021-03-16 08:25:41,259 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-16 08:25:41,488 - data_util.log - INFO - iter:517050  mle_loss:3.472  reward:0.0000
2021-03-16 08:25:48,918 - data_util.log - INFO - iter:517100  mle_loss:3.520  reward:0.0000
2021-03-16 08:25:56,369 - data_util.log - INFO - iter:517150  mle_loss:3.456  reward:0.0000
2021-03-16 08:26:04,071 - data_util.log - INFO - iter:517200  mle_loss:3.503  reward:0.0000
2021-03-16 08:26:11,431 - data_util.log - INFO - iter:517250  mle_loss:3.509  reward:0.0000
2021-03-16 08:26:17,750 - data_util.log - INFO - iter:517300  mle_loss:3.488  reward:0.0000
2021-03-16 08:26:23,923 - data_util.log - INFO - iter:517350  mle_loss:3.670  reward:0.0000
2021-03-16 08:26:30,206 - data_util.log - INFO - iter:517400  mle_loss:3.510  reward:0.0000
2021-03-16 08:26:36,375 - data_util.log - INFO - iter:517450  mle_loss:3.606  reward:0.0000
2021-03-16 08:26:41,319 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
```

然后运行以下指令（主要是找出来哪个模型最好，以便于使用强化学习接着训练）：

运行

```shell
sh eval.sh
```

```shell
sh eval.sh
```

或者

```shell
python eval.py --task=validate --start_from=0005000.tar
```

从第一个模型开始评分：

```shell
2021-03-18 14:48:03,350 - data_util.log - INFO - log启动
2021-03-18 14:49:32,299 - data_util.log - INFO - log启动
2021-03-18 14:49:34,276 - data_util.log - INFO - 
2021-03-18 14:49:49,181 - data_util.log - INFO - 0005000.tar rouge_1:0.1865 rouge_2:0.0661 rouge_l:0.1900
2021-03-18 14:49:49,227 - data_util.log - INFO - 
2021-03-18 14:50:00,993 - data_util.log - INFO - 0010000.tar rouge_1:0.1965 rouge_2:0.0716 rouge_l:0.2019
2021-03-18 14:50:01,038 - data_util.log - INFO - 
2021-03-18 14:50:12,789 - data_util.log - INFO - 0015000.tar rouge_1:0.2301 rouge_2:0.0868 rouge_l:0.2244
...
2021-03-18 15:00:36,958 - data_util.log - INFO - 
2021-03-18 15:00:48,971 - data_util.log - INFO - 0265000.tar rouge_1:0.2856 rouge_2:0.1161 rouge_l:0.2828
2021-03-18 15:00:49,024 - data_util.log - INFO - 
2021-03-18 15:01:01,095 - data_util.log - INFO - 0270000.tar rouge_1:0.2802 rouge_2:0.1028 rouge_l:0.2716
2021-03-18 15:01:49,636 - data_util.log - INFO - 
...
2021-03-18 15:10:38,436 - data_util.log - INFO - 
2021-03-18 15:10:52,565 - data_util.log - INFO - 0505000.tar rouge_1:0.2513 rouge_2:0.0926 rouge_l:0.2486
2021-03-18 15:10:52,690 - data_util.log - INFO - 
2021-03-18 15:11:06,227 - data_util.log - INFO - 0510000.tar rouge_1:0.2461 rouge_2:0.0885 rouge_l:0.2474
2021-03-18 15:11:06,338 - data_util.log - INFO - 
2021-03-18 15:11:20,102 - data_util.log - INFO - 0515000.tar rouge_1:0.2560 rouge_2:0.1003 rouge_l:0.2515
```

可以看出来026500.tar模型比较好，用来接着强化训练：

```shell
2021-03-18 15:22:07,362 - data_util.log - INFO - Bucket queue size: 10, Input queue size: 20657
2021-03-18 15:22:35,495 - data_util.log - INFO - iter:265050  mle_loss:3.474  reward:0.2142
2021-03-18 15:22:58,299 - data_util.log - INFO - iter:265100  mle_loss:3.413  reward:0.2192
2021-03-18 15:23:07,424 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-18 15:23:21,118 - data_util.log - INFO - iter:265150  mle_loss:3.449  reward:0.2186
2021-03-18 15:23:43,964 - data_util.log - INFO - iter:265200  mle_loss:3.554  reward:0.2085
2021-03-18 15:24:06,700 - data_util.log - INFO - iter:265250  mle_loss:3.540  reward:0.2105
...
2021-03-19 11:00:55,158 - data_util.log - INFO - Bucket queue size: 14, Input queue size: 21296
2021-03-19 11:01:16,996 - data_util.log - INFO - iter:465050  mle_loss:2.570  reward:0.3077
2021-03-19 11:01:33,264 - data_util.log - INFO - iter:465100  mle_loss:2.588  reward:0.3014
2021-03-19 11:01:49,346 - data_util.log - INFO - iter:465150  mle_loss:2.620  reward:0.2966
2021-03-19 11:01:55,266 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-19 11:02:05,480 - data_util.log - INFO - iter:465200  mle_loss:2.601  reward:0.3004
2021-03-19 11:02:21,891 - data_util.log - INFO - iter:465250  mle_loss:2.589  reward:0.3069
2021-03-19 11:02:38,028 - data_util.log - INFO - iter:465300  mle_loss:2.627  reward:0.3010
2021-03-19 11:02:54,130 - data_util.log - INFO - iter:465350  mle_loss:2.660  reward:0.3000
2021-03-19 11:02:55,323 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
....
2021-03-20 10:54:40,981 - data_util.log - INFO - iter:710400  mle_loss:2.293  reward:0.3385
2021-03-20 10:55:13,309 - data_util.log - INFO - iter:710450  mle_loss:2.271  reward:0.3549
2021-03-20 10:55:14,011 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-20 10:55:47,040 - data_util.log - INFO - iter:710500  mle_loss:2.247  reward:0.3531
2021-03-20 10:56:14,072 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-20 10:56:19,350 - data_util.log - INFO - iter:710550  mle_loss:2.265  reward:0.3451
2021-03-20 10:56:45,297 - data_util.log - INFO - iter:710600  mle_loss:2.262  reward:0.3450
2021-03-20 10:57:14,139 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 30000
2021-03-20 10:57:19,086 - data_util.log - INFO - iter:710650  mle_loss:2.239  reward:0.3375
2021-03-20 10:57:49,225 - data_util.log - INFO - iter:710700  mle_loss:2.287  reward:0.3432
```

## 测试

到这一步的时候，已经产生很多个模型，结束训练之后，重复eval步骤，找到最佳模型进行测试。

```shell
2021-03-20 11:00:28,335 - data_util.log - INFO - log启动
2021-03-20 11:00:31,198 - data_util.log - INFO - 
2021-03-20 11:00:44,904 - data_util.log - INFO - 0005000.tar rouge_1:0.1865 rouge_2:0.0661 rouge_l:0.1900
2021-03-20 11:00:44,991 - data_util.log - INFO - 
2021-03-20 11:00:56,660 - data_util.log - INFO - 0010000.tar rouge_1:0.1965 rouge_2:0.0716 rouge_l:0.2019
...
021-03-20 11:29:07,925 - data_util.log - INFO - 
2021-03-20 11:29:19,739 - data_util.log - INFO - 0685000.tar rouge_1:0.3226 rouge_2:0.1505 rouge_l:0.3173
2021-03-20 11:29:19,785 - data_util.log - INFO - 
2021-03-20 11:29:31,498 - data_util.log - INFO - 0690000.tar rouge_1:0.3285 rouge_2:0.1612 rouge_l:0.3278
2021-03-20 11:29:31,544 - data_util.log - INFO - 
2021-03-20 11:29:43,224 - data_util.log - INFO - 0695000.tar rouge_1:0.3336 rouge_2:0.1663 rouge_l:0.3274
2021-03-20 11:29:43,282 - data_util.log - INFO - 
2021-03-20 11:29:54,924 - data_util.log - INFO - 0700000.tar rouge_1:0.3366 rouge_2:0.1652 rouge_l:0.3350
2021-03-20 11:29:54,970 - data_util.log - INFO - 
2021-03-20 11:30:06,664 - data_util.log - INFO - 0705000.tar rouge_1:0.3406 rouge_2:0.1646 rouge_l:0.3383
2021-03-20 11:30:06,709 - data_util.log - INFO - 
2021-03-20 11:30:18,694 - data_util.log - INFO - 0710000.tar rouge_1:0.3175 rouge_2:0.1606 rouge_l:0.3181
```

经过查找之后，可以发现0705000.tar模型是最优的：rouge_1的分数为34.06，rouge_2的分数为16.46，rouge_l的分数为33.83。

查看生成的文章摘要文件：

```
article: 和平时期 瑞德 的 身份 是 叙利亚 男足 国家青年队 成员 7 个 月 前 他 的 身份 变为 寄居 黎巴嫩 的 难民 现在 他 的 身份 是 肾源 叙利亚 难民 正面 对 第三个 冬天 瑞德 的 故事 变得 普遍 肾 却 廉价 他 依靠 卖出 的 肾 度过 这个 冬天 却 不 知道 下个 冬天 会 怎样
ref: 叙利亚 前国足 队员 靠 卖 肾 过冬
dec: 叙利亚 [UNK] 队员 靠 卖 肾 过冬

article: A股 上市公司 一 季报 披露 完毕 投资 动向 隐秘 的 私募 大佬 们 的 不少 重仓股 也 随之 浮出 水面 比起 年报 相对 滞后 的 数据 一 季报 中 披露 出来 的 私募 重仓股 的 数据 显然 时效性 更强 投资者 也 更 能 从 数据 中一 窥 他们 对 当年 行情 的 大概 布局 思路
ref: 私募 大佬 一季度 重仓股 含金量 分析
dec: 私募 大佬 重仓股 曝光 王亚伟 重仓股 曝光

article: 杭州市 有关 部门 规定 如果 商品房 实际 成交价 低于 备案 价格 15% 以上 将 通过 技术手段 限制 网 签限降 还是 不限降 其实 不难 甄别 最能 让 政策 不 被 误读 的 方法 是 不要 对 楼市 价格 出现 的 变动 轻易 表态 让 市场 自行决定 楼市 价格 走势 但 这 一点 恐怕 做 不到
ref: 杭州 商品房 限降 有 多少 误读
dec: 杭州 商品房 价格 不能 误读

article: 近日 私募 排排 网对 全国 近 60 家 私募 基金 进行 问卷调查 显示 私募 基金 对 12 月 行情 相对 乐观 6607% 私募 看涨 相比 11 月 大幅提高 3232% 认为 12 月 存在 结构性 机会 2857% 私募 认为 会 横盘 整理 仅 有 536% 私募 看跌 但 对于 创业板 绝大部分 私募 表示 谨慎
ref: 六成 私募 基金 看涨 12 月 股市行情
dec: 六成 私募 基金 对 12 月 策略 乐观

article: 拟 选址 的 九峰 项目 位于 余杭 中 泰 街道 靠近 临安 这里 以前 是 个 矿坑 相对 较 封闭 偏僻 住户 也 较 少 当然 项目 正式 开工 前 还要 进行 环境影响 评价 有关 部门 表示 会 邀请 媒体 市民 代表 献计献策 一起 参与 提出 自己 的 意见 和 建议
ref: 杭州 西部 规划 建设 垃圾焚烧 厂
dec: 杭州 杭州 将建 垃圾焚烧 城 市民 市民 请 注意

article: 一种 货币 要 成为 国际 货币 必 满足 几个 基本 条件 首先 是 经济体 的 大小 再 是 币值 的 稳定性 另外 惯性 也 是 货币 国际化 中 不可 忽略 的 一点 中长期 来看 人民币 国际化 的 方向 固然 不变 但 速度 上 仍 需 配合 国内 经济 的 转型
ref: 人民币 国际化 还 缺什么
dec: 人民币 国际化 的 [UNK]

article: 国务院 5 月 16 日 公布 通知 要求 大力 促进 就业 公平 高校 毕业生 招聘 不得 设置 性别 毕业 院校 年龄 户籍 等 作为 限制性 要求 据 教育部 新近 公布 的 数字 2013 年 全国 高校 毕业生 达 699 万人 比 2012 年 增加 19 万 刷新纪录
ref: 国务院 要求 招聘 高校 毕业生 不得 设置 年龄 性别 等 限制
dec: 国务院 高校 毕业生 招聘 禁止 设 年龄 限制

article: 国务院 总理 李克强 昨日 主持 召开 国务院 常务会议 部署 加快 推进 节水 供水 重大 水利工程 建设 决定 大幅 增加 国家 创投 引导 资金 促进 新兴产业 发展 开展 大型 灌区 建设工程 建立 政府 和 市场 有机 结合 的 机制 鼓励 和 吸引 社会 资本 参与 工程建设 和 管理
ref: 国务院 推进 172 项 重大 水利工程 建设
dec: 国务院 加快 推进 节水 供水 重大 水利工程 建设

article: 日前 国家 卫 计委 起草 了 医疗 质量 管理 办法 征求意见 稿 并 已 公开 征求意见 其中 规定 医护人员 由于 不负责任 延误 急危 患者 抢救 和 诊治 造成 严重后果 泄露 患者 隐私 开展 医疗 活动 未 遵守 知情 同意 原则 等 构成犯罪 的 依法追究 刑事责任
ref: 医生 泄露 患者 隐私 拟 追 刑责
dec: 中国 泄露 患者 隐私 拟 追 刑责

article: 任志强 表示 之前 几年 能够 预测 敢于 预测 是因为 通过观察 总结 已经 摸清 了 前任 政府 的 楼市 政策 思路 但是 这 一届 政府 只要 不 知道 政策 走向 就 无法 预测 根据 惯例 十八 届 三中全会 将 决定 本届 政府 的 经济 政策 届时 才能 看 明白 中国 经济 周刊
ref: 任志强 不敢 做 房价 预测 帝 了
dec: 任志强 今年 将 [UNK] 预测

article: 高考 在 即 备受 关注 的 高考 改革方案 初见端倪 据 媒体报道 包括 高考 改革 在内 的 教育 考试制度 的 基本 框架 和 总体 思路 目前 已 完成 初稿 笔者 以为 如果 不 愿意 撼动 既得利益 不能 从 自身 放权 做起 将 贻误 改革 的 时机 也 将 消耗 政府部门 的 公信力
ref: 高考 改革 要 啃 硬骨头
dec: 高考 改革 要 啃 硬骨头
```

现在开始测试

运行

```shell
sh test.sh
```

或者

```shell
python eval.py --task=test --load_model=0705000.tar
```

实验结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210320161426639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0ODk2MjA5,size_16,color_FFFFFF,t_70)

说实话不上个图都觉得自己跑出来个假数据。

可以看到测试集的效果：rouge1：31.87,rouge2:15.47,rouge_l:30.93

## 实验结果

| 指标    | 验证集 | 测试集 |
| ------- | ------ | ------ |
| ROUGE-1 | 34.06  | 31.87  |
| ROUGE-2 | 16.46  | 15.47  |
| ROUGE-L | 33.83  | 30.93  |

## 实验结果分析

```
article: 和平时期 瑞德 的 身份 是 叙利亚 男足 国家青年队 成员 7 个 月 前 他 的 身份 变为 寄居 黎巴嫩 的 难民 现在 他 的 身份 是 肾源 叙利亚 难民 正面 对 第三个 冬天 瑞德 的 故事 变得 普遍 肾 却 廉价 他 依靠 卖出 的 肾 度过 这个 冬天 却 不 知道 下个 冬天 会 怎样
ref: 叙利亚 前国足 队员 靠 卖 肾 过冬
dec: 叙利亚 [UNK] 队员 靠 卖 肾 过冬

article: A股 上市公司 一 季报 披露 完毕 投资 动向 隐秘 的 私募 大佬 们 的 不少 重仓股 也 随之 浮出 水面 比起 年报 相对 滞后 的 数据 一 季报 中 披露 出来 的 私募 重仓股 的 数据 显然 时效性 更强 投资者 也 更 能 从 数据 中一 窥 他们 对 当年 行情 的 大概 布局 思路
ref: 私募 大佬 一季度 重仓股 含金量 分析
dec: 私募 大佬 重仓股 曝光 王亚伟 重仓股 曝光
```

从上文罗列出的两个文本来看，虽然说的是两个模型进行了整合，尤其是指针生成网络，按理说应该不会存在UNK的，但是从生成的数据来看，仍然存在OOV问题，和重复等问题，这么短的文本都能出现OOV和重复问题，确实令人寻味。

-----

## 后记

用百度网盘上传模型真的好慢，换迅雷网盘。

[迅雷网盘下载]()已经训练好的模型验证模型，可以直接保存模型，然后运行test.sh，即可得到实验数据。

