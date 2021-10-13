The meta data can be downloaded from https://github.com/lzcn/Fashion-Hash-Net.  
Download `images,sentence_vector,tuples_630`, then convert images to `lmdb` format.  
Use `data_process.ipynb` to get `train.csv,test.csv,u_pre.csv`,`u_pre.csv` stores the historical behavior data of each user.

原始数据的说明可以看https://github.com/lzcn/Fashion-Hash-Net/blob/master/data/README.md.  
`train.csv, test_vsc` 中数据每行都是`user,top,bottom,shoe,target`.`target`为正负样本的标注，正样本为1，负样本为0。  
`train.csv` 是`tuples_train_posi,tuples_train_nega`的集合，并且剔除了`u_pre.csv`中的样本，保证与用户历史互斥。
`u_pre.csv` 是从用户的正样本中随机选择的50条样本作为该用户的历史行为记录，共`630*50`条。  

