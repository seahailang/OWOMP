# OWOMP
one week one model project

# 目标
快速实现一些简单的深度学习模型

# 工具
python 3.6

tensorflow 1.4

numpy

scipy

...

# 代码组织
## config.py
配置文件,包含模型的配置和训练参数的配置

模型的配置由Config类决定

训练参数由FLAG类决定

## model.py
模型文件

包含模型的主体文件,模型搭建的参数由Config类决定
```
class Model(object)
def __int__(self,iterator,config,**argv):
	...
def builf_graph(self):
	...
	self.logits = ...
	return self.logits
def losses(self):
	return losses
@property
def trainable_variable(self):
	return variables
def compute_gradients(self,losses,val_list):
	return grads_and_vars
def apply_gradients(self,grads_and_vars,global_step):
	return run_op
```

## model_utils.py
用于实现特定任务中的一些特定功能,比如图像识别中的预处理操作

## datasets.py
数据接口,由于产生tf.data.Dataset类

## __main__.py
-- 模型入口
