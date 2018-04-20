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
def trainable_variables(self):
	return variables
def compute_gradients(self,losses,val_list):
	return grads_and_vars
def apply_gradients(self,grads_and_vars,global_step):
	return run_op
```

## model_utils.py
用于实现特定任务中的一些特定功能,比如图像识别中的预处理操作


## net_utils.py
网络的基础模块,公用库

```
class _Base_Module(object):
    def __init__(self, layers):
        self.layers = layers
        self._trainable_variables = []

    @abc.abstractmethod
    def call(self, X):
        return X

    def __call__(self, X,**kwargs):
        return self.call(X)

    @property
    def trainable_variables(self):
        for i in range(len(self.layers)):
            self._trainable_variables.append(self.layers[i].trainable_variables)
            return self._trainable_variables


class Flow(_Base_Module):
    def call(self,X):
        for i in range(len(self.layers)):
            X = self.layers[i](X)
        return X


class ResFlow(Flow):
    def call(self,X,add=False):
        X_ = super().call(X)
        if add:
            X = X_+X
        else:
            X = tf.concat([X_,X],axis=-1)
        return X


class DenseFlow(_Base_Module):
    def call(self,X):
        for i in range(len(self.layers)):
            X_= self.layers[i](X)
            X = tf.concat([X_,X],axis=-1)
        return X


class Block(_Base_Module):
    def call(self,X):
        X_ = []
        for i in range(len(self.layers)):
            X_.append(self.layers[i](X))
        X = tf.concat(X_,axis=-1)
        return X
```

## datasets.py
数据接口,由于产生tf.data.Dataset类

## __main__.py
-- 模型入口
