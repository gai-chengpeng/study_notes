# 简介
Hyperopt：是进行超参数优化的一个类库。有了它我们就可以摆脱手动调参的烦恼，并且往往能够在相对较短的时间内获取原优于手动调参的最终结果。

一般而言，使用hyperopt的方式的过程可以总结为：
- 用于最小化的目标函数
- 搜索空间
- 存储搜索过程中所有点组合以及效果的方法
- 要使用的搜索算法

# 第一个例子：
```
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}
```
# 更多信息
[Hyperopt: Distributed Hyperparameter Optimization](https://github.com/hyperopt/hyperopt)