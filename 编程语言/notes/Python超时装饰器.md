最近工作有点多,趁周末有空,继续分享我在学习和使用python过程中的一些小tips.

有没有遇到过这样的事情:对数据库执行插入或更新操作,因为数据量大或其他原因,导致此次操作非常耗时,有时甚至等上好几个小时,也无法完成.很郁闷,怎么操作不超时啊？因为数据库配置时超时时间很长,并且有些操作又是需要很长时间的,所以不能修改默认的超时时间.

因为客观条件不允许,我们不能靠数据库超时来终止此次操作,所以必须要在自己的方法逻辑模块里实现超时检测的功能.

在python里有没有可以不用修改原来的方法内部逻辑,就能实现超时检测呢？肯定有啦,就是利用装饰器.装饰器是什么？在博客园找到了一篇介绍文章:函数和方法装饰漫谈(Function decorator).

废话听完,我现在介绍主角出场:超时装饰器,timeout decorator.

超时检测逻辑:启动新子线程执行指定的方法,主线程等待子线程的运行结果,若在指定时间内子线程还未执行完毕,则判断为超时,抛出超时异常,并杀掉子线程；否则未超时,返回子线程所执行的方法的返回值.

在实现过程中,发现python默认模块里是没有方法可以杀掉线程的,怎么办呢？当然先问问google或百度,果然,keill thread这个关键词很热门,很快就搜索到我想要的东西了:"Kill a thread in Python",就是以下这个KThread类,它继承了threading.Thread,并添加了kill方法,让我们能杀掉它:
```
import sys


class KThread(threading.Thread):
"""A subclass of threading.Thread, with a kill()
    method.

    Come from:
    Kill a thread in Python: 
    http://mail.python.org/pipermail/python-list/2004-May/260937.html
"""
def__init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.killed = False

def start(self):
"""Start the thread."""
        self.__run_backup= self.run
        self.run = self.__run# Force the Thread to install our trace.
        threading.Thread.start(self)

def__run(self):
"""Hacked run function, which installs the
        trace."""
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

def globaltrace(self, frame, why, arg):
if why =='call':
return self.localtrace
else:
return None

def localtrace(self, frame, why, arg):
if self.killed:
if why =='line':
raise SystemExit()
return self.localtrace

def kill(self):
        self.killed = True
```
好了,万事戒备,让我们来完成剩下的代码吧,也就是
```
timeout decorator:
class Timeout(Exception):
"""function run timeout"""

def timeout(seconds):
"""超时装饰器,指定超时时间
    若被装饰的方法在指定的时间内未返回,则抛出Timeout异常"""
def timeout_decorator(func):
"""真正的装饰器"""

def _new_func(oldfunc, result, oldfunc_args, oldfunc_kwargs):
            result.append(oldfunc(*oldfunc_args, **oldfunc_kwargs))

def _(*args, **kwargs):
            result = []
            new_kwargs = { # create new args for _new_func, because we want to get the func return val to result list
'oldfunc': func,
'result': result,
'oldfunc_args': args,
'oldfunc_kwargs': kwargs
            }
            thd = KThread(target=_new_func, args=(), kwargs=new_kwargs)
            thd.start()
            thd.join(seconds)
            alive = thd.isAlive()
            thd.kill() # kill the child thread
if alive:
raise Timeout(u'function run too long, timeout %d seconds.'% seconds)
else:
return result[0]
        _.__name__= func.__name__
        _.__doc__= func.__doc__
return _
return timeout_decorator
真的能运行吗？写个测试程序运行运行:
@timeout(5)
def method_timeout(seconds, text):
print'start', seconds, text
    time.sleep(seconds)
print'finish', seconds, text
return seconds

if__name__=='__main__':
for sec in range(1, 10):
try:
print'*'*20
print method_timeout(sec, 'test waiting %d seconds'% sec)
except Timeout, e:
print e
```