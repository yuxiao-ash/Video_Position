from collections import OrderedDict

class GradualWarmupScheduler():
  def __init__(self, optimizer, total_epoch, after_scheduler=None):
    self.total_epoch = total_epoch
    self.after_scheduler = after_scheduler
    self.last_epoch = 0
    self.init_lr = optimizer.param_groups[0]['lr']
    self.optimizer = optimizer
    self.last_lr = self.init_lr / 10
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.init_lr / 10
  
  def state_dict(self):
    out = OrderedDict()
    out['total_epoch'] = self.total_epoch
    out['after_scheduler'] = self.after_scheduler.state_dict()
    out['last_epoch'] = self.last_epoch
    out['init_lr'] = self.init_lr
    out['last_lr'] = self.last_lr
    return out
  
  def load_state_dict(self, orderdict):
    self.total_epoch = orderdict['total_epoch']
    self.after_scheduler.load_state_dict(orderdict['after_scheduler'])
    self.last_epoch = orderdict['last_epoch']
    self.init_lr = orderdict['init_lr']
    self.last_lr = orderdict['last_lr']

  def step(self, epoch=None, metrics=None):
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.last_lr
    self.last_epoch += 1
    if self.last_epoch > self.total_epoch:
      if self.last_epoch == self.total_epoch + 1:
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = self.init_lr
      self.after_scheduler.step()
    else:
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = (self.init_lr-self.init_lr/10)/self.total_epoch*self.last_epoch + self.init_lr/10
    self.last_lr = self.optimizer.param_groups[0]['lr']
    