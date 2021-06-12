from tensorboardX import SummaryWriter
import os 

class SummaryWriterManager:
    def __init__(self, path):
        if not os.path.exists(path): os.makedirs(path) 
        self.writer = SummaryWriter(path)
        
    def add_summary(self, tag, value, global_step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)

    def add_summaries(self, dict_, global_step):
        for key in dict_.keys():
            self.add_summary(tag=str(key), value=dict_[key], global_step=global_step)
