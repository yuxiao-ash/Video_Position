import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input , list):
                for i in range(len(self.next_input)):
                    if torch.is_tensor(self.next_input[i]):
                        self.next_input[i] = self.next_input[i].cuda(non_blocking=True)
            else:
                self.next_input = self.next_input.cuda(non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input
