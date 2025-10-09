print("START", flush=True)
import torch
print("import torch OK", torch.__version__, flush=True)
print("cuda.is_available(): ", torch.cuda.is_available(), flush=True)