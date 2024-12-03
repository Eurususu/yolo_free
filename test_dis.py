import torch.distributed as dist
dist.init_process_group(backend="nccl", init_method="env://")
def print_func():
    print(f"Process {dist.get_rank()} initialized.")
    print('end!!')


if __name__ == '__main__':
    print_func()