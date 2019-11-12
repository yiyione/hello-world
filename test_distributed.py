import torch
import socket
import argparse
import os
from apex import amp
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel
# from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.n_nodes = int(os.environ['NNODES'])
    args.global_rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.n_gpu_per_node = int(os.environ['NGPU'])
    args.is_master = args.global_rank == 0 and args.local_rank == 0
    args.multi_node = args.n_nodes > 1
    args.multi_gpu = args.world_size > 1

    args.distributed = int(os.environ['WORLD_SIZE']) > 1

# summary
PREFIX = "%i - " % args.global_rank
print(PREFIX + "Number of nodes: %i" % args.n_nodes)
print(PREFIX + "Local rank     : %i" % args.local_rank)
print(PREFIX + "Global rank    : %i" % args.global_rank)
print(PREFIX + "World size     : %i" % args.world_size)
print(PREFIX + "GPUs per node  : %i" % args.n_gpu_per_node)
print(PREFIX + "Master         : %s" % str(args.is_master))
print(PREFIX + "Multi-node     : %s" % str(args.multi_node))
print(PREFIX + "Multi-GPU      : %s" % str(args.multi_gpu))
print(PREFIX + "Hostname       : %s" % socket.gethostname())

if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

# Each process receives its own batch of "fake input data" and "fake target data."
# The "training loop" in each process just uses this fake batch over and over.
# https://github.com/NVIDIA/apex/tree/master/examples/imagenet provides a more realistic
# example of distributed data sampling for both training and validation.
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if args.distributed:
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    # apex.parallel.DistributedDataParallel.
    model = DistributedDataParallel(model)

    # torch.nn.parallel.DistributedDataParallel is also fine, with some added args:
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank)

loss_fn = torch.nn.MSELoss()

for t in range(10):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    if args.local_rank == 0:
        print("=== End of epoch %i ===" % t)
        print("loss = ", loss)

if args.local_rank == 0:
    print("final loss = ", loss)