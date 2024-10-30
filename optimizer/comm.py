import torch
import deepspeed.comm as dist
import numpy as np
import argparse
import deepspeed
import os
from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.accelerator import get_accelerator
from statistics import mean
'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
from deepspeed import comm as dist
import cupy
import numpy as np

from deepspeed.runtime.compression.cupy import CupyBackend
from deepspeed.accelerator import get_accelerator

class NcclBackend(object):
    def __init__(self, mpu=None):
        if mpu is None:
            self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        else:
            self.mpu = mpu
            self.world_group = self.mpu.get_data_parallel_group()
        self.rank = dist.get_rank(group=self.world_group)
        self.size = dist.get_world_size(group=self.world_group)
        self.compression_backend = CupyBackend()
        self.bool_not_supported = False
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR >= 1 and TORCH_MINOR >= 10:
            self.bool_not_supported = True
        
        self.toggle_flag = True

    def my_igather(self, rank, size, group, sendbuf, recvbuf, root):
        req = []
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    req.append(dist.irecv(recvbuf[idx], src=idx, group=group))
                else:
                    recvbuf[rank] = sendbuf
        else:
            req.append(dist.isend(sendbuf, group=group, dst=root))
        return req

    def my_gather(self, rank, size, group, sendbuf, recvbuf, root):
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    dist.recv(recvbuf[idx], src=idx, group=group)
                else:
                    recvbuf[rank] = sendbuf
        else:
            dist.send(sendbuf, group=group, dst=root)

    def compressed_allreduce(self,
                             buffer_m: torch.tensor,
                             local_rank):

        # all_start_time = time.time()
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)
        original_size = buffer_m.numel()
        cupy.cuda.Device(local_rank).use()

        worker_scale = torch.tensor(1.00, device = buffer_m.device)

        if self.bool_not_supported:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    buffer_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                self.size)
        else:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(buffer_m.sign_().add_(1).bool()),
                self.size)
        cupy_worker_scale = self.compression_backend.torch2cupy(worker_scale)

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        # cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        sign_list_packed = [
            self.compression_backend.cupy2torch(cupy_sign_list_packed[idx])
            for idx in range(self.size)
        ]

        # worker_scale = self.compression_backend.cupy2torch(cupy_worker_scale)
        recvbuf_sign = self.compression_backend.cupy2torch(cupy_recvbuf_sign)
        #recvbuf_scale = self.compression_backend.cupy2torch(cupy_recvbuf_scale)
        recvbuf_scale = [
            torch.zeros(1,
                        dtype=worker_scale.dtype,
                        device=torch.device(get_accelerator().device_name(local_rank)))
            for i in range(self.size)
        ]

        # communication phase 1
        # gather_start = time.time()
        # Alltoall for sign
        dist.all_to_all_single(recvbuf_sign,
                               torch.stack(sign_list_packed),
                               group=self.world_group)
        # Allgather for scale
        dist.all_gather(recvbuf_scale, worker_scale, group=self.world_group)

        # gather_end = time.time()

        # cupy_sign_list_packed, sign_list_packed, cupy_worker_scale, worker_scale = None, None, None, None
        cupy_sign_list_packed = None

        cupy_recvbuf_sign = self.compression_backend.torch2cupy(recvbuf_sign)
        #cupy_recvbuf_scale = self.compression_backend.torch2cupy(torch.stack(recvbuf_scale))

        compensated_server_m = self.compression_backend.cupy2torch(
            (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
                self.size,
                -1)).float().add_(-0.5).mul_(2.0).mul_(
                    torch.stack(recvbuf_scale).mul_(1 / self.size)).sum(0)
        # server_scale = torch.norm(compensated_server_m) / np.sqrt(
        #     compensated_server_m.numel())
        server_scale =  torch.tensor(1.000, device = buffer_m.device)

        # 合計が0の要素に対して、交互に1と-1を適用する処理
        zero_elements = (compensated_server_m == 0)  # 合計が0の要素を取得
        if self.toggle_flag:
            compensated_server_m[zero_elements] = 1
        else:
            compensated_server_m[zero_elements] = -1
        self.toggle_flag = not self.toggle_flag  # フラグの切り替え

        # cupy_server_scale = self.compression_backend.torch2cupy(server_scale)

        if self.bool_not_supported:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                1)
        else:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool()),
                1)
        compensated_server_m = None

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_recvbuf_sign.dtype)
        # cupy_recvbuf_sign, recvbuf_sign = None, None
        cupy_recvbuf_sign = None

        server_sign_packed = [
            self.compression_backend.cupy2torch(cupy_server_sign_packed[0])
        ]
        recvbuf_sign_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_sign_server[idx])
            for idx in range(self.size)
        ]

        # server_scale = self.compression_backend.cupy2torch(cupy_server_scale)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)
        # cupy_recvbuf_scale, recvbuf_scale = None, None

        recvbuf_scale_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_scale_server[idx])
            for idx in range(self.size)
        ]

        # Communication Phase 2
        dist.all_gather(recvbuf_sign_server,
                        server_sign_packed[0],
                        group=self.world_group)
        dist.all_gather(recvbuf_scale_server, server_scale, group=self.world_group)

        cupy_server_sign_packed = None

        # need to convert from a tensor list to a single tensor
        # dist.all_gather only provides a tensor list as the recv/output buffer
        recvbuf_sign_server = torch.stack(recvbuf_sign_server)

        cupy_recvbuf_sign_server = self.compression_backend.torch2cupy(
            recvbuf_sign_server)

        buffer_m.data.copy_(
            self.compression_backend.cupy2torch(
                (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(
                    self.size,
                    -1)).float().add_(-0.5).mul_(2.0).mul_(
                        self.compression_backend.cupy2torch(
                            cupy_recvbuf_scale_server)).flatten().data)
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)

        return buffer_m
    
    def compressed_allreduce_nonflip(self,
                             buffer_m: torch.tensor,
                             local_rank):

        # all_start_time = time.time()
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)
        original_size = buffer_m.numel()
        cupy.cuda.Device(local_rank).use()

        worker_scale = torch.tensor(1.00, device = buffer_m.device)

        if self.bool_not_supported:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    buffer_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                self.size)
        else:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(buffer_m.sign_().add_(1).bool()),
                self.size)
        cupy_worker_scale = self.compression_backend.torch2cupy(worker_scale)

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        # cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        sign_list_packed = [
            self.compression_backend.cupy2torch(cupy_sign_list_packed[idx])
            for idx in range(self.size)
        ]

        # worker_scale = self.compression_backend.cupy2torch(cupy_worker_scale)
        recvbuf_sign = self.compression_backend.cupy2torch(cupy_recvbuf_sign)
        #recvbuf_scale = self.compression_backend.cupy2torch(cupy_recvbuf_scale)
        recvbuf_scale = [
            torch.zeros(1,
                        dtype=worker_scale.dtype,
                        device=torch.device(get_accelerator().device_name(local_rank)))
            for i in range(self.size)
        ]

        # communication phase 1
        # gather_start = time.time()
        # Alltoall for sign
        dist.all_to_all_single(recvbuf_sign,
                               torch.stack(sign_list_packed),
                               group=self.world_group)
        # Allgather for scale
        dist.all_gather(recvbuf_scale, worker_scale, group=self.world_group)

        # gather_end = time.time()

        # cupy_sign_list_packed, sign_list_packed, cupy_worker_scale, worker_scale = None, None, None, None
        cupy_sign_list_packed = None

        cupy_recvbuf_sign = self.compression_backend.torch2cupy(recvbuf_sign)
        #cupy_recvbuf_scale = self.compression_backend.torch2cupy(torch.stack(recvbuf_scale))

        compensated_server_m = self.compression_backend.cupy2torch(
            (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
                self.size,
                -1)).float().add_(-0.5).mul_(2.0).mul_(
                    torch.stack(recvbuf_scale).mul_(1 / self.size)).sum(0)
        # server_scale = torch.norm(compensated_server_m) / np.sqrt(
        #     compensated_server_m.numel())
        server_scale =  torch.tensor(1.000, device = buffer_m.device)
        # cupy_server_scale = self.compression_backend.torch2cupy(server_scale)

        if self.bool_not_supported:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                1)
        else:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool()),
                1)
        compensated_server_m = None

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_recvbuf_sign.dtype)
        # cupy_recvbuf_sign, recvbuf_sign = None, None
        cupy_recvbuf_sign = None

        server_sign_packed = [
            self.compression_backend.cupy2torch(cupy_server_sign_packed[0])
        ]
        recvbuf_sign_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_sign_server[idx])
            for idx in range(self.size)
        ]

        # server_scale = self.compression_backend.cupy2torch(cupy_server_scale)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)
        # cupy_recvbuf_scale, recvbuf_scale = None, None

        recvbuf_scale_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_scale_server[idx])
            for idx in range(self.size)
        ]

        # Communication Phase 2
        dist.all_gather(recvbuf_sign_server,
                        server_sign_packed[0],
                        group=self.world_group)
        dist.all_gather(recvbuf_scale_server, server_scale, group=self.world_group)

        cupy_server_sign_packed = None

        # need to convert from a tensor list to a single tensor
        # dist.all_gather only provides a tensor list as the recv/output buffer
        recvbuf_sign_server = torch.stack(recvbuf_sign_server)

        cupy_recvbuf_sign_server = self.compression_backend.torch2cupy(
            recvbuf_sign_server)

        buffer_m.data.copy_(
            self.compression_backend.cupy2torch(
                (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(
                    self.size,
                    -1)).float().add_(-0.5).mul_(2.0).mul_(
                        self.compression_backend.cupy2torch(
                            cupy_recvbuf_scale_server)).flatten().data)
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)

        return buffer_m
    
    def compressed_allreduce_woscale(self,
                             buffer_m: torch.tensor,
                             local_rank):

        # all_start_time = time.time()
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)
        original_size = buffer_m.numel()
        cupy.cuda.Device(local_rank).use()

        worker_scale = torch.norm(buffer_m) / np.sqrt(buffer_m.numel())
        #worker_scale = torch.tensor([1])
        print("worker_scale",worker_scale)

        if self.bool_not_supported:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    buffer_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                self.size)
        else:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(buffer_m.sign_().add_(1).bool()),
                self.size)
        cupy_worker_scale = self.compression_backend.torch2cupy(worker_scale)

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        # cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        sign_list_packed = [
            self.compression_backend.cupy2torch(cupy_sign_list_packed[idx])
            for idx in range(self.size)
        ]

        # worker_scale = self.compression_backend.cupy2torch(cupy_worker_scale)
        recvbuf_sign = self.compression_backend.cupy2torch(cupy_recvbuf_sign)
        #recvbuf_scale = self.compression_backend.cupy2torch(cupy_recvbuf_scale)
        recvbuf_scale = [
            torch.zeros(1,
                        dtype=worker_scale.dtype,
                        device=torch.device(get_accelerator().device_name(local_rank)))
            for i in range(self.size)
        ]

        # communication phase 1
        # gather_start = time.time()
        # Alltoall for sign
        dist.all_to_all_single(recvbuf_sign,
                               torch.stack(sign_list_packed),
                               group=self.world_group)
        # Allgather for scale
        dist.all_gather(recvbuf_scale, worker_scale, group=self.world_group)

        # gather_end = time.time()

        # cupy_sign_list_packed, sign_list_packed, cupy_worker_scale, worker_scale = None, None, None, None
        cupy_sign_list_packed = None

        cupy_recvbuf_sign = self.compression_backend.torch2cupy(recvbuf_sign)
        #cupy_recvbuf_scale = self.compression_backend.torch2cupy(torch.stack(recvbuf_scale))

        compensated_server_m = self.compression_backend.cupy2torch(
            (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
                self.size,
                -1)).float().add_(-0.5).mul_(2.0).mul_(
                    torch.stack(recvbuf_scale).mul_(1 / self.size)).sum(0)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(
            compensated_server_m.numel())
        server_scale = torch.tensor(1.000, device = compensated_server_m.device)

        # cupy_server_scale = self.compression_backend.torch2cupy(server_scale)

        if self.bool_not_supported:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                1)
        else:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool()),
                1)
        compensated_server_m = None

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_recvbuf_sign.dtype)
        # cupy_recvbuf_sign, recvbuf_sign = None, None
        cupy_recvbuf_sign = None

        server_sign_packed = [
            self.compression_backend.cupy2torch(cupy_server_sign_packed[0])
        ]
        recvbuf_sign_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_sign_server[idx])
            for idx in range(self.size)
        ]

        # server_scale = self.compression_backend.cupy2torch(cupy_server_scale)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)
        # cupy_recvbuf_scale, recvbuf_scale = None, None

        recvbuf_scale_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_scale_server[idx])
            for idx in range(self.size)
        ]

        # Communication Phase 2
        dist.all_gather(recvbuf_sign_server,
                        server_sign_packed[0],
                        group=self.world_group)
        dist.all_gather(recvbuf_scale_server, server_scale, group=self.world_group)

        cupy_server_sign_packed = None

        # need to convert from a tensor list to a single tensor
        # dist.all_gather only provides a tensor list as the recv/output buffer
        recvbuf_sign_server = torch.stack(recvbuf_sign_server)

        cupy_recvbuf_sign_server = self.compression_backend.torch2cupy(
            recvbuf_sign_server)

        buffer_m.data.copy_(
            self.compression_backend.cupy2torch(
                (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(
                    self.size,
                    -1)).float().add_(-0.5).mul_(2.0).mul_(
                        self.compression_backend.cupy2torch(
                            cupy_recvbuf_scale_server)).flatten().data)
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)

        return buffer_m
    

    def compressed_allreduce_ef(self,
                             buffer_m: torch.tensor,
                             worker_error,
                             server_error,
                             local_rank):

        # all_start_time = time.time()
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)
        original_size = buffer_m.numel()
        worker_error_size = worker_error.numel()
        cupy.cuda.Device(local_rank).use()

        if original_size != worker_error_size:
            empty_tensor = torch.zeros(worker_error_size - original_size,
                                       device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(buffer_m.numel())
        worker_error.set_(buffer_m - worker_scale *
                          buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        if self.bool_not_supported:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    buffer_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                self.size)
        else:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(buffer_m.sign_().add_(1).bool()),
                self.size)
        cupy_worker_scale = self.compression_backend.torch2cupy(worker_scale)

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        # cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        sign_list_packed = [
            self.compression_backend.cupy2torch(cupy_sign_list_packed[idx])
            for idx in range(self.size)
        ]

        # worker_scale = self.compression_backend.cupy2torch(cupy_worker_scale)
        recvbuf_sign = self.compression_backend.cupy2torch(cupy_recvbuf_sign)
        #recvbuf_scale = self.compression_backend.cupy2torch(cupy_recvbuf_scale)
        recvbuf_scale = [
            torch.zeros(1,
                        dtype=worker_scale.dtype,
                        device=torch.device(get_accelerator().device_name(local_rank)))
            for i in range(self.size)
        ]

        # communication phase 1
        # gather_start = time.time()
        # Alltoall for sign
        dist.all_to_all_single(recvbuf_sign,
                               torch.stack(sign_list_packed),
                               group=self.world_group)
        # Allgather for scale
        dist.all_gather(recvbuf_scale, worker_scale, group=self.world_group)

        # gather_end = time.time()

        # cupy_sign_list_packed, sign_list_packed, cupy_worker_scale, worker_scale = None, None, None, None
        cupy_sign_list_packed = None

        cupy_recvbuf_sign = self.compression_backend.torch2cupy(recvbuf_sign)
        #cupy_recvbuf_scale = self.compression_backend.torch2cupy(torch.stack(recvbuf_scale))

        compensated_server_m = self.compression_backend.cupy2torch(
            (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
                self.size,
                -1)).float().add_(-0.5).mul_(2.0).mul_(
                    torch.stack(recvbuf_scale).mul_(1 / self.size)).sum(0)
        compensated_server_m.add_(server_error)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(
            compensated_server_m.numel())
        server_error.set_(
            compensated_server_m - server_scale *
            compensated_server_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        # cupy_server_scale = self.compression_backend.torch2cupy(server_scale)

        if self.bool_not_supported:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                1)
        else:
            cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    compensated_server_m.sign_().add_(1).bool()),
                1)
        compensated_server_m = None

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_recvbuf_sign.dtype)
        # cupy_recvbuf_sign, recvbuf_sign = None, None
        cupy_recvbuf_sign = None

        server_sign_packed = [
            self.compression_backend.cupy2torch(cupy_server_sign_packed[0])
        ]
        recvbuf_sign_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_sign_server[idx])
            for idx in range(self.size)
        ]

        # server_scale = self.compression_backend.cupy2torch(cupy_server_scale)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)
        # cupy_recvbuf_scale, recvbuf_scale = None, None

        recvbuf_scale_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_scale_server[idx])
            for idx in range(self.size)
        ]

        # Communication Phase 2
        dist.all_gather(recvbuf_sign_server,
                        server_sign_packed[0],
                        group=self.world_group)
        dist.all_gather(recvbuf_scale_server, server_scale, group=self.world_group)

        cupy_server_sign_packed = None

        # need to convert from a tensor list to a single tensor
        # dist.all_gather only provides a tensor list as the recv/output buffer
        recvbuf_sign_server = torch.stack(recvbuf_sign_server)

        cupy_recvbuf_sign_server = self.compression_backend.torch2cupy(
            recvbuf_sign_server)

        buffer_m.data.copy_(
            self.compression_backend.cupy2torch(
                (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(
                    self.size,
                    -1)).float().add_(-0.5).mul_(2.0).mul_(
                        self.compression_backend.cupy2torch(
                            cupy_recvbuf_scale_server)).flatten().data)
        if original_size != worker_error_size:
            buffer_m = buffer_m[0:original_size]
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)

        return buffer_m

    def compressed_allreduce_ef_worker(self,
                                   buffer_m: torch.tensor,
                                   worker_error,
                                   local_rank):
        """
        Performs a compressed all-reduce operation focusing only on the worker's error without handling server errors.

        Args:
            buffer_m (torch.tensor): The tensor buffer to be reduced.
            worker_error (torch.tensor): The error tensor associated with the worker.
            local_rank (int): The local rank of the process.
        """

        # Preserve the original shape of the buffer
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)
        original_size = buffer_m.numel()
        worker_error_size = worker_error.numel()

        # Set the CUDA device
        cupy.cuda.Device(local_rank).use()

        # Ensure buffer_m and worker_error have the same size
        if original_size != worker_error_size:
            empty_tensor = torch.zeros(worker_error_size - original_size,
                                       device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        # Add worker_error to buffer_m and update worker_error
        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(buffer_m.numel())
        worker_error.set_(buffer_m - worker_scale *
                          buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        # Compress the sign of buffer_m
        if self.bool_not_supported:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(
                    buffer_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
                self.size)
        else:
            cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
                self.compression_backend.torch2cupy(buffer_m.sign_().add_(1).bool()),
                self.size)
        cupy_worker_scale = self.compression_backend.torch2cupy(worker_scale)

        # Prepare receive buffers
        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)

        sign_list_packed = [
            self.compression_backend.cupy2torch(cupy_sign_list_packed[idx])
            for idx in range(self.size)
        ]

        recvbuf_sign = self.compression_backend.cupy2torch(cupy_recvbuf_sign)
        recvbuf_scale = [
            torch.zeros(1,
                        dtype=worker_scale.dtype,
                        device=torch.device(get_accelerator().device_name(local_rank)))
            for _ in range(self.size)
        ]

        # Communication Phase 1
        # Alltoall for sign
        dist.all_to_all_single(recvbuf_sign,
                               torch.stack(sign_list_packed),
                               group=self.world_group)
        # Allgather for scale
        dist.all_gather(recvbuf_scale, worker_scale, group=self.world_group)

        # Clear the packed sign list to free memory
        cupy_sign_list_packed = None

        # Update buffer_m with the received signs and scales
        buffer_m.data.copy_(
            self.compression_backend.cupy2torch(
                (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
                    self.size,
                    -1)).float().add_(-0.5).mul_(2.0).mul_(
                        self.compression_backend.cupy2torch(
                            cupy_worker_scale)).flatten().data)

        # Restore the original size and shape of buffer_m
        if original_size != worker_error_size:
            buffer_m = buffer_m[0:original_size]
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)
    
        return buffer_m