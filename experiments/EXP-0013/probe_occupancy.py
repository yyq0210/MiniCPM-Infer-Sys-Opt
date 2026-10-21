"""Read CoreX's predicted occupancy for the exact EXP13 compiled kernels.

No achieved-occupancy measurement or official throughput run is performed.
The caller must hold the shared GPU lock and enforce the run-window deadline.
"""
import ctypes as C
import datetime
import hashlib
import json
from pathlib import Path

import torch
import trial


def main():
    torch.cuda.init()
    lib = C.CDLL('/usr/local/corex/lib64/libcuda.so.1')
    attr = lib.ixdrvDeviceGetAttribute
    attr.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
    attr.restype = C.c_int
    fattr = lib.ixdrvFuncGetAttribute
    fattr.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
    fattr.restype = C.c_int
    occ = lib.ixdrvOccupancyMaxActiveBlocksPerMultiprocessor
    occ.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
    occ.restype = C.c_int

    def get(fn, *args):
        value = C.c_int(-1)
        rc = fn(C.byref(value), *args)
        if rc != 0:
            raise RuntimeError(f'{fn.__name__} returned {rc}')
        return value.value

    # Enum values verified against the installed CoreX IX/ixrt/ix.h.
    device = torch.cuda.current_device()
    props = {name: get(attr, enum, device) for name, enum in (
        ('warp_size', 10), ('multiprocessor_count', 16),
        ('max_threads_per_multiprocessor', 39),
        ('max_shared_memory_per_multiprocessor', 81),
        ('max_registers_per_multiprocessor', 82))}
    result = dict(kind='CoreX_driver_theoretical_occupancy_not_achieved_occupancy',
                  measured_at=datetime.datetime.now().astimezone().isoformat(),
                  properties=props, kernels=[])
    for b, length in ((64, 4700), (30, 16384), (1, 4096)):
        x = trial.make([1]*b, [length-i % 17 for i in range(b)])
        for layout, rows in ((0, 16), (1, 16), (0, 8), (1, 8)):
            cfg = dict(decode_warps=4, decode_stages=1, segments=16,
                       decode_layout=layout, decode_rows=rows)
            main_fn, _ = trial.components(x, cfg)
            kernel = main_fn()
            torch.cuda.synchronize()
            handle = C.c_void_p(kernel.function)
            threads = kernel.metadata.num_warps * kernel.metadata.target.warp_size
            assert threads == 4 * props['warp_size']
            static_shared = get(fattr, 1, handle)
            dynamic_shared = kernel.metadata.shared
            active_blocks = get(occ, handle, threads, dynamic_shared)
            regs = get(fattr, 4, handle)
            assert regs == kernel.n_regs
            result['kernels'].append(dict(B=b, L=length, config=cfg,
                block_threads=threads, registers_per_thread=regs,
                static_shared_bytes=static_shared, dynamic_shared_bytes=dynamic_shared,
                local_bytes_per_thread=get(fattr, 3, handle),
                theoretical_active_blocks_per_sm=active_blocks,
                theoretical_active_warps_per_sm=active_blocks * kernel.metadata.num_warps,
                theoretical_occupancy_percent=100 * active_blocks * threads /
                    props['max_threads_per_multiprocessor'],
                ir_sha256={kind: hashlib.sha256(kernel.asm[kind].encode()).hexdigest()
                           for kind in ('ttir','ttgir','llir','asm')
                           if isinstance(kernel.asm.get(kind), str)}))
        del x
        torch.cuda.empty_cache()
    Path('occupancy.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
