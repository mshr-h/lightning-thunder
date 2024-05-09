import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id1(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]
    )
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T3 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T5 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]
    )
    T6 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T7 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]
    )
    T8 = fd.define_tensor(
        shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0]
    )
    T9 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T10 = fd.ops.permute(T9, dims=[0, 2, 1, 3])
    T11 = fd.ops.stride_order(T10, stride_order=[3, 2, 1, 0])
    S12 = fd.define_scalar(16, dtype=DataType.Int)
    S13 = fd.define_scalar(128, dtype=DataType.Int)
    S14 = fd.define_scalar(1600, dtype=DataType.Int)
    V15 = fd.define_vector([S12, S13, S14], dtype=DataType.Int)
    T16 = fd.ops.reshape(T11, new_shape=V15)
    S17 = fd.define_scalar(2048, dtype=DataType.Int)
    S18 = fd.define_scalar(1600, dtype=DataType.Int)
    V19 = fd.define_vector([S17, S18], dtype=DataType.Int)
    T20 = fd.ops.reshape(T16, new_shape=V19)
    T21 = fd.ops.linear(T20, T1, T0)
    S22 = fd.define_scalar(16, dtype=DataType.Int)
    S23 = fd.define_scalar(128, dtype=DataType.Int)
    S24 = fd.define_scalar(1600, dtype=DataType.Int)
    V25 = fd.define_vector([S22, S23, S24], dtype=DataType.Int)
    T26 = fd.ops.reshape(T21, new_shape=V25)
    S27 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S28 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S29 = fd.define_scalar(16, dtype=DataType.Int)
    S30 = fd.define_scalar(128, dtype=DataType.Int)
    S31 = fd.define_scalar(1600, dtype=DataType.Int)
    V32 = fd.define_vector([S29, S30, S31], dtype=DataType.Int)
    T33 = fd.ops.uniform(S27, S28, shape=V32, dtype=DataType.BFloat16)
    S34 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T35 = fd.ops.lt(T33, S34)
    T36 = fd.ops.cast(T26, dtype=DataType.Float)
    T37 = fd.ops.cast(T35, dtype=DataType.Float)
    T38 = fd.ops.mul(T36, T37)
    S39 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T40 = fd.ops.mul(T38, S39)
    T41 = fd.ops.cast(T8, dtype=DataType.Float)
    T42 = fd.ops.add(T41, T40)
    T43, T44 = fd.ops.var_mean(T42, dims=[2], correction=0, keepdim=False)
    S45 = fd.define_scalar(16, dtype=DataType.Int)
    S46 = fd.define_scalar(128, dtype=DataType.Int)
    S47 = fd.define_scalar(1, dtype=DataType.Int)
    V48 = fd.define_vector([S45, S46, S47], dtype=DataType.Int)
    T49 = fd.ops.broadcast_in_dim(T43, shape=V48, broadcast_dims=[0, 1])
    S50 = fd.define_scalar(16, dtype=DataType.Int)
    S51 = fd.define_scalar(128, dtype=DataType.Int)
    S52 = fd.define_scalar(1, dtype=DataType.Int)
    V53 = fd.define_vector([S50, S51, S52], dtype=DataType.Int)
    T54 = fd.ops.broadcast_in_dim(T44, shape=V53, broadcast_dims=[0, 1])
    S55 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T56 = fd.ops.add(T49, S55)
    T57 = fd.ops.rsqrt(T56)
    S58 = fd.define_scalar(16, dtype=DataType.Int)
    S59 = fd.define_scalar(128, dtype=DataType.Int)
    S60 = fd.define_scalar(1600, dtype=DataType.Int)
    V61 = fd.define_vector([S58, S59, S60], dtype=DataType.Int)
    T62 = fd.ops.broadcast_in_dim(T54, shape=V61, broadcast_dims=[0, 1, 2])
    T63 = fd.ops.sub(T42, T62)
    S64 = fd.define_scalar(16, dtype=DataType.Int)
    S65 = fd.define_scalar(128, dtype=DataType.Int)
    S66 = fd.define_scalar(1600, dtype=DataType.Int)
    V67 = fd.define_vector([S64, S65, S66], dtype=DataType.Int)
    T68 = fd.ops.broadcast_in_dim(T57, shape=V67, broadcast_dims=[0, 1, 2])
    T69 = fd.ops.mul(T63, T68)
    S70 = fd.define_scalar(16, dtype=DataType.Int)
    S71 = fd.define_scalar(128, dtype=DataType.Int)
    S72 = fd.define_scalar(1600, dtype=DataType.Int)
    V73 = fd.define_vector([S70, S71, S72], dtype=DataType.Int)
    T74 = fd.ops.broadcast_in_dim(T3, shape=V73, broadcast_dims=[2])
    T75 = fd.ops.cast(T74, dtype=DataType.Float)
    T76 = fd.ops.mul(T69, T75)
    S77 = fd.define_scalar(16, dtype=DataType.Int)
    S78 = fd.define_scalar(128, dtype=DataType.Int)
    S79 = fd.define_scalar(1600, dtype=DataType.Int)
    V80 = fd.define_vector([S77, S78, S79], dtype=DataType.Int)
    T81 = fd.ops.broadcast_in_dim(T2, shape=V80, broadcast_dims=[2])
    T82 = fd.ops.cast(T81, dtype=DataType.Float)
    T83 = fd.ops.add(T76, T82)
    T84 = fd.ops.cast(T83, dtype=DataType.BFloat16)
    S85 = fd.define_scalar(2048, dtype=DataType.Int)
    S86 = fd.define_scalar(1600, dtype=DataType.Int)
    V87 = fd.define_vector([S85, S86], dtype=DataType.Int)
    T88 = fd.ops.reshape(T84, new_shape=V87)
    T89 = fd.ops.linear(T88, T5, T4)
    S90 = fd.define_scalar(16, dtype=DataType.Int)
    S91 = fd.define_scalar(128, dtype=DataType.Int)
    S92 = fd.define_scalar(6400, dtype=DataType.Int)
    V93 = fd.define_vector([S90, S91, S92], dtype=DataType.Int)
    T94 = fd.ops.reshape(T89, new_shape=V93)
    T95 = fd.ops.cast(T94, dtype=DataType.Float)
    T96 = fd.ops.mul(T95, T95)
    T97 = fd.ops.mul(T96, T95)
    S98 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T99 = fd.ops.mul(S98, T95)
    S100 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T101 = fd.ops.mul(S100, T97)
    T102 = fd.ops.add(T95, T101)
    S103 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T104 = fd.ops.mul(S103, T102)
    T105 = fd.ops.tanh(T104)
    S106 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T107 = fd.ops.add(S106, T105)
    T108 = fd.ops.mul(T99, T107)
    T109 = fd.ops.cast(T108, dtype=DataType.BFloat16)
    S110 = fd.define_scalar(2048, dtype=DataType.Int)
    S111 = fd.define_scalar(6400, dtype=DataType.Int)
    V112 = fd.define_vector([S110, S111], dtype=DataType.Int)
    T113 = fd.ops.reshape(T109, new_shape=V112)
    T114 = fd.ops.linear(T113, T7, T6)
    S115 = fd.define_scalar(16, dtype=DataType.Int)
    S116 = fd.define_scalar(128, dtype=DataType.Int)
    S117 = fd.define_scalar(1600, dtype=DataType.Int)
    V118 = fd.define_vector([S115, S116, S117], dtype=DataType.Int)
    T119 = fd.ops.reshape(T114, new_shape=V118)
    S120 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S121 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S122 = fd.define_scalar(16, dtype=DataType.Int)
    S123 = fd.define_scalar(128, dtype=DataType.Int)
    S124 = fd.define_scalar(1600, dtype=DataType.Int)
    V125 = fd.define_vector([S122, S123, S124], dtype=DataType.Int)
    T126 = fd.ops.uniform(S120, S121, shape=V125, dtype=DataType.BFloat16)
    S127 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T128 = fd.ops.lt(T126, S127)
    T129 = fd.ops.cast(T119, dtype=DataType.Float)
    T130 = fd.ops.cast(T128, dtype=DataType.Float)
    T131 = fd.ops.mul(T129, T130)
    S132 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T133 = fd.ops.mul(T131, S132)
    T134 = fd.ops.add(T42, T133)
    T135 = fd.ops.cast(T134, dtype=DataType.BFloat16)
    fd.add_output(T135)


with FusionDefinition() as fd:
    nvfuser_fusion_id1(fd)

inputs = [
    torch.randn((1600,), dtype=torch.bfloat16, device="cuda:0").as_strided((1600,), (1,)),
    torch.randn((2560000,), dtype=torch.bfloat16, device="cuda:0").as_strided((1600, 1600), (1600, 1)),
    torch.randn((1600,), dtype=torch.bfloat16, device="cuda:0").as_strided((1600,), (1,)),
    torch.randn((1600,), dtype=torch.bfloat16, device="cuda:0").as_strided((1600,), (1,)),
    torch.randn((6400,), dtype=torch.bfloat16, device="cuda:0").as_strided((6400,), (1,)),
    torch.randn((10240000,), dtype=torch.bfloat16, device="cuda:0").as_strided((6400, 1600), (1600, 1)),
    torch.randn((1600,), dtype=torch.bfloat16, device="cuda:0").as_strided((1600,), (1,)),
    torch.randn((10240000,), dtype=torch.bfloat16, device="cuda:0").as_strided((1600, 6400), (6400, 1)),
    torch.randn((3276800,), dtype=torch.bfloat16, device="cuda:0").as_strided((16, 128, 1600), (204800, 1600, 1)),
    torch.randn((3276800,), dtype=torch.bfloat16, device="cuda:0").as_strided((16, 25, 128, 64), (204800, 8192, 64, 1)),
]
fd.execute(inputs)
