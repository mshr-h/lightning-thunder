__all__ = [
    "datatype",
    "exact",
    "signedinteger",
    "int8",
    "int8_",
    "int16",
    "int16_",
    "int32",
    "int32_",
    "int64",
    "int64_",
    "unsignedinteger",
    "uint8",
    "uint8_",
    "bool_",
    "bool8",
    "bool8_" "inexact",
    "floating",
    "bfloat16",
    "bfloat16_",
    "float16",
    "float16_",
    "float32",
    "float32_",
    "float64",
    "float64_",
    "complexfloating",
    "complex32",
    "complex32_",
    "complex64",
    "complex64_",
    "complex128",
    "complex128_",
    "all_datatypes",
    "integer_dtypes",
    "low_precision_dtypes",
    "float_dtypes",
    "complex_dtypes",
    "weak_dtypes",
    "strong_dtypes",
    "corresponding_real_dtype",
    "corresponding_complex_dtype",
    "corresponding_weak_dtype",
    "corresponding_strong_dtype",
]

# This file defines Thunder's datatypes. The class hierarchy of these datatypes follows
#   NumPy's, see https://numpy.org/doc/stable/reference/arrays.scalars.html.

# This file depends on trace.py.

# TODO: maybe add nicknames for datatypes, like how torch.long is torch.int64
# TODO: support extensible dtype registration

# TODO: control printing based on ctx
# TODO: control equality testing so torch.int8 == int8 and int8_

# TODO: maybe move dtype-related utilities here


_abstract_classes = set()


class datatype:

    # TODO: in the future might want to use ABCMeta to prevent this and the
    #   abstract classes from being instantiated
    def __new__(cls, *args, **kwargs):
        if cls in _abstract_classes:
            raise TypeError(f"{cls} is abstract and cannot be instantiated!")

        return object.__new__(cls)

    def __init__(self, *, python_type, name, bytes, is_weak):
        self._python_type = python_type
        self._name = name
        self._bytes = bytes
        self._is_weak = is_weak

    # NOTE: these are properties so they appear as read-only
    @property
    def python_type(self):
        return self._python_type

    @property
    def bytes(self):
        return self._bytes

    @property
    def is_weak(self):
        return self._is_weak

    def __repr__(self):
        return f"{self._name}{8 * self._bytes}{'_' if self._is_weak else ''}"

    def __str__(self):
        return self.__repr__()


class exact(datatype):
    """Abstract base class for the signedinteger, unsignedinteger and bool_ datatypes."""

    pass


class signedinteger(exact):
    """Base class for the signed integer datatypes: int8, int16, int32, int64."""

    def __init__(self, name, *, bytes, is_weak):
        super().__init__(python_type=int, name=name, bytes=bytes, is_weak=is_weak)


int8 = signedinteger("int", bytes=1, is_weak=False)
int8_ = signedinteger("int", bytes=1, is_weak=True)
int16 = signedinteger("int", bytes=2, is_weak=False)
int16_ = signedinteger("int", bytes=2, is_weak=True)
int32 = signedinteger("int", bytes=4, is_weak=False)
int32_ = signedinteger("int", bytes=4, is_weak=True)
int64 = signedinteger("int", bytes=8, is_weak=False)
int64_ = signedinteger("int", bytes=8, is_weak=True)


class unsignedinteger(exact):
    """Base class for the unsigned integer datatypes: uint8."""

    def __init__(self, name, *, bytes, is_weak):
        super().__init__(python_type=int, name=name, bytes=bytes, is_weak=is_weak)


uint8 = unsignedinteger("uint", bytes=1, is_weak=False)
uint8_ = unsignedinteger("uint", bytes=1, is_weak=True)


class bool_(exact):
    """Base class for the boolean datatype: bool8."""

    def __init__(self, name, *, is_weak):
        super().__init__(python_type=bool, name=name, bytes=1, is_weak=is_weak)


# NOTE: bool has a weak variant for completeness, but the boolean datatype could always
#   be considered strong or weak without any effect
bool8 = bool_("bool", is_weak=False)
bool8_ = bool_("bool", is_weak=True)


class inexact(datatype):
    """Abstract base class for the floating and complexfloating datatypes."""

    pass


class floating(inexact):
    """Base class for the floating datatypes: bfloat16, float16, float32, float64."""

    def __init__(self, name, *, bytes, is_weak):
        super().__init__(python_type=float, name=name, bytes=bytes, is_weak=is_weak)


bfloat16 = floating("bfloat", bytes=2, is_weak=False)
bfloat16_ = floating("bfloat", bytes=2, is_weak=True)
float16 = floating("float", bytes=2, is_weak=False)
float16_ = floating("float", bytes=2, is_weak=True)
float32 = floating("float", bytes=4, is_weak=False)
float32_ = floating("float", bytes=4, is_weak=True)
float64 = floating("float", bytes=8, is_weak=False)
float64_ = floating("float", bytes=8, is_weak=True)


class complexfloating(inexact):
    """Base class for the complex floating datatypes: complex32, complex64, complex128."""

    def __init__(self, name, *, bytes, is_weak):
        super().__init__(python_type=complex, name=name, bytes=bytes, is_weak=is_weak)


complex32 = complexfloating("complex", bytes=4, is_weak=False)
complex32_ = complexfloating("complex", bytes=4, is_weak=True)
complex64 = complexfloating("complex", bytes=8, is_weak=False)
complex64_ = complexfloating("complex", bytes=8, is_weak=True)
complex128 = complexfloating("complex", bytes=16, is_weak=False)
complex128_ = complexfloating("complex", bytes=16, is_weak=True)


_abstract_classes.update((datatype, exact, inexact))

all_datatypes = (
    bool8,
    bool8_,
    uint8,
    uint8_,
    int8,
    int8_,
    int16,
    int16_,
    int32,
    int32_,
    int64,
    int64_,
    bfloat16,
    bfloat16_,
    float16,
    float16_,
    float32,
    float32_,
    float64,
    float64_,
    complex32,
    complex32_,
    complex64,
    complex64_,
    complex128,
    complex128_,
)


def _filter_dtypes(cls):
    return (dtype for dtype in all_datatypes if isinstance(dtype, cls) and not dtype.is_weak)


# Translates a sequence of dtypes and dtype classes into a concrete set of corresponding (strong) dtypes
def resolve_dtypes(args):
    dtypes = set()
    for arg in args:
        if isinstance(arg, datatype):
            dtypes.add(arg)
            continue

        assert arg in (datatype, exact, signedinteger, unsignedinteger, bool_, inexact, floating, complexfloating)
        dtypes.update(_filter_dtypes(arg))

    return dtypes


integer_dtypes = tuple(d for d in all_datatypes if isinstance(d, exact))

low_precision_dtypes = tuple(d for d in all_datatypes if isinstance(d, inexact) and d.bytes <= 2)

float_dtypes = tuple(d for d in all_datatypes if isinstance(d, floating))

complex_dtypes = tuple(d for d in all_datatypes if isinstance(d, complexfloating))

weak_dtypes = tuple(d for d in all_datatypes if d.is_weak)

strong_dtypes = tuple(d for d in all_datatypes if not d.is_weak)

_complex_to_real_dtype_map = {
    complex128_: float64_,
    complex128: float64,
    complex64_: float32_,
    complex64: float32,
    complex32_: float16_,
    complex32: float16,
    complex: float,
}

_real_to_complex_dtype_map = {
    bfloat16_: complex64_,
    bfloat16: complex64,
    float16_: complex32_,
    float16: complex32,
    float32_: complex64_,
    float32: complex64,
    float64_: complex128_,
    float64: complex128,
    float: complex,
}


def corresponding_real_dtype(dtype):
    return _complex_to_real_dtype_map[dtype]


def corresponding_complex_dtype(dtype):
    return _real_to_complex_dtype_map[dtype]


_dtype_to_weak_dtype_map = {
    bool8: bool8_,
    uint8: uint8_,
    int8: int8_,
    int16: int16_,
    int32: int32_,
    int64: int64_,
    bfloat16: bfloat16_,
    float16: float16_,
    float32: float32_,
    float64: float64_,
    complex32: complex32_,
    complex64: complex64_,
    complex128: complex128_,
}

_weak_dtype_to_dtype_map = {v: k for k, v in _dtype_to_weak_dtype_map.items()}


def corresponding_weak_dtype(dtype):
    if dtype.is_weak:
        return dtype
    return _dtype_to_weak_dtype_map[dtype]


def corresponding_strong_dtype(dtype):
    if not dtype.is_weak:
        return dtype
    return _weak_dtype_to_dtype_map[dtype]
