"""Build a C Python binding and CUDA backend without importing PyTorch.

Use ``cmdclass={"build_ext": CUDABuildExt}`` with a normal setuptools
``Extension`` containing .c and .cu sources. Metadata and source-distribution
creation do not require CUDA. Compiling a wheel requires the CUDA toolkit and
a compatible host compiler. The default target is 7.5+PTX; set
TORCH_CUDA_ARCH_LIST (for example ``8.6;12.0+PTX``) or CUDAARCHS (for example
``86-real;120-real;120-virtual``) to select other targets.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import re
import shutil

from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError, PlatformError


def cuda_arch_flags(environ=None):
    """Translate explicit architecture settings into nvcc argument strings.

TORCH_CUDA_ARCH_LIST wins when both variables are set. CUDAARCHS follows
CMake's integer convention: an unsuffixed number emits native code and PTX.
No GPU is queried during a build.
    """
    environ = os.environ if environ is None else environ
    torch_archs = environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    cmake_archs = environ.get("CUDAARCHS", "").strip()
    use_cmake = not torch_archs and bool(cmake_archs)
    setting = torch_archs or cmake_archs or "7.5+PTX"
    flags = []
    for entry in re.split(r"[;,\s]+", setting):
        match = re.fullmatch(
            r"(?:(sm|compute)_)?([1-9][0-9]*(?:\.[0-9])?)(\+PTX|-real|-virtual)?",
            entry,
            re.IGNORECASE,
        )
        if match is None:
            raise PlatformError(
                f"Invalid CUDA architecture {entry!r}; use numeric targets such "
                "as TORCH_CUDA_ARCH_LIST='8.6;12.0+PTX' or CUDAARCHS='86;120'."
            )
        prefix, number, suffix = match.groups()
        prefix = (prefix or "").lower()
        arch = number.replace(".", "")
        if len(arch) < 2 or len(arch) > 3:
            raise PlatformError(f"Invalid CUDA architecture {entry!r}")
        suffix = (suffix or "").lower()
        virtual_only = prefix == "compute" or suffix == "-virtual"
        ptx = virtual_only or suffix == "+ptx" or (use_cmake and not suffix and not prefix)
        codes = ([] if virtual_only else [f"sm_{arch}"]) + ([f"compute_{arch}"] if ptx else [])
        for code in codes:
            flag = f"-gencode=arch=compute_{arch},code={code}"
            if flag not in flags:
                flags.append(flag)
    return flags


def find_cuda_toolkit():
    """Locate nvcc and its toolkit; run only when compilation is requested."""
    executable = "nvcc.exe" if os.name == "nt" else "nvcc"
    explicit_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        nvcc = root / "bin" / executable
        if not nvcc.is_file():
            raise PlatformError(f"CUDA_HOME/CUDA_PATH does not contain {nvcc}")
        return root, nvcc

    on_path = shutil.which(executable)
    if on_path:
        nvcc = Path(on_path).resolve()
        return nvcc.parent.parent, nvcc

    candidates = [Path("/usr/local/cuda")]
    if os.name == "nt":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        base = base / "NVIDIA GPU Computing Toolkit" / "CUDA"
        candidates = sorted(
            base.glob("v*"),
            key=lambda path: tuple(int(part) for part in re.findall(r"\d+", path.name)),
            reverse=True,
        ) if base.is_dir() else []
    for root in candidates:
        nvcc = root / "bin" / executable
        if nvcc.is_file():
            return root.resolve(), nvcc.resolve()
    raise PlatformError(
        "Building cuda_rankic requires the NVIDIA CUDA toolkit (nvcc) and a "
        "compatible C/C++ compiler. Install the toolkit and set CUDA_HOME or CUDA_PATH."
    )


class CUDABuildExt(build_ext):
    """Compile .cu files with nvcc and let setuptools build/link C sources."""

    def build_extension(self, ext):
        cuda_sources = [source for source in ext.sources if source.endswith(".cu")]
        if not cuda_sources:
            return super().build_extension(ext)

        root, nvcc = find_cuda_toolkit()
        compiler_type = self.compiler.compiler_type
        if compiler_type not in ("msvc", "unix"):
            raise PlatformError(f"Unsupported CUDA host compiler: {compiler_type}")
        if compiler_type == "msvc" and not self.compiler.initialized:
            self.compiler.initialize(self.plat_name)

        include_dir = root / "include"
        library_candidates = ([root / "lib" / "x64"] if compiler_type == "msvc" else [
            root / "lib64",
            root / "targets" / f"{platform.machine()}-linux" / "lib",
            root / "lib",
        ])
        library_dir = next((path for path in library_candidates if path.is_dir()), None)
        if not include_dir.is_dir() or library_dir is None:
            raise PlatformError(f"CUDA headers or runtime libraries are missing under {root}")

        fields = (
            "sources", "include_dirs", "library_dirs", "libraries", "extra_objects",
            "extra_compile_args", "depends", "language", "runtime_library_dirs",
        )
        original = {field: getattr(ext, field) for field in fields}
        extra = ext.extra_compile_args or []
        cuda_extra = list(extra.get("nvcc", [])) if isinstance(extra, dict) else []
        host_extra = list(extra.get("c", extra.get("cxx", []))) if isinstance(extra, dict) else list(extra)
        try:
            ext.sources = [source for source in ext.sources if source not in cuda_sources]
            ext.include_dirs = list(ext.include_dirs or []) + [str(include_dir)]
            ext.library_dirs = list(ext.library_dirs or []) + [str(library_dir)]
            ext.libraries = list(ext.libraries or [])
            if "cudart" not in ext.libraries:
                ext.libraries.append("cudart")
            ext.extra_compile_args = host_extra
            ext.extra_objects = list(ext.extra_objects or [])
            ext.depends = list(ext.depends or [])
            # This selects the C++ linker on Unix. Source suffixes still select
            # the C compiler for .c files, including Python's C API binding.
            ext.language = "c++"
            ext.runtime_library_dirs = list(ext.runtime_library_dirs or [])
            if compiler_type == "unix":
                ext.runtime_library_dirs.append(str(library_dir))

            include_dirs = list(dict.fromkeys(
                list(self.compiler.include_dirs or [])
                + list(type(self.compiler).include_dirs or [])
                + list(self.include_dirs or []) + ext.include_dirs
            ))
            common = [str(nvcc), "-c", "-O3", "-std=c++17", *cuda_arch_flags()]
            common += [f"-I{path}" for path in include_dirs]
            common += [f"-D{name}" if value is None else f"-D{name}={value}"
                       for name, value in ext.define_macros or []]
            common += [f"-U{name}" for name in ext.undef_macros or []]
            if compiler_type == "msvc":
                common += ["--use-local-env", "-ccbin", str(Path(self.compiler.cc).parent),
                           "-Xcompiler", "/MD", "-Xcompiler", "/EHsc"]
            else:
                common += ["-Xcompiler", "-fPIC"]
                host_compiler = os.environ.get("CUDAHOSTCXX")
                if host_compiler:
                    common += ["-ccbin", host_compiler]
            common += cuda_extra

            target_dir = Path(self.build_temp) / "cuda" / ext.name.replace(".", "_")
            self.mkpath(str(target_dir))
            for source in cuda_sources:
                digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
                output = target_dir / (Path(source).stem + "_" + digest + self.compiler.obj_extension)
                try:
                    # MSVC's spawn supplies the discovered toolchain PATH to
                    # nvcc as well as cl; all include paths are explicit above.
                    self.compiler.spawn([*common, source, "-o", str(output)])
                except Exception as error:
                    raise CompileError(f"CUDA compilation failed for {source}: {error}") from error
                ext.extra_objects.append(str(output))
                # A newly compiled CUDA object must trigger relinking even if
                # only architecture flags changed since the previous build.
                ext.depends.append(str(output))
            super().build_extension(ext)
        finally:
            for field, value in original.items():
                setattr(ext, field, value)
