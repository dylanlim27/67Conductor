import ctypes
import os
import platform
import subprocess
import sys

print("test_call.py: Script loaded", flush=True)

def main():
    haskell_dir = os.path.dirname(os.path.abspath(__file__))
    hs_source = os.path.join(haskell_dir, "HSLib.hs")
    
    system = platform.system()
    if system == "Windows":
        lib_name = "HSLib.dll"
        compile_cmd = ["ghc", "-shared", "-o", lib_name, "HSLib.hs", "StartEnd.c", "HSLib.def"]
    elif system == "Darwin":
        lib_name = "libHSLib.dylib"
        compile_cmd = ["ghc", "-dynamic", "-shared", "-fPIC", "-o", lib_name, "HSLib.hs"]
    else:  # Linux/Unix
        lib_name = "libHSLib.so"
        compile_cmd = ["ghc", "-dynamic", "-shared", "-fPIC", "-o", lib_name, "HSLib.hs"]

    lib_path = os.path.join(haskell_dir, lib_name)

    # 1. Compile the Haskell library if it doesn't exist
    if not os.path.exists(lib_path):
        print(f"Compiling Haskell FFI library: {' '.join(compile_cmd)}...", flush=True)
        try:
            subprocess.run(compile_cmd, cwd=haskell_dir, check=True)
            print("Compilation successful.", flush=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error compiling Haskell library: {e}", flush=True)
            print("Make sure 'ghc' is installed and on your PATH.", flush=True)
            sys.exit(1)

    # 2. Load the library using ctypes
    print(f"Loading shared library from: {lib_path}", flush=True)
    try:
        if system == "Windows":
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(haskell_dir)
        lib = ctypes.CDLL(lib_path)
        print("Library loaded successfully.", flush=True)
    except Exception as e:
        print(f"Failed to load shared library: {e}", flush=True)
        print("Note: On Windows, GHC FFI DLLs may require GHC's runtime DLLs on the PATH.", flush=True)
        sys.exit(1)

    # Define ctypes signatures for our exported Haskell functions
    lib.hs_add.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.hs_add.restype = ctypes.c_int

    lib.hs_reverse.argtypes = [ctypes.c_char_p]
    lib.hs_reverse.restype = ctypes.c_char_p

    lib.hs_free.argtypes = [ctypes.c_char_p]
    lib.hs_free.restype = None

    # 3. Call hs_add
    val1, val2 = 10, 20
    result_add = lib.hs_add(val1, val2)
    print(f"Haskell hs_add({val1}, {val2}) -> {result_add}", flush=True)

    # 4. Call hs_reverse
    test_str = b"Hello from Python!"
    result_ptr = lib.hs_reverse(test_str)
    
    if result_ptr:
        reversed_str = ctypes.string_at(result_ptr).decode('utf-8')
        print(f"Haskell hs_reverse({test_str.decode('utf-8')!r}) -> {reversed_str!r}", flush=True)
        lib.hs_free(result_ptr)
    else:
        print("Haskell hs_reverse returned a null pointer.", flush=True)

if __name__ == "__main__":
    main()