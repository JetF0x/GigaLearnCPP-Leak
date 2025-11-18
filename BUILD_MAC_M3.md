# Building GigaLearnCPP on Mac M3 (Apple Silicon)

This guide provides step-by-step instructions for building GigaLearnCPP on Mac M3 (Apple Silicon / ARM64).

## Prerequisites

### 1. Install Xcode Command Line Tools

```bash
xcode-select --install
```

### 2. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. Install CMake

```bash
brew install cmake
```

### 4. Install Python 3

```bash
brew install python@3.11
```

Verify installation:
```bash
python3 --version
which python3
```

### 5. Download LibTorch for macOS ARM64

**Important**: You MUST use the ARM64 (Apple Silicon) version of LibTorch.

1. Go to: https://pytorch.org/get-started/locally/
2. Select:
   - **PyTorch Build**: Stable
   - **Your OS**: Mac
   - **Package**: LibTorch
   - **Language**: C++/Java
   - **Compute Platform**: CPU (M3 doesn't support CUDA, use Metal Performance Shaders if needed)

3. Download the **ARM64** version (look for `arm64` or `aarch64` in filename)
   - Example: `libtorch-macos-arm64-2.x.x.zip`

4. Extract to the project:
   ```bash
   cd /path/to/GigaLearnCPP-Leak
   unzip ~/Downloads/libtorch-macos-arm64-*.zip -d GigaLearnCPP/
   ```

   Or extract to a system location and note the path for later.

### 6. Install WandB (Optional, for metrics)

```bash
pip3 install wandb
```

### 7. Obtain Arena Collision Meshes

You need `.cmf` files for RocketSim. These must be dumped from Rocket League:

- Use tools like RLArenaCollisionDumper
- Place `.cmf` files in the project root directory
- At minimum, you need `soccar.cmf` for standard gameplay

## Build Instructions

### Step 1: Clone Repository with Submodules

```bash
git clone --recursive https://github.com/JetF0x/GigaLearnCPP-Leak.git
cd GigaLearnCPP-Leak
```

If already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### Step 2: Create Build Directory

```bash
mkdir build
cd build
```

### Step 3: Configure with CMake

**Option A: LibTorch in project directory** (`GigaLearnCPP/libtorch/`):
```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="$(pwd)/../GigaLearnCPP/libtorch" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64
```

**Option B: LibTorch in custom location**:
```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64
```

**Option C: Debug build**:
```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="$(pwd)/../GigaLearnCPP/libtorch" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_OSX_ARCHITECTURES=arm64
```

### Step 4: Build

```bash
# Use all CPU cores for faster build
cmake --build . --config Release -j $(sysctl -n hw.ncpu)
```

Or for Debug:
```bash
cmake --build . --config Debug -j $(sysctl -n hw.ncpu)
```

### Step 5: Verify Build

Check that the following were created:
```bash
ls -lh build/
# Should see:
# - GigaLearnBot (executable)
# - libGigaLearnCPP.dylib (shared library)
# - python_scripts/ (directory)
```

Test the executable:
```bash
./GigaLearnBot --help  # May not have --help, will error if missing .cmf files
```

## Running Training

### Step 1: Ensure Collision Meshes are Available

```bash
# From project root
ls -lh *.cmf
# Should show: soccar.cmf (and optionally others)
```

If missing, the program will crash with an error about missing collision mesh files.

### Step 2: Start Metric Receiver (Optional)

In a separate terminal:
```bash
cd build/python_scripts
python3 metric_receiver.py
```

### Step 3: Run Training

```bash
cd build
./GigaLearnBot
```

## Potential Issues and Solutions

### Issue 1: LibTorch Not Found

**Error**: `Could not find package configuration file provided by "Torch"`

**Solution**:
- Ensure you downloaded the **ARM64** version of LibTorch (not x86_64)
- Verify `CMAKE_PREFIX_PATH` points to the correct directory
- Check that `libtorch/share/cmake/Torch/TorchConfig.cmake` exists

### Issue 2: Architecture Mismatch

**Error**: `building for macOS-arm64 but attempting to link with file built for macOS-x86_64`

**Solution**:
- Re-download LibTorch, ensuring you get the ARM64 version
- Clean build directory: `rm -rf build/*` and reconfigure
- Ensure `-DCMAKE_OSX_ARCHITECTURES=arm64` is set

### Issue 3: Python Not Found

**Error**: `Could NOT find Python`

**Solution**:
```bash
# Install Python 3
brew install python@3.11

# If multiple Python versions, specify explicitly:
cmake .. \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DPython_EXECUTABLE=$(which python3) \
  -DCMAKE_OSX_ARCHITECTURES=arm64
```

### Issue 4: Missing Python Headers

**Error**: `Python.h: No such file or directory`

**Solution**:
```bash
# Reinstall Python with headers
brew reinstall python@3.11

# Or install python-dev equivalent (headers usually included in Homebrew Python)
```

### Issue 5: pybind11 Build Errors

**Error**: pybind11 compilation errors

**Solution**:
- Ensure submodules are initialized: `git submodule update --init --recursive`
- Update pybind11: `cd GigaLearnCPP/pybind11 && git pull origin master`

### Issue 6: RLBotCPP Platform Issues

**Warning**: RLBotCPP has Linux and Windows platform files, but no macOS-specific files.

**Current Status**:
- The code uses `if (UNIX)` which includes macOS
- Linux platform/socket code *may* work on macOS (both are UNIX-based)
- If you encounter issues, RLBot functionality may need macOS-specific implementation

**Workaround**:
- For training only, RLBot is not required (it's only for deployment)
- You can still train agents using the framework without RLBot integration

### Issue 7: Missing Collision Meshes

**Error**: `Failed to load collision mesh` or crash on startup

**Solution**:
- Obtain `.cmf` files using RLArenaCollisionDumper or similar tools
- Place in project root directory
- Ensure files are readable: `chmod 644 *.cmf`

### Issue 8: Metal Performance Shaders (GPU Acceleration)

**Note**: LibTorch for macOS uses **Metal Performance Shaders (MPS)** for GPU acceleration on Apple Silicon, not CUDA.

**Using MPS**:
```cpp
// In your training code, use MPS device:
cfg.device = torch::kMPS;  // For M3 GPU acceleration

// Or use auto-detect:
cfg.device = torch::kAUTO;  // Will select MPS if available
```

**Check MPS availability**:
```cpp
if (torch::mps::is_available()) {
    std::cout << "MPS is available!" << std::endl;
    cfg.device = torch::kMPS;
} else {
    std::cout << "MPS not available, using CPU" << std::endl;
    cfg.device = torch::kCPU;
}
```

**Important**: Not all LibTorch operations are MPS-optimized yet. If you encounter MPS errors, fall back to CPU:
```cpp
cfg.device = torch::kCPU;
```

### Issue 9: Slow Build Times

**Solution**:
```bash
# Use all CPU cores (M3 has 8-16 cores depending on variant)
cmake --build . -j $(sysctl -n hw.ncpu)

# Or specify number of jobs manually:
cmake --build . -j 8
```

### Issue 10: Dynamic Library Loading Issues

**Error**: `dyld: Library not loaded: @rpath/libtorch.dylib`

**Solution**:
```bash
# Set library path before running
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
./GigaLearnBot

# Or add to your ~/.zshrc or ~/.bash_profile:
echo 'export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

## Performance Considerations

### M3 Variants

- **M3**: 8-core CPU (4 performance + 4 efficiency), 10-core GPU
- **M3 Pro**: Up to 12-core CPU, up to 18-core GPU
- **M3 Max**: Up to 16-core CPU, up to 40-core GPU

### Optimal Configuration

For best training performance on M3:

```cpp
// Training config optimizations for M3
cfg.numGames = 128;  // Start conservative, increase if memory allows
cfg.timestepsPerIteration = 25000;
cfg.ppo.batchSize = 25000;
cfg.device = torch::kMPS;  // Use Metal GPU acceleration

// Enable optimizations
cfg.ppo.halfPrecisionInference = true;  // Faster inference
cfg.ppo.policy.layerNorm = true;        // Helps training stability
```

**Memory Considerations**:
- M3: 8GB or 16GB unified memory (shared with GPU)
- M3 Pro: Up to 36GB
- M3 Max: Up to 128GB

Adjust `numGames` and batch sizes based on available memory.

## Complete Build Script

Save this as `build_mac_m3.sh`:

```bash
#!/bin/bash
set -e  # Exit on error

echo "Building GigaLearnCPP for Mac M3..."

# Configuration
LIBTORCH_PATH="$PWD/GigaLearnCPP/libtorch"
BUILD_TYPE="Release"
NUM_JOBS=$(sysctl -n hw.ncpu)

# Check LibTorch exists
if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "ERROR: LibTorch not found at $LIBTORCH_PATH"
    echo "Please download ARM64 LibTorch from https://pytorch.org and extract to GigaLearnCPP/libtorch/"
    exit 1
fi

# Check for submodules
if [ ! -f "GigaLearnCPP/pybind11/CMakeLists.txt" ]; then
    echo "Initializing submodules..."
    git submodule update --init --recursive
fi

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DPython_EXECUTABLE=$(which python3)

# Build
echo "Building with $NUM_JOBS jobs..."
cmake --build . --config "$BUILD_TYPE" -j "$NUM_JOBS"

echo ""
echo "Build complete!"
echo "Executable: build/GigaLearnBot"
echo "Library: build/libGigaLearnCPP.dylib"
echo ""
echo "Before running, ensure you have .cmf collision mesh files in the project root."
echo "Run with: cd build && ./GigaLearnBot"
```

Make it executable and run:
```bash
chmod +x build_mac_m3.sh
./build_mac_m3.sh
```

## Testing the Build

Create a simple test to verify everything works:

```bash
cd build

# Check if binary is ARM64
file GigaLearnBot
# Should show: "Mach-O 64-bit executable arm64"

# Check linked libraries
otool -L GigaLearnBot | grep -E "torch|python"

# If you have .cmf files, try running
./GigaLearnBot
```

## Next Steps

1. **Place collision meshes** in project root
2. **Edit `src/ExampleMain.cpp`** to customize training
3. **Rebuild** if you make code changes: `cmake --build build -j 8`
4. **Start training** with `./build/GigaLearnBot`
5. **Monitor metrics** with WandB (if enabled)

## Notes on Metal Performance Shaders (MPS)

LibTorch on macOS uses **MPS** instead of CUDA for GPU acceleration. Key differences:

- **No CUDA support** on macOS/Apple Silicon
- **MPS is automatic** when using `torch::kMPS` device
- **Some operations** may fall back to CPU if not MPS-optimized
- **Performance** is generally good but may not match NVIDIA GPUs for large models
- **Unified memory** means RAM is shared between CPU and GPU

For pure CPU training (more compatible):
```cpp
cfg.device = torch::kCPU;
```

## Additional Resources

- **PyTorch MPS Documentation**: https://pytorch.org/docs/stable/notes/mps.html
- **LibTorch C++ Docs**: https://pytorch.org/cppdocs/
- **RocketSim**: https://github.com/ZealanL/RocketSim
- **CMake Documentation**: https://cmake.org/documentation/

## Troubleshooting Checklist

- [ ] Downloaded **ARM64** version of LibTorch (not x86_64)
- [ ] LibTorch extracted to `GigaLearnCPP/libtorch/` or custom path
- [ ] Submodules initialized (`git submodule update --init --recursive`)
- [ ] CMake 3.8+ installed (`cmake --version`)
- [ ] Python 3 installed (`python3 --version`)
- [ ] Xcode Command Line Tools installed (`xcode-select -p`)
- [ ] Collision mesh files (`.cmf`) in project root
- [ ] Using `-DCMAKE_OSX_ARCHITECTURES=arm64` in CMake config
- [ ] `CMAKE_PREFIX_PATH` points to correct LibTorch directory

---

**Last Updated**: 2024
**Tested On**: macOS 14+ (Sonoma), Apple Silicon M3
**Supported Architectures**: ARM64 only (no Rosetta x86_64 emulation)
