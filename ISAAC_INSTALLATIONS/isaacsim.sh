#!/bin/bash

# Isaac Sim UV Environment Wrapper
# This script sets up Isaac Sim 5.0.0 source build with UV virtual environment

set -e

# ============================================================================
# ARGUMENT PARSING AND CONFIGURATION
# ============================================================================

# Parse command line arguments
if [ $# -ge 1 ] && [ "$1" != "" ]; then
    ISAACSIM_PATH="$1"
    echo "🔧 Using custom HOME path: $ISAACSIM_PATH"
else
    ISAACSIM_PATH="$HOME/IsaacSim"
    echo "🔧 Using default HOME path: $ISAACSIM_PATH"
fi

# Validate ISAACSIM_PATH exists
if [ ! -d "$ISAACSIM_PATH" ]; then
    echo "❌ Error: ISAACSIM_PATH path does not exist: $ISAACSIM_PATH"
    exit 1
fi

# Print usage information
print_usage() {
    echo "Usage: $0 [ISAACSIM_PATH]"
    echo ""
    echo "Arguments:"
    echo "  ISAACSIM_PATH    - Custom home directory path (default: \$HOME)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use defaults"
    echo "  $0 /custom/home                       # Custom home, default UV env"
    echo "  $0 /custom/home /custom/venv/my_env   # Custom home and UV env"
    echo ""
}

# Show usage if --help is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    print_usage
    exit 0
fi

echo "=== Isaac Sim 5.0.0 UV Environment Setup ==="
echo "🏠 Home Path: $ISAACSIM_PATH"
echo ""

# ============================================================================
# 1. LOAD ISAAC SIM ENVIRONMENT
# ============================================================================
echo "🔧 Loading Isaac Sim environment from ~/.bash_isaacsim..."

# Function to create .bash_isaacsim if it doesn't exist
create_bash_isaacsim() {
    echo "🔧 Creating $HOME/.bash_isaacsim..."
    cat > "$HOME/.bash_isaacsim" << EOF
# ~/.bash_isaacsim - Isaac Sim 5.0.0 Environment Variables
# This file is sourced by ~/.bashrc

# Isaac Sim path configuration
export ISAAC_SIM_PATH="$ISAACSIM_PATH/_build/linux-x86_64/release"

# Check if Isaac Sim build exists
if [ ! -d "\$ISAAC_SIM_PATH" ]; then
    echo "⚠️  Warning: Isaac Sim build not found at \$ISAAC_SIM_PATH"
    return 1
fi

# Core Isaac Sim environment variables
export ISAAC_PATH="\$ISAAC_SIM_PATH"
export CARB_APP_PATH="\$ISAAC_SIM_PATH/kit"
export EXP_PATH="\$ISAAC_SIM_PATH/apps"
export RESOURCE_NAME="IsaacSim"

# Library paths for Isaac Sim
export LD_LIBRARY_PATH="\$ISAAC_SIM_PATH:\$ISAAC_SIM_PATH/kit:\$ISAAC_SIM_PATH/kit/kernel/plugins:\$ISAAC_SIM_PATH/kit/libs/iray:\$ISAAC_SIM_PATH/kit/plugins:\$ISAAC_SIM_PATH/kit/plugins/bindings-python:\$ISAAC_SIM_PATH/kit/plugins/carb_gfx:\$ISAAC_SIM_PATH/kit/plugins/rtx:\$ISAAC_SIM_PATH/kit/plugins/gpu.foundation:\${LD_LIBRARY_PATH:-}"

# WAR for missing libcarb.so
export LD_PRELOAD="\$ISAAC_SIM_PATH/kit/libcarb.so"

# Python paths for Isaac Sim modules
export PYTHONPATH="\$ISAAC_SIM_PATH/python_packages:\$ISAAC_SIM_PATH/exts/isaacsim.simulation_app:\$ISAAC_SIM_PATH/extsDeprecated/omni.isaac.kit:\$ISAAC_SIM_PATH/kit/kernel/py:\$ISAAC_SIM_PATH/kit/plugins/bindings-python:\${PYTHONPATH:-}"

# Isaac Sim aliases
alias isaacsim_gui="cd \$ISAAC_SIM_PATH && ./isaac-sim.sh"
alias isaacsim_headless="cd \$ISAAC_SIM_PATH && ./kit/python/bin/python3"
alias isaacsim_version="cat \$ISAAC_SIM_PATH/VERSION"

# Quick Isaac Sim test function
test_isaac_sim() {
    python -c "
try:
    from isaacsim import SimulationApp
    print('✅ Isaac Sim import successful')
except ImportError as e:
    print(f'❌ Isaac Sim import failed: {e}')
"
}
EOF
    echo "✅ $HOME/.bash_isaacsim created"
}

# Function to add sourcing to .bashrc if not already present
setup_bashrc_sourcing() {
    if [ -f "$HOME/.bashrc" ]; then
        # Check if .bash_isaacsim is already sourced in .bashrc
        if ! grep -q "bash_isaacsim" "$HOME/.bashrc"; then
            echo "🔧 Adding .bash_isaacsim sourcing to $HOME/.bashrc..."
            cat >> "$HOME/.bashrc" << EOF

# Isaac Sim 5.0.0 Environment
if [ -f "$HOME/.bash_isaacsim" ]; then
    source "$HOME/.bash_isaacsim"
fi
EOF
            echo "✅ $HOME/.bashrc updated to source .bash_isaacsim"
        else
            echo "✅ $HOME/.bashrc already sources .bash_isaacsim"
        fi
    else
        echo "⚠️  $HOME/.bashrc not found, creating it..."
        cat > "$HOME/.bashrc" << EOF
# Isaac Sim 5.0.0 Environment
if [ -f "$HOME/.bash_isaacsim" ]; then
    source "$HOME/.bash_isaacsim"
fi
EOF
        echo "✅ $HOME/.bashrc created with .bash_isaacsim sourcing"
    fi
}

# Check if .bash_isaacsim exists, create if not
if [ ! -f "$HOME/.bash_isaacsim" ]; then
    echo "📝 $HOME/.bash_isaacsim not found, creating it..."
    create_bash_isaacsim
    setup_bashrc_sourcing
else
    echo "✅ $HOME/.bash_isaacsim already exists"
fi

# Source the Isaac Sim environment
if [ -f "$HOME/.bash_isaacsim" ]; then
    source "$HOME/.bash_isaacsim"
    echo "✅ Isaac Sim environment loaded"
else
    echo "❌ Error: Failed to create $HOME/.bash_isaacsim"
    exit 1
fi

echo "✅ Inside activated UV environment: $VIRTUAL_ENV"

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: UV virtual environment not activated properly"
    exit 1
fi
# ============================================================================
# 3. ENSURE ISAAC SIM COMPATIBILITY WITH UV
# ============================================================================
echo "🔧 Ensuring Isaac Sim compatibility with UV environment..."

# Store UV Python info
UV_PYTHON_EXE="$VIRTUAL_ENV/bin/python"
UV_PYTHON_VERSION=$($UV_PYTHON_EXE --version 2>&1)

# Re-export Isaac Sim paths to ensure they're available in UV environment
export PYTHONPATH="$ISAAC_SIM_PATH/python_packages:$ISAAC_SIM_PATH/exts/isaacsim.simulation_app:$ISAAC_SIM_PATH/extsDeprecated/omni.isaac.kit:$ISAAC_SIM_PATH/kit/kernel/py:$ISAAC_SIM_PATH/kit/plugins/bindings-python:${PYTHONPATH:-}"

echo "✅ Isaac Sim paths configured for UV environment"

# ============================================================================
# 4. CREATE ISAAC SIM WRAPPER FOR UV
# ============================================================================
echo "🔧 Creating Isaac Sim wrapper for UV environment..."

create_isaacsim_wrapper() {
    cat > "$VIRTUAL_ENV/bin/isaacsim" << EOF
#!/bin/bash
# Isaac Sim 5.0.0 launcher with UV environment support

# Source Isaac Sim environment
if [ -f "$HOME/.bash_isaacsim" ]; then
    source "$HOME/.bash_isaacsim" >/dev/null 2>&1
fi

# Parse arguments and launch Isaac Sim
case "\${1:-gui}" in
    "gui"|"")
        echo "🚀 Launching Isaac Sim 5.0.0 GUI..."
        cd "\$ISAAC_SIM_PATH"
        exec "\$ISAAC_SIM_PATH/isaac-sim.sh" "\${@:2}"
        ;;
    "--headless")
        echo "🚀 Launching Isaac Sim 5.0.0 headless with UV environment..."
        cd "\$ISAAC_SIM_PATH"
        exec "\$VIRTUAL_ENV/bin/python" "\${@:2}"
        ;;
    "--version")
        echo "Isaac Sim \$(cat \$ISAAC_SIM_PATH/VERSION)"
        echo "UV Environment: \$VIRTUAL_ENV"
        echo "Python: \$(python --version)"
        ;;
    *)
        echo "🚀 Launching Isaac Sim with custom arguments..."
        cd "\$ISAAC_SIM_PATH"
        exec "\$ISAAC_SIM_PATH/isaac-sim.sh" "\$@"
        ;;
esac
EOF
    chmod +x "$VIRTUAL_ENV/bin/isaacsim"
}

create_isaacsim_wrapper
echo "✅ Isaac Sim wrapper created"

# ============================================================================
# 5. CREATE ENHANCED TEST SCRIPT
# ============================================================================
echo "🔧 Creating Isaac Sim test script..."

create_isaac_sim_test() {
    cat > "$VIRTUAL_ENV/bin/test_isaac_sim.py" << EOF
#!/usr/bin/env python3
"""Test Isaac Sim 5.0.0 import and basic functionality with UV environment"""

import sys
import os
import subprocess

# Isaac Sim configuration
ISAAC_SIM_PATH = "$ISAACSIM_PATH/_build/linux-x86_64/release"

# Set required environment variables
os.environ['ISAAC_PATH'] = ISAAC_SIM_PATH
os.environ['CARB_APP_PATH'] = f"{ISAAC_SIM_PATH}/kit"
os.environ['EXP_PATH'] = f"{ISAAC_SIM_PATH}/apps"

def test_isaac_sim_import():
    """Test Isaac Sim import and basic functionality"""
    print("🔧 Testing Isaac Sim 5.0.0 import...")
    
    try:
        from isaacsim import SimulationApp
        print("✅ Isaac Sim import successful")
        
        # Test basic initialization
        print("🔧 Testing Isaac Sim initialization...")
        config = {"headless": True}
        app = SimulationApp(config)
        
        import omni.kit.app
        version = omni.kit.app.get_app().get_kit_version()
        print(f"✅ Isaac Sim Version: {version}")
        
        app.close()
        print("✅ Isaac Sim test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Isaac Sim test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_uv_pip():
    """Test UV pip functionality"""
    print("🔧 Testing UV pip functionality...")
    
    try:
        # Test UV pip list
        result = subprocess.run(['uv', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ UV pip list works")
        else:
            print("❌ UV pip list failed")
            return False
            
        # Test torch availability
        if 'torch' in result.stdout:
            print("✅ PyTorch available in UV environment")
        else:
            print("⚠️  PyTorch not found in UV environment")
            
        return True
        
    except Exception as e:
        print(f"❌ UV pip test failed: {e}")
        return False

def test_environment():
    """Test environment variables and paths"""
    print("🔧 Testing environment setup...")
    
    # Check virtual environment
    venv = os.environ.get('VIRTUAL_ENV', '')
    if venv:
        print("✅ UV virtual environment active")
    else:
        print("❌ UV virtual environment not active")
        return False
    
    # Check Isaac Sim paths
    isaac_path = os.environ.get('ISAAC_PATH', '')
    if isaac_path:
        print("✅ ISAAC_PATH set")
    else:
        print("❌ ISAAC_PATH not set")
        return False
    
    # Check PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    if 'python_packages' in pythonpath:
        print("✅ Isaac Sim in PYTHONPATH")
    else:
        print("❌ Isaac Sim not in PYTHONPATH")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Isaac Sim UV Environment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment),
        ("UV Pip Functionality", test_uv_pip),
        ("Isaac Sim Import", test_isaac_sim_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Isaac Sim UV environment is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
    chmod +x "$VIRTUAL_ENV/bin/test_isaac_sim.py"
}

create_isaac_sim_test
echo "✅ Isaac Sim test script created"

# ============================================================================
# 6. VERIFICATION AND STATUS
# ============================================================================
echo ""
echo "=== Environment Status ==="
echo "✅ Isaac Sim Version: $(cat $ISAAC_SIM_PATH/VERSION)"
echo "✅ UV Environment: $VIRTUAL_ENV"
echo "✅ Python: $UV_PYTHON_VERSION"
echo "✅ Isaac Sim Path: $ISAAC_SIM_PATH"

# Test UV pip functionality
echo ""
echo "=== UV Pip Status ==="
if command -v uv >/dev/null 2>&1; then
    echo "✅ UV available: $(uv --version)"
    if uv pip list 2>/dev/null | grep -q "torch"; then
        torch_version=$(uv pip show torch 2>/dev/null | grep "Version:" | awk '{print $2}')
        echo "✅ PyTorch found: $torch_version"
    else
        echo "⚠️  PyTorch not found in UV environment"
    fi
else
    echo "❌ UV not available"
fi

# Test direct Isaac Sim import
echo ""
echo "=== Isaac Sim Import Test ==="
python -c "
try:
    from isaacsim import SimulationApp
    print('✅ Direct Isaac Sim import successful')
except ImportError as e:
    print(f'❌ Direct Isaac Sim import failed: {e}')
" 2>/dev/null

# ============================================================================
# 7. FINAL SETUP AND USAGE INFO
# ============================================================================
echo ""
echo "=== Available Commands ==="
echo "  isaacsim               - Launch Isaac Sim GUI"
echo "  isaacsim --headless    - Launch Isaac Sim headless mode"
echo "  isaacsim --version     - Show version information"
echo "  test_isaac_sim.py      - Run comprehensive test suite"
echo "  python                 - UV environment Python with Isaac Sim"
echo "  uv pip                 - UV package manager"
echo ""

# Run comprehensive test
echo "=== Running Comprehensive Test ==="
if python "$VIRTUAL_ENV/bin/test_isaac_sim.py"; then
    echo ""
    echo "🎉 SUCCESS: Isaac Sim 5.0.0 UV environment is fully configured!"
    echo ""
    echo "You can now:"
    echo "  • Use 'uv pip' for package management"
    echo "  • Import Isaac Sim directly: from isaacsim import SimulationApp"
    echo "  • Launch Isaac Sim with 'isaacsim' command"
    echo "  • Use PyTorch and other UV packages with Isaac Sim"
else
    echo ""
    echo "⚠️  Setup completed but some tests failed."
    echo "Isaac Sim may still work for basic usage."
fi

# Deactivate and reactivate the UV environment to ensure all changes take effect
echo "🔄 Refreshing UV environment..."
if [ -n "$VIRTUAL_ENV" ] && command -v deactivate >/dev/null 2>&1; then
    deactivate
fi
source "$VIRTUAL_ENV/bin/activate"
echo "✅ UV environment refreshed"