#!/bin/bash
# Setup script for JARVIS Prime VS Code environment

set -e

echo "🔧 Setting up JARVIS Prime for VS Code..."
echo ""

# Get Python path
PYTHON_PATH=$(which python)
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

echo "📍 Python Configuration:"
echo "   Python: $PYTHON_PATH"
echo "   Version: $PYTHON_VERSION"
echo "   Site packages: $SITE_PACKAGES"
echo ""

# Verify packages are installed
echo "📦 Verifying dependencies..."
python -c "import torch; print(f'   ✅ torch {torch.__version__}')" || echo "   ❌ torch not found"
python -c "import transformers; print(f'   ✅ transformers {transformers.__version__}')" || echo "   ❌ transformers not found"
python -c "import peft; print(f'   ✅ peft {peft.__version__}')" || echo "   ❌ peft not found"
echo ""

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Update VS Code settings with correct Python path
echo "⚙️  Updating VS Code settings..."
cat > .vscode/settings.json << EOF
{
  "python.defaultInterpreterPath": "$PYTHON_PATH",
  "python.analysis.extraPaths": [
    "\${workspaceFolder}/jarvis_prime",
    "$SITE_PACKAGES"
  ],
  "python.analysis.typeCheckingMode": "basic",
  "python.terminal.activateEnvironment": true,
  "terminal.integrated.env.osx": {
    "TRANSFORMERS_NO_TF": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3"
  },
  "terminal.integrated.env.linux": {
    "TRANSFORMERS_NO_TF": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "isort.args": ["--profile", "black"],
  "black-formatter.args": ["--line-length", "100"],
  "python.analysis.diagnosticSeverityOverrides": {
    "reportMissingImports": "none"
  },
  "basedpyright.analysis.diagnosticSeverityOverrides": {
    "reportMissingImports": "none",
    "reportMissingModuleSource": "none"
  }
}
EOF

echo "   ✅ .vscode/settings.json updated"

# Update pyrightconfig.json
echo "⚙️  Updating pyrightconfig.json..."
cat > pyrightconfig.json << 'EOF'
{
  "pythonVersion": "3.10",
  "pythonPlatform": "Darwin",
  "typeCheckingMode": "basic",
  "reportMissingImports": "none",
  "reportMissingModuleSource": "none",
  "reportMissingTypeStubs": false,
  "useLibraryCodeForTypes": true,
  "include": [
    "jarvis_prime",
    "examples"
  ],
  "exclude": [
    "**/node_modules",
    "**/__pycache__",
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "*.egg-info"
  ]
}
EOF

echo "   ✅ pyrightconfig.json updated"
echo ""

# Test import
echo "🧪 Testing imports..."
TRANSFORMERS_NO_TF=1 python -c "
import jarvis_prime
from jarvis_prime.configs import LlamaPresets
from jarvis_prime.models import LlamaModel
print('   ✅ All imports successful!')
" || echo "   ❌ Import test failed"

echo ""
echo "✅ VS Code setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Reload VS Code window (Cmd+Shift+P > 'Developer: Reload Window')"
echo "   2. Select Python interpreter: $PYTHON_PATH"
echo "   3. Type checker warnings should now be resolved"
echo ""
