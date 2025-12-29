#!/bin/bash
# Script to install Cursor extensions in dev container
# Run this manually after container starts if extensions weren't installed automatically

EXTENSIONS=(
    "james-yu.latex-workshop"
    "valentjn.vscode-ltex"
    "streetsidesoftware.code-spell-checker"
    "streetsidesoftware.code-spell-checker-russian"
    "tecosaur.latex-utilities"
    "quarto.quarto"
)

echo "Installing Cursor extensions..."

for ext in "${EXTENSIONS[@]}"; do
    echo "Installing $ext..."
    # Try cursor CLI first, fall back to code
    code --install-extension "$ext" --force --verbose
done

echo "Done! Please reload the window (Ctrl+Shift+P -> 'Developer: Reload Window')"

