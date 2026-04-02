Set-Location (Join-Path $PSScriptRoot "..")

# Stop the script immediately if any command fails
$ErrorActionPreference = "Stop"

Write-Host "=== Starting Setup ===" -ForegroundColor Cyan

# 0. Copy .env from template if needed
if (Test-Path -Path ".env.template" -PathType Leaf -and -not (Test-Path -Path ".env" -PathType Leaf)) {
    Write-Host "Creating .env from .env.template..." -ForegroundColor Cyan
    Copy-Item -Path ".env.template" -Destination ".env"
}

# 1. Remove existing virtual environment for a clean setup
if (Test-Path -Path ".venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Path ".venv" -Recurse -Force
}

# 2. Remove existing pycache directories
Write-Host "Cleaning up __pycache__ directories..." -ForegroundColor Yellow
Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# 3. Create a fresh Python 3.11 virtual environment
Write-Host "Creating fresh Python 3.11 virtual environment..." -ForegroundColor Cyan
py -3.11 -m venv .venv

# 4. Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

# 5. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# 6. Apply CMAKE_ARGS to force MSVC to compile the Linux-style C++ code
Write-Host "Setting CMAKE_ARGS and building ur_analytic_ik..." -ForegroundColor Magenta
$env:CMAKE_ARGS="-DCMAKE_CXX_FLAGS=""/D_USE_MATH_DEFINES /permissive- /std:c++17 /Zc:iso"""
pip install ur_analytic_ik --no-cache-dir

# 7. Install the rest of the dependencies
Write-Host "Installing remaining dependencies from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Your environment is ready. (Remember to run '.\.venv\Scripts\Activate.ps1' when opening a new terminal)." -ForegroundColor Cyan