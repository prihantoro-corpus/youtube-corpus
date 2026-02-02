# ===============================
# Dictionary Lab Launcher
# ===============================

$projectPath = "C:\Users\priha\Documents\dictionary-lab"

Write-Host "📁 Checking project directory..."
if (!(Test-Path $projectPath)) {
    Write-Host "❌ Project folder not found: $projectPath"
    Pause
    exit
}

Set-Location $projectPath
Write-Host "✅ Working directory set to $projectPath"

# -------------------------------
# Check Python
# -------------------------------
Write-Host "🐍 Checking Python..."
$python = Get-Command python -ErrorAction SilentlyContinue
if (!$python) {
    Write-Host "❌ Python not found. Please install Python first."
    Pause
    exit
}
Write-Host "✅ Python found"

# -------------------------------
# Check Internet Connection
# -------------------------------
function Test-Internet {
    try {
        Test-Connection -ComputerName "pypi.org" -Count 1 -Quiet
    } catch {
        return $false
    }
}

# -------------------------------
# Check Streamlit
# -------------------------------
Write-Host "📦 Checking Streamlit..."
$streamlit = python -m pip show streamlit 2>$null
if (!$streamlit) {
    Write-Host "⚠️ Streamlit not installed."

    if (!(Test-Internet)) {
        Write-Host "🌐 No internet connection detected."
        Write-Host "👉 Please connect to the internet and run again."
        Pause
        exit
    }

    Write-Host "🌐 Internet OK. Installing requirements..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
}

# -------------------------------
# Final check: requirements
# -------------------------------
Write-Host "📦 Ensuring all requirements are installed..."
if (Test-Internet) {
    python -m pip install -r requirements.txt
} else {
    Write-Host "⚠️ No internet. Skipping dependency install."
}

# -------------------------------
# Run App
# -------------------------------
Write-Host "🚀 Launching Dictionary Lab..."
python -m streamlit run app.py

Pause
