$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$env:http_proxy = $null
$env:https_proxy = $null
$env:HTTP_PROXY = $null
$env:HTTPS_PROXY = $null

if (!(Test-Path ".\data\iclr2026_index.json")) {
    conda run -n aix-rag python build_index.py
}

conda run -n aix-rag streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
