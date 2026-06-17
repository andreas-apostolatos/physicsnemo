# Direct commands to copy and run from the repository root:
#
# Setup:
# python -m venv .venv
# .\.venv\Scripts\python.exe -m pip install pandas torch
#
# CPU smoke test:
# powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode cpu-smoke
#
# CPU full-batch smoke test:
# powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode cpu-smoke-full
#
# GPU smoke test:
# powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-smoke
#
# GPU full-batch smoke test:
# powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-smoke-full
#
# Full GPU training, mini-batch:
# powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-train
#
# Full GPU training, MATLAB-style full-batch:
# powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-train-full
#
# Inference after training:
# powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode infer
#
# Main outputs:
# examples\thermal_conductivity\outputs\heat_gnn.pt
# examples\thermal_conductivity\outputs\loss_history.csv
# examples\thermal_conductivity\outputs\loss_history.svg
# examples\thermal_conductivity\outputs\predictions.csv

param(
    [ValidateSet("print", "setup", "cpu-smoke", "cpu-smoke-full", "gpu-smoke", "gpu-smoke-full", "gpu-train", "gpu-train-full", "infer", "all-smoke")]
    [string]$Mode = "print"
)

$commands = [ordered]@{
    Setup = @(
        "python -m venv .venv",
        ".\.venv\Scripts\python.exe -m pip install pandas torch"
    )
    CpuSmoke = @(
        "powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode cpu-smoke"
    )
    CpuSmokeFullBatch = @(
        "powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode cpu-smoke-full"
    )
    GpuSmoke = @(
        "powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-smoke"
    )
    GpuSmokeFullBatch = @(
        "powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-smoke-full"
    )
    GpuTrainMinibatch = @(
        "powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-train"
    )
    GpuTrainFullBatch = @(
        "powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode gpu-train-full"
    )
    Infer = @(
        "powershell -ExecutionPolicy Bypass -File examples\thermal_conductivity\run_heat_gnn.ps1 -Mode infer"
    )
}

function Show-Commands {
    foreach ($section in $commands.Keys) {
        Write-Host ""
        Write-Host "[$section]"
        foreach ($command in $commands[$section]) {
            Write-Host $command
        }
    }
    Write-Host ""
    Write-Host "Outputs:"
    Write-Host "examples\thermal_conductivity\outputs\heat_gnn.pt"
    Write-Host "examples\thermal_conductivity\outputs\loss_history.csv"
    Write-Host "examples\thermal_conductivity\outputs\loss_history.svg"
    Write-Host "examples\thermal_conductivity\outputs\predictions.csv"
}

function Invoke-StepCommands($stepCommands) {
    foreach ($command in $stepCommands) {
        Write-Host ""
        Write-Host "> $command"
        Invoke-Expression $command
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}

switch ($Mode) {
    "print" {
        Show-Commands
    }
    "setup" {
        Invoke-StepCommands $commands.Setup
    }
    "cpu-smoke" {
        Invoke-StepCommands $commands.CpuSmoke
    }
    "cpu-smoke-full" {
        Invoke-StepCommands $commands.CpuSmokeFullBatch
    }
    "gpu-smoke" {
        Invoke-StepCommands $commands.GpuSmoke
    }
    "gpu-smoke-full" {
        Invoke-StepCommands $commands.GpuSmokeFullBatch
    }
    "gpu-train" {
        Invoke-StepCommands $commands.GpuTrainMinibatch
    }
    "gpu-train-full" {
        Invoke-StepCommands $commands.GpuTrainFullBatch
    }
    "infer" {
        Invoke-StepCommands $commands.Infer
    }
    "all-smoke" {
        Invoke-StepCommands $commands.Setup
        Invoke-StepCommands $commands.CpuSmoke
        Invoke-StepCommands $commands.GpuSmoke
    }
}
