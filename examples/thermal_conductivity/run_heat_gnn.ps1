param(
    [ValidateSet("cpu-smoke", "cpu-smoke-full", "gpu-smoke", "gpu-smoke-full", "gpu-train", "gpu-train-full", "infer")]
    [string]$Mode = "cpu-smoke",

    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Checkpoint = "examples\thermal_conductivity\outputs\heat_gnn.pt",
    [string]$SmokeCheckpoint = "examples\thermal_conductivity\outputs\smoke_heat_gnn.pt",
    [string]$LossHistory = "examples\thermal_conductivity\outputs\loss_history.csv",
    [string]$LossPlot = "examples\thermal_conductivity\outputs\loss_history.svg",
    [string]$SmokeLossHistory = "examples\thermal_conductivity\outputs\smoke_loss_history.csv",
    [string]$SmokeLossPlot = "examples\thermal_conductivity\outputs\smoke_loss_history.svg",
    [string]$Predictions = "examples\thermal_conductivity\outputs\predictions.csv"
)

$Script = "examples\thermal_conductivity\train_infer_heat_gnn.py"

switch ($Mode) {
    "cpu-smoke" {
        & $Python $Script train `
            --epochs 1 `
            --limit-observations 8 `
            --batch-size 4 `
            --training-mode minibatch `
            --device cpu `
            --checkpoint $SmokeCheckpoint `
            --loss-history $SmokeLossHistory `
            --loss-plot $SmokeLossPlot
    }

    "cpu-smoke-full" {
        & $Python $Script train `
            --epochs 1 `
            --limit-observations 8 `
            --training-mode full-batch `
            --device cpu `
            --checkpoint $SmokeCheckpoint `
            --loss-history $SmokeLossHistory `
            --loss-plot $SmokeLossPlot
    }

    "gpu-smoke" {
        & $Python $Script train `
            --epochs 1 `
            --limit-observations 8 `
            --batch-size 4 `
            --training-mode minibatch `
            --device cuda `
            --checkpoint $SmokeCheckpoint `
            --loss-history $SmokeLossHistory `
            --loss-plot $SmokeLossPlot
    }

    "gpu-smoke-full" {
        & $Python $Script train `
            --epochs 1 `
            --limit-observations 8 `
            --training-mode full-batch `
            --device cuda `
            --checkpoint $SmokeCheckpoint `
            --loss-history $SmokeLossHistory `
            --loss-plot $SmokeLossPlot
    }

    "gpu-train" {
        & $Python $Script train `
            --device cuda `
            --epochs 20000 `
            --batch-size 16 `
            --training-mode minibatch `
            --checkpoint $Checkpoint `
            --loss-history $LossHistory `
            --loss-plot $LossPlot
    }

    "gpu-train-full" {
        & $Python $Script train `
            --device cuda `
            --epochs 20000 `
            --training-mode full-batch `
            --checkpoint $Checkpoint `
            --loss-history $LossHistory `
            --loss-plot $LossPlot
    }

    "infer" {
        & $Python $Script infer `
            --checkpoint $Checkpoint `
            --output $Predictions `
            --batch-size 16 `
            --device cuda
    }
}
