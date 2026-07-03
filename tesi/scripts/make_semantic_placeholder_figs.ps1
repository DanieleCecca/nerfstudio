$ErrorActionPreference = "Stop"

$dir = Join-Path $PSScriptRoot "..\figures"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

# 1x1 transparent PNG (minimal valid PNG bytes)
$b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ZkAAAAASUVORK5CYII="
$bytes = [Convert]::FromBase64String($b64)

$names = @(
  "semantic_pipeline_text_box_mask_3d.png",
  "grounding_dino_architecture_overview.png",
  "grounding_dino_subsentence_attention_mask.png",
  "sam2_interactive_masklet_timeline.png",
  "sam2_memory_bank_attention.png"
)

foreach ($n in $names) {
  $out = Join-Path $dir $n
  [IO.File]::WriteAllBytes($out, $bytes)
}

Write-Host "Wrote $($names.Count) placeholder PNGs to: $dir"
