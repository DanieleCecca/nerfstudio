$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$dir = "C:\Users\cecca\Desktop\nerfstudio\tesi\figures"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$names = @(
  "semantic_pipeline_text_box_mask_3d.png",
  "grounding_dino_architecture_overview.png",
  "grounding_dino_subsentence_attention_mask.png",
  "sam2_interactive_masklet_timeline.png",
  "sam2_memory_bank_attention.png"
)

foreach ($n in $names) {
  $bmp = New-Object System.Drawing.Bitmap 16, 16
  for ($x = 0; $x -lt 16; $x++) {
    for ($y = 0; $y -lt 16; $y++) {
      $bmp.SetPixel($x, $y, [System.Drawing.Color]::White)
    }
  }
  $outPath = Join-Path $dir $n
  $bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
  $bmp.Dispose()
}

Write-Host "Wrote $($names.Count) valid PNG placeholders in $dir"
