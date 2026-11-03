# Шаг 1. Чтение Experiment_Log (1).xls через локальный Excel (без сети).
# Результат: data\interim\adapt_log_sheet1.tsv (табуляция, UTF-8 без BOM).
$ErrorActionPreference = "Stop"
$root    = "D:\Satelite Second Lab"
$src     = Join-Path $root "Experiment_Log (1).xls"
$interim = Join-Path $root "data\interim"
New-Item -ItemType Directory -Force -Path $interim | Out-Null
$dst = Join-Path $interim "adapt_log_sheet1.tsv"

$x = New-Object -ComObject Excel.Application
$x.Visible = $false
$x.DisplayAlerts = $false
$wb = $x.Workbooks.Open($src, 0, $true)   # только чтение, исходник не меняется
$ws = $wb.Worksheets.Item(1)
$ur = $ws.UsedRange
$data = $ur.Value2
$rows = $ur.Rows.Count
$cols = $ur.Columns.Count

$sb = New-Object System.Text.StringBuilder
for ($r = 1; $r -le $rows; $r++) {
    $cells = @()
    for ($c = 1; $c -le $cols; $c++) {
        $v = $data.GetValue($r, $c)
        if ($null -eq $v) { $v = "" }
        $cells += ([string]$v).Replace("`t", " ").Replace("`r", " ").Replace("`n", " ")
    }
    [void]$sb.AppendLine(($cells -join "`t"))
}
[System.IO.File]::WriteAllText($dst, $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))

$wb.Close($false)
$x.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($x) | Out-Null
Write-Output "OK: $dst ($rows x $cols)"
