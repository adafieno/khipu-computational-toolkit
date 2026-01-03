#!/usr/bin/env pwsh
# Comprehensive script testing with UTF-8 encoding

$env:PYTHONIOENCODING="utf-8"
$ErrorActionPreference = "Continue"

$scripts = @(
    # Extraction scripts
    "extract_color_data.py",
    "extract_cord_hierarchy.py", 
    "extract_knot_data.py",
    
    # Test scripts
    "test_summation_hypotheses.py",
    "test_alternative_summation.py",
    "test_hierarchical_summation.py",
    "test_color_hypotheses.py",
    
    # Analysis scripts
    "analyze_high_match_khipus.py",
    "analyze_geographic_correlations.py",
    "analyze_information_capacity.py",
    "analyze_robustness.py",
    "analyze_variance.py",
    
    # Clustering/Graph scripts
    "cluster_khipus.py",
    "compute_graph_similarities.py",
    "classify_khipu_function.py"
)

$results = @()

Write-Host "=" * 80
Write-Host "TESTING ALL SCRIPTS"
Write-Host "=" * 80
Write-Host ""

foreach ($script in $scripts) {
    Write-Host "Testing: $script" -ForegroundColor Cyan
    Write-Host "-" * 80
    
    $startTime = Get-Date
    $output = python "scripts\$script" 2>&1
    $exitCode = $LASTEXITCODE
    $duration = (Get-Date) - $startTime
    
    $status = if ($exitCode -eq 0) { "✓ PASS" } else { "✗ FAIL" }
    $color = if ($exitCode -eq 0) { "Green" } else { "Red" }
    
    Write-Host "$status (${duration}s, exit code: $exitCode)" -ForegroundColor $color
    
    if ($exitCode -ne 0) {
        # Show error details
        $errors = $output | Select-String -Pattern "(Error|Traceback|File.*line)"
        if ($errors) {
            Write-Host "  Errors:" -ForegroundColor Yellow
            $errors | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" }
        }
    }
    
    $results += [PSCustomObject]@{
        Script = $script
        Status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }
        ExitCode = $exitCode
        Duration = $duration.TotalSeconds
    }
    
    Write-Host ""
}

Write-Host "=" * 80
Write-Host "SUMMARY"
Write-Host "=" * 80

$passed = ($results | Where-Object { $_.Status -eq "PASS" }).Count
$failed = ($results | Where-Object { $_.Status -eq "FAIL" }).Count
$total = $results.Count

Write-Host "Total: $total | Passed: $passed | Failed: $failed"
Write-Host ""

if ($failed -gt 0) {
    Write-Host "Failed scripts:" -ForegroundColor Red
    $results | Where-Object { $_.Status -eq "FAIL" } | ForEach-Object {
        Write-Host "  - $($_.Script) (exit code: $($_.ExitCode))"
    }
}

# Export results
$results | Export-Csv -Path "test_results.csv" -NoTypeInformation
Write-Host ""
Write-Host "Results exported to test_results.csv"
