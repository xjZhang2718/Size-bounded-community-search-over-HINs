Get-Content "Amazon_queries.txt" | ForEach-Object {
    $parts = $_ -split ' '
    Write-Host "python .\enumerateVertex.py $_"
    python .\enumerateVertex.py @parts
}
