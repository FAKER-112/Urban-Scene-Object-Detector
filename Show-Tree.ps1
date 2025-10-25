# Show-Tree.ps1
# Displays a folder tree (like tree /F) while excluding specific folders and limiting depth.

param(
    [string]$Path = ".",
    [string[]]$Exclude = @("mlruns", "runs", "logs",'data', 'artifacts'),  # folders to exclude
    [int]$Depth = [int]::MaxValue   # by default, show full depth
)

# Resolve full root path
$Root = (Resolve-Path $Path).Path

function Show-Tree($Folder, $Level) {
    # Stop if depth exceeded
    if ($Level -ge $Depth) { return }

    # Print folder name (except for root at level -1)
    if ($Level -ge 0) {
        $Indent = "  " * [Math]::Max(0, $Level)
        Write-Host "$Indent+ $(Split-Path $Folder -Leaf)"
    }

    # Print files in current folder
    Get-ChildItem -Path $Folder -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            $Indent = "  " * ([Math]::Max(0, $Level + 1))
            Write-Host "$Indent- $($_.Name)"
        }

    # Recurse into subfolders (if not excluded)
    Get-ChildItem -Path $Folder -Directory -ErrorAction SilentlyContinue |
        Where-Object { $Exclude -notcontains $_.Name } |
        ForEach-Object {
            Show-Tree $_.FullName ($Level + 1)
        }
}

# Start recursion (level -1 = root)
Show-Tree $Root -1
