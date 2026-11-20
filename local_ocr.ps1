param([Parameter(Mandatory=$true)][string] $ImagePath)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$fileType = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$streamType = [Windows.Storage.Streams.IRandomAccessStreamWithContentType, Windows.Storage.Streams, ContentType=WindowsRuntime]
$decoderType = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$bitmapType = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$resultType = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType=WindowsRuntime]
$engineType = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethodDefinition -and
    $_.GetParameters().Count -eq 1 -and $_.GetGenericArguments().Count -eq 1
} | Select-Object -First 1

function Resolve-OcrOperation($operation, [type] $type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($operation))
    return $task.GetAwaiter().GetResult()
}

$engine = $engineType::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    @{status = 'unavailable'} | ConvertTo-Json -Compress
    exit
}

$stream = $null
$bitmap = $null
try {
    $file = Resolve-OcrOperation ($fileType::GetFileFromPathAsync($ImagePath)) $fileType
    $stream = Resolve-OcrOperation ($file.OpenReadAsync()) $streamType
    $decoder = Resolve-OcrOperation ($decoderType::CreateAsync($stream)) $decoderType
    # Bound decoded size even when the compressed upload is small.
    if ($decoder.PixelWidth -gt $engineType::MaxImageDimension -or
        $decoder.PixelHeight -gt $engineType::MaxImageDimension -or
        ([long] $decoder.PixelWidth * $decoder.PixelHeight) -gt 16000000) {
        @{status = 'dimensions'} | ConvertTo-Json -Compress
        exit
    }
    $bitmap = Resolve-OcrOperation ($decoder.GetSoftwareBitmapAsync()) $bitmapType
    $result = Resolve-OcrOperation ($engine.RecognizeAsync($bitmap)) $resultType
    @{status = 'ok'; text = ($result.Lines.Text -join "`n"); language = $engine.RecognizerLanguage.LanguageTag} |
        ConvertTo-Json -Compress
} catch {
    @{status = 'unreadable'} | ConvertTo-Json -Compress
} finally {
    if ($null -ne $bitmap) { $bitmap.Dispose() }
    if ($null -ne $stream) { $stream.Dispose() }
}
