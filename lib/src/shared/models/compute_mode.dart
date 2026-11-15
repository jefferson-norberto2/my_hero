/// Execution hardware mode reported for an active Ollama model in memory.
enum ComputeMode {
  /// No model memory is currently allocated (`size == 0` and `size_vram == 0`).
  none,

  /// Model is loaded exclusively in system RAM (`size_vram == 0 && size > 0`).
  cpu,

  /// Model is loaded exclusively in GPU VRAM (`size_vram > 0 && size_vram >= size`).
  gpu,

  /// Model is split between GPU VRAM and system RAM (`size_vram > 0 && size_vram < size`).
  hybrid,
}

/// UI presentation helpers for [ComputeMode].
extension ComputeModeX on ComputeMode {
  /// Compact hardware tag label shown next to memory usage in the AppBar.
  String get tagLabel {
    switch (this) {
      case ComputeMode.none:
        return '';
      case ComputeMode.cpu:
        return 'CPU';
      case ComputeMode.gpu:
        return 'GPU';
      case ComputeMode.hybrid:
        return 'GPU/CPU';
    }
  }

  /// Whether an active compute hardware tag should be displayed.
  bool get isActive => this != ComputeMode.none;
}
