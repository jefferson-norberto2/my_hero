import 'package:my_hero/src/shared/models/compute_mode.dart';

/// Immutable breakdown of an Ollama model's runtime memory allocation.
class MemoryAllocation {
  /// Total runtime memory reported by Ollama in bytes.
  final int totalBytes;

  /// Video memory (VRAM) portion allocated on the GPU in bytes.
  final int vramBytes;

  /// System memory (RAM) portion allocated on the CPU in bytes.
  final int ramBytes;

  /// Primary memory value displayed in the UI according to [computeMode].
  final int displayedBytes;

  /// Classified hardware execution mode for this allocation.
  final ComputeMode computeMode;

  const MemoryAllocation({
    required this.totalBytes,
    required this.vramBytes,
    required this.ramBytes,
    required this.displayedBytes,
    required this.computeMode,
  });

  /// Represents an unloaded model with zero memory consumption.
  const MemoryAllocation.empty()
      : totalBytes = 0,
        vramBytes = 0,
        ramBytes = 0,
        displayedBytes = 0,
        computeMode = ComputeMode.none;

  /// Formats a byte count into a human-readable GB/MB string.
  static String formatBytes(int bytes) {
    if (bytes <= 0) return '0.0 GB';
    final gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1.0) {
      return '${gb.toStringAsFixed(1)} GB';
    }
    final mb = bytes / (1024 * 1024);
    return '${mb.toStringAsFixed(0)} MB';
  }

  /// Formatted primary memory value shown in the status pill (e.g. `3.8 GB` or `0.0 GB`).
  String get formattedMemory => formatBytes(displayedBytes);

  /// Formatted VRAM amount.
  String get formattedVram => formatBytes(vramBytes);

  /// Formatted system RAM amount.
  String get formattedRam => formatBytes(ramBytes);

  /// Detailed Portuguese description displayed in tooltips and metric dialogs.
  String get hardwareTooltip {
    switch (computeMode) {
      case ComputeMode.none:
        return 'Nenhum modelo carregado na memória';
      case ComputeMode.cpu:
        return 'Memória RAM (CPU): $formattedRam';
      case ComputeMode.gpu:
      case ComputeMode.hybrid:
        return 'VRAM: $formattedVram | RAM: $formattedRam';
    }
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is MemoryAllocation &&
          runtimeType == other.runtimeType &&
          totalBytes == other.totalBytes &&
          vramBytes == other.vramBytes &&
          ramBytes == other.ramBytes &&
          displayedBytes == other.displayedBytes &&
          computeMode == other.computeMode;

  @override
  int get hashCode =>
      Object.hash(totalBytes, vramBytes, ramBytes, displayedBytes, computeMode);
}
