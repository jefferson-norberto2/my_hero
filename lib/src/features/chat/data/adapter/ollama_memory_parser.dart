import 'package:my_hero/src/shared/models/compute_mode.dart';
import 'package:my_hero/src/shared/models/memory_allocation.dart';

/// Parses Ollama `/api/ps` entries and classifies hardware execution modes.
class OllamaMemoryParser {
  const OllamaMemoryParser._();

  /// Extracts `size` and `size_vram` from a `/api/ps` model JSON entry.
  static MemoryAllocation parsePsEntry(Map<String, dynamic>? json) {
    if (json == null) return const MemoryAllocation.empty();

    final size = (json['size'] as num?)?.toInt() ?? 0;
    final sizeVram = (json['size_vram'] as num?)?.toInt() ?? 0;

    return fromBytes(sizeBytes: size, vramBytes: sizeVram);
  }

  /// Computes [MemoryAllocation] and [ComputeMode] from raw byte counters.
  static MemoryAllocation fromBytes({
    required int sizeBytes,
    required int vramBytes,
  }) {
    final safeSize = sizeBytes > 0 ? sizeBytes : 0;
    final safeVram = vramBytes > 0 ? vramBytes : 0;

    if (safeSize == 0 && safeVram == 0) {
      return const MemoryAllocation.empty();
    }

    if (safeVram == 0 && safeSize > 0) {
      return MemoryAllocation(
        totalBytes: safeSize,
        vramBytes: 0,
        ramBytes: safeSize,
        displayedBytes: safeSize,
        computeMode: ComputeMode.cpu,
      );
    }

    if (safeVram > 0 && safeVram >= safeSize) {
      return MemoryAllocation(
        totalBytes: safeVram,
        vramBytes: safeVram,
        ramBytes: 0,
        displayedBytes: safeVram,
        computeMode: ComputeMode.gpu,
      );
    }

    final ramBytes = safeSize - safeVram;
    return MemoryAllocation(
      totalBytes: safeSize,
      vramBytes: safeVram,
      ramBytes: ramBytes,
      displayedBytes: safeSize,
      computeMode: ComputeMode.hybrid,
    );
  }
}
