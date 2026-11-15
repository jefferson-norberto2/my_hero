import 'package:my_hero/src/shared/models/compute_mode.dart';
import 'package:my_hero/src/shared/models/memory_allocation.dart';

/// Represents an Ollama model returned by the local or remote server.
class OllamaModel {
  final String name;
  final String model;
  final int size;
  final String? parameterSize;
  final String? family;
  final String? format;
  final String? quantizationLevel;
  final bool isLoaded;
  final int sizeVram;
  final MemoryAllocation memoryAllocation;
  final int contextLength;

  const OllamaModel({
    required this.name,
    required this.model,
    required this.size,
    this.parameterSize,
    this.family,
    this.format,
    this.quantizationLevel,
    this.isLoaded = false,
    this.sizeVram = 0,
    this.memoryAllocation = const MemoryAllocation.empty(),
    this.contextLength = 4096,
  });

  String get sizeFormatted {
    if (size <= 0) return '';
    return MemoryAllocation.formatBytes(size);
  }

  String get sizeVramFormatted {
    return MemoryAllocation.formatBytes(sizeVram);
  }

  ComputeMode get computeMode => memoryAllocation.computeMode;

  String get memoryFormatted => memoryAllocation.formattedMemory;

  factory OllamaModel.fromJson(
    Map<String, dynamic> json, {
    bool isLoaded = false,
    int sizeVram = 0,
    MemoryAllocation memoryAllocation = const MemoryAllocation.empty(),
    int? contextLength,
  }) {
    final details = json['details'] as Map<String, dynamic>? ?? {};
    final defaultCtx = (details['context_length'] as num?)?.toInt() ?? 4096;

    return OllamaModel(
      name: json['name'] as String? ?? '',
      model: json['model'] as String? ?? '',
      size: (json['size'] as num?)?.toInt() ?? 0,
      parameterSize: details['parameter_size'] as String?,
      family: details['family'] as String?,
      format: details['format'] as String?,
      quantizationLevel: details['quantization_level'] as String?,
      isLoaded: isLoaded,
      sizeVram: sizeVram,
      memoryAllocation: memoryAllocation,
      contextLength: contextLength ?? defaultCtx,
    );
  }

  OllamaModel copyWith({
    String? name,
    String? model,
    int? size,
    String? parameterSize,
    String? family,
    String? format,
    String? quantizationLevel,
    bool? isLoaded,
    int? sizeVram,
    MemoryAllocation? memoryAllocation,
    int? contextLength,
  }) {
    return OllamaModel(
      name: name ?? this.name,
      model: model ?? this.model,
      size: size ?? this.size,
      parameterSize: parameterSize ?? this.parameterSize,
      family: family ?? this.family,
      format: format ?? this.format,
      quantizationLevel: quantizationLevel ?? this.quantizationLevel,
      isLoaded: isLoaded ?? this.isLoaded,
      sizeVram: sizeVram ?? this.sizeVram,
      memoryAllocation: memoryAllocation ?? this.memoryAllocation,
      contextLength: contextLength ?? this.contextLength,
    );
  }
}
