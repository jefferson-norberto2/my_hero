class OllamaModel {
  final String name;
  final String model;
  final int size;
  final String? parameterSize;
  final String? family;
  final String? format;
  final String? quantizationLevel;
  final bool isLoaded;

  const OllamaModel({
    required this.name,
    required this.model,
    required this.size,
    this.parameterSize,
    this.family,
    this.format,
    this.quantizationLevel,
    this.isLoaded = false,
  });

  String get sizeFormatted {
    if (size <= 0) return '';
    final gb = size / (1024 * 1024 * 1024);
    if (gb >= 1.0) {
      return '${gb.toStringAsFixed(1)} GB';
    }
    final mb = size / (1024 * 1024);
    return '${mb.toStringAsFixed(0)} MB';
  }

  factory OllamaModel.fromJson(Map<String, dynamic> json, {bool isLoaded = false}) {
    final details = json['details'] as Map<String, dynamic>? ?? {};
    return OllamaModel(
      name: json['name'] as String? ?? '',
      model: json['model'] as String? ?? '',
      size: (json['size'] as num?)?.toInt() ?? 0,
      parameterSize: details['parameter_size'] as String?,
      family: details['family'] as String?,
      format: details['format'] as String?,
      quantizationLevel: details['quantization_level'] as String?,
      isLoaded: isLoaded,
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
    );
  }
}
