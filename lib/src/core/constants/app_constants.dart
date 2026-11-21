import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';

class GemmaModelSpec {
  final String displayName;
  final String shortLabel;
  final InferenceVariant catalogVariant;
  final int minContextTokens;
  final int maxContextTokens;

  const GemmaModelSpec({
    required this.displayName,
    required this.shortLabel,
    required this.catalogVariant,
    required this.minContextTokens,
    required this.maxContextTokens,
  });

  String get fileName =>
      catalogVariant.file?.installFilename ?? '${catalogVariant.modelId}.litertlm';

  String get downloadUrl => catalogVariant.file?.url ?? '';

  int? get expectedSizeBytes => catalogVariant.file?.sizeBytes;

  ModelType get modelType => catalogVariant.modelType;

  ModelFileType get fileType => catalogVariant.fileType;

  bool get supportsThinking => catalogVariant.supportsThinking;

  bool get supportsImage => catalogVariant.supportsImage;
}

class AppConstants {
  static final InferenceVariant model2bCatalogVariant =
      LiteRtLmModels.gemma4E2bIt.generic;
  static final InferenceVariant model4bCatalogVariant =
      LiteRtLmModels.gemma4E4bIt.generic;

  static const String model2bFileName = 'gemma-4-E2B-it.litertlm';
  static const String model4bFileName = 'gemma-4-E4B-it.litertlm';

  /// Limites oficiais de contexto dos bundles LiteRT-LM Gemma 4 E2B e E4B:
  /// - Mínimo rígido: 1024 tokens (`kv_cache_max_len` do runtime `.litertlm`).
  /// - Máximo suportado pelo modelo: 32.768 tokens (32K).
  static const int minContextTokens = 1024;
  static const int maxContextTokens = 32768;
  static const int recommendedMinVisionTokens = 4096;
  static const List<int> contextTokenPresets = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
  ];

  static final GemmaModelSpec model2b = GemmaModelSpec(
    displayName: 'Gemma 4 E2B',
    shortLabel: '2B',
    catalogVariant: model2bCatalogVariant,
    minContextTokens: minContextTokens,
    maxContextTokens: maxContextTokens,
  );

  static final GemmaModelSpec model4b = GemmaModelSpec(
    displayName: 'Gemma 4 E4B',
    shortLabel: '4B',
    catalogVariant: model4bCatalogVariant,
    minContextTokens: minContextTokens,
    maxContextTokens: maxContextTokens,
  );

  static GemmaModelSpec modelSpec({required bool is2B}) =>
      is2B ? model2b : model4b;

  static const String systemInstruction = 'You are a helpful assistant.';

  /// Mantido como padrão seguro para Desktop.
  static const int maxTokens = 8192;

  static int defaultMaxTokensForPlatform([TargetPlatform? platform]) {
    final effective = platform ?? defaultTargetPlatform;
    return switch (effective) {
      TargetPlatform.android || TargetPlatform.iOS => 4096,
      TargetPlatform.linux ||
      TargetPlatform.windows ||
      TargetPlatform.macOS ||
      TargetPlatform.fuchsia => 8192,
    };
  }

  /// NPU é suportado no LiteRT-LM FFI no Android (Qualcomm Snapdragon / Google
  /// Tensor — via variantes por SoC + opt-in) e no Windows (Intel Lunar Lake /
  /// Panther Lake — embarcado por padrão). No Linux x64 não há NPU.
  static bool supportsNpuOnPlatform([TargetPlatform? platform]) {
    if (kIsWeb) return false;
    final effective = platform ?? defaultTargetPlatform;
    return effective == TargetPlatform.android ||
        effective == TargetPlatform.windows;
  }

  static List<PreferredBackend> availableBackendsForPlatform([
    TargetPlatform? platform,
  ]) {
    return [
      PreferredBackend.cpu,
      PreferredBackend.gpu,
      if (supportsNpuOnPlatform(platform)) PreferredBackend.npu,
    ];
  }

  /// `image_picker` não oferece captura de câmera nativa no Linux/Windows/macOS.
  static bool supportsCameraOnPlatform([TargetPlatform? platform]) {
    if (kIsWeb) return false;
    final effective = platform ?? defaultTargetPlatform;
    return effective == TargetPlatform.android ||
        effective == TargetPlatform.iOS;
  }

  static int clampMaxTokens(int value, {bool is2B = true}) {
    final spec = modelSpec(is2B: is2B);
    if (value < spec.minContextTokens) return spec.minContextTokens;
    if (value > spec.maxContextTokens) return spec.maxContextTokens;
    return value;
  }

  static String backendLabel(PreferredBackend backend) {
    return switch (backend) {
      PreferredBackend.cpu => 'CPU',
      PreferredBackend.gpu => 'GPU',
      PreferredBackend.npu => 'NPU',
    };
  }

  static String formatTokenCount(int tokens) {
    if (tokens >= 1024 && tokens % 1024 == 0) {
      return '${tokens ~/ 1024}K';
    }
    return tokens.toString();
  }
}
