import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter_gemma/flutter_gemma.dart';

class GemmaModelSpec {
  final String displayName;
  final String shortLabel;
  final String fileName;
  final String downloadUrl;
  final int minContextTokens;
  final int maxContextTokens;
  final ModelType modelType;
  final ModelFileType fileType;
  final bool supportsThinking;
  final bool supportsImage;

  const GemmaModelSpec({
    required this.displayName,
    required this.shortLabel,
    required this.fileName,
    required this.downloadUrl,
    required this.minContextTokens,
    required this.maxContextTokens,
    this.modelType = ModelType.gemma4,
    this.fileType = ModelFileType.litertlm,
    this.supportsThinking = true,
    this.supportsImage = true,
  });
}

class AppConstants {
  static const String model2bUrl =
      'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm';
  static const String model4bUrl =
      'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm';

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

  static const GemmaModelSpec model2b = GemmaModelSpec(
    displayName: 'Gemma 4 E2B',
    shortLabel: '2B',
    fileName: model2bFileName,
    downloadUrl: model2bUrl,
    minContextTokens: minContextTokens,
    maxContextTokens: maxContextTokens,
  );

  static const GemmaModelSpec model4b = GemmaModelSpec(
    displayName: 'Gemma 4 E4B',
    shortLabel: '4B',
    fileName: model4bFileName,
    downloadUrl: model4bUrl,
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

  /// NPU é suportado no LiteRT-LM FFI apenas no Android (Qualcomm Snapdragon)
  /// e no Windows (Intel LunarLake/PantherLake).
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
