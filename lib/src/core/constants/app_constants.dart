import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:my_hero/src/shared/services/device_resources_service.dart';

class AiModelSpec {
  final String id;
  final String displayName;
  final String family;
  final InferenceVariant catalogVariant;
  final int minContextTokens;
  final int maxContextTokens;

  const AiModelSpec({
    required this.id,
    required this.displayName,
    required this.family,
    required this.catalogVariant,
    required this.minContextTokens,
    required this.maxContextTokens,
  });

  String get fileName =>
      catalogVariant.file?.installFilename ??
      '${catalogVariant.modelId}.litertlm';

  String get downloadUrl => catalogVariant.file?.url ?? '';

  int get sizeBytes => catalogVariant.file?.sizeBytes ?? 0;

  String get formattedSize => AppConstants.formatBytes(sizeBytes);

  ModelType get modelType => catalogVariant.modelType;

  ModelFileType get fileType => catalogVariant.fileType;

  bool get supportsThinking => catalogVariant.supportsThinking;

  bool get supportsImage => catalogVariant.supportsImage;

  bool get supportsAudio => catalogVariant.supportsAudio;

  List<int> get contextTokenPresets => AppConstants.contextTokenPresets
      .where((t) => t >= minContextTokens && t <= maxContextTokens)
      .toList(growable: false);
}

typedef GemmaModelSpec = AiModelSpec;

class AppConstants {
  static final InferenceVariant model2bCatalogVariant =
      LiteRtLmModels.gemma4E2bIt.generic;
  static final InferenceVariant model4bCatalogVariant =
      LiteRtLmModels.gemma4E4bIt.generic;

  static const String model2bFileName = 'gemma-4-E2B-it.litertlm';
  static const String model4bFileName = 'gemma-4-E4B-it.litertlm';

  /// Limites globais de contexto dos bundles LiteRT-LM:
  /// - Mínimo rígido: 1024 tokens (`kv_cache_max_len` do runtime `.litertlm`).
  /// - Máximo suportado (Gemma 4): 32.768 tokens (32K).
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

  /// Modelos do catálogo excluídos por serem demonstrações de fine-tune específico.
  static const Set<String> _excludedCatalogModelIds = {
    'functiongemma-270m-flutter-demo',
  };

  /// Ordem de destaque no topo do catálogo.
  static const List<String> _featuredModelIds = [
    'gemma-4-e2b-it',
    'gemma-4-e4b-it',
  ];

  /// Catálogo dinâmico de todos os modelos abertos (sem exigência de token no
  /// Hugging Face) disponíveis em `LiteRtLmModels` para execução em CPU/GPU.
  static final List<AiModelSpec> availableModels = _buildOpenCatalogModels();

  static final AiModelSpec model2b = modelSpecById(model2bCatalogVariant.id);

  static final AiModelSpec model4b = modelSpecById(model4bCatalogVariant.id);

  static String get defaultModelId => model2b.id;

  static List<AiModelSpec> _buildOpenCatalogModels() {
    final specs = <AiModelSpec>[];

    for (final model in LiteRtLmModels.all) {
      if (_excludedCatalogModelIds.contains(model.id)) continue;

      final variant = _pickOpenGenericVariant(model);
      if (variant == null) continue;

      final limits = _contextLimitsFor(model.id, variant);
      specs.add(
        AiModelSpec(
          id: variant.id,
          displayName: model.displayName,
          family: model.family,
          catalogVariant: variant,
          minContextTokens: limits.min,
          maxContextTokens: limits.max,
        ),
      );
    }

    specs.sort((a, b) {
      final aFeatured = _featuredModelIds.indexOf(a.catalogVariant.modelId);
      final bFeatured = _featuredModelIds.indexOf(b.catalogVariant.modelId);
      if (aFeatured != -1 || bFeatured != -1) {
        if (aFeatured == -1) return 1;
        if (bFeatured == -1) return -1;
        return aFeatured.compareTo(bFeatured);
      }
      final sizeCmp = a.sizeBytes.compareTo(b.sizeBytes);
      if (sizeCmp != 0) return sizeCmp;
      return a.displayName.compareTo(b.displayName);
    });

    return List.unmodifiable(specs);
  }

  static LiteRtLmVariant? _pickOpenGenericVariant(LiteRtLmModel model) {
    for (final variant in model.variants) {
      if (variant.isWebBundle) continue;
      if (variant.npuTarget != null) continue;
      if (variant.file.gated) continue;
      if (variant.backendsHere.isEmpty) continue;
      return variant;
    }
    return null;
  }

  static ({int min, int max}) _contextLimitsFor(
    String modelId,
    LiteRtLmVariant variant,
  ) {
    if (modelId == 'gemma-4-e2b-it' || modelId == 'gemma-4-e4b-it') {
      return (min: 1024, max: 32768);
    }

    final lowerName = '${variant.id} ${variant.file.installFilename}'
        .toLowerCase();
    if (lowerName.contains('ekv1024')) {
      return (min: 1024, max: 1024);
    }
    if (lowerName.contains('ekv4096')) {
      return (min: 1024, max: 4096);
    }
    if (modelId == 'translategemma-4b-it') {
      // TranslateGemma int4/int8 usa prefill mínimo de 1024 tokens: precisa de >= 2048.
      return (min: 2048, max: 4096);
    }
    return (min: 1024, max: 8192);
  }

  static AiModelSpec modelSpecById(String? modelId) {
    if (modelId != null) {
      for (final spec in availableModels) {
        if (spec.id == modelId || spec.catalogVariant.modelId == modelId) {
          return spec;
        }
      }
    }
    return availableModels.first;
  }

  static AiModelSpec modelSpec({required bool is2B}) =>
      is2B ? model2b : model4b;

  static const String systemInstruction = 'You are a helpful assistant.';

  /// Mantido como padrão seguro para Desktop.
  static const int maxTokens = 8192;

  static int defaultMaxTokensForPlatform([TargetPlatform? platform]) {
    final effective = platform ?? defaultTargetPlatform;
    return switch (effective) {
      TargetPlatform.android || TargetPlatform.iOS => 2048,
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

  /// Backends suportados pelo [spec] na plataforma atual, respeitando
  /// [InferenceVariant.backendsHere] (ex.: `TranslateGemma 4B` restringe GPU,
  /// e bundles genéricos não compilam para NPU).
  static List<PreferredBackend> availableBackendsForModel(
    AiModelSpec spec, [
    TargetPlatform? platform,
  ]) {
    final platformBackends = availableBackendsForPlatform(platform);
    final supportedByModel = spec.catalogVariant.backendsHere;
    if (supportedByModel.isEmpty) {
      return platformBackends;
    }

    final filtered = [
      for (final backend in platformBackends)
        if (supportedByModel.contains(backend)) backend,
    ];
    if (filtered.isNotEmpty) {
      return filtered;
    }
    return const [PreferredBackend.cpu];
  }

  static PreferredBackend defaultBackendForModel(
    AiModelSpec spec, [
    TargetPlatform? platform,
  ]) {
    final available = availableBackendsForModel(spec, platform);
    if (available.contains(PreferredBackend.cpu)) {
      return PreferredBackend.cpu;
    }
    return available.first;
  }

  static int requiredFreeStorageBytesForDownload(AiModelSpec spec) {
    const baseMargin = 512 * 1024 * 1024;
    if (spec.sizeBytes <= 0) return baseMargin;
    return (spec.sizeBytes * 1.4).round() + baseMargin;
  }

  static int estimateModelRamBytes(
    AiModelSpec spec, {
    required int maxTokens,
    required bool enableVision,
  }) {
    final weightsOverhead = (spec.sizeBytes * 1.25).round();
    final kvCacheOverhead = maxTokens * 64 * 1024;
    final visionOverhead =
        (spec.supportsImage && enableVision) ? 512 * 1024 * 1024 : 0;
    const runtimeBase = 350 * 1024 * 1024;
    return weightsOverhead + kvCacheOverhead + visionOverhead + runtimeBase;
  }

  static ModelMemoryAssessment assessModelMemory(
    AiModelSpec spec, {
    required int maxTokens,
    required bool enableVision,
    required DeviceResourcesInfo? resources,
  }) {
    final estimated = estimateModelRamBytes(
      spec,
      maxTokens: maxTokens,
      enableVision: enableVision,
    );
    if (resources == null || !resources.hasRamInfo) {
      return ModelMemoryAssessment(
        risk: ModelMemoryRisk.unknown,
        estimatedRequiredRamBytes: estimated,
      );
    }

    final estimatedStr = formatBytes(estimated);
    final totalStr = formatBytes(resources.totalRamBytes);
    final availStr = resources.availableRamBytes > 0
        ? formatBytes(resources.availableRamBytes)
        : null;
    final ramDetails = availStr != null
        ? 'estimado: ~$estimatedStr • RAM livre: $availStr de $totalStr'
        : 'estimado: ~$estimatedStr • RAM total: $totalStr';

    final isHighRisk = resources.lowMemory ||
        estimated >= (resources.totalRamBytes * 0.78) ||
        (resources.availableRamBytes > 0 &&
            estimated > (resources.availableRamBytes * 1.35));

    if (isHighRisk) {
      return ModelMemoryAssessment(
        risk: ModelMemoryRisk.high,
        estimatedRequiredRamBytes: estimated,
        message:
            'Alto risco de erro de memória no Android ($ramDetails). Reduza os tokens para 1K–2K, desative o suporte a imagens ou escolha um modelo menor.',
      );
    }

    final isWarningRisk = estimated >= (resources.totalRamBytes * 0.58) ||
        (resources.availableRamBytes > 0 &&
            estimated > (resources.availableRamBytes * 0.95));

    if (isWarningRisk) {
      return ModelMemoryAssessment(
        risk: ModelMemoryRisk.warning,
        estimatedRequiredRamBytes: estimated,
        message:
            'Memória RAM justa para ${spec.displayName} ($ramDetails). Se falhar ao carregar, tente 1K–2K tokens ou desative a visão.',
      );
    }

    return ModelMemoryAssessment(
      risk: ModelMemoryRisk.safe,
      estimatedRequiredRamBytes: estimated,
    );
  }

  /// `image_picker` não oferece captura de câmera nativa no Linux/Windows/macOS.
  static bool supportsCameraOnPlatform([TargetPlatform? platform]) {
    if (kIsWeb) return false;
    final effective = platform ?? defaultTargetPlatform;
    return effective == TargetPlatform.android ||
        effective == TargetPlatform.iOS;
  }

  static int clampMaxTokens(
    int value, {
    String? modelId,
    bool is2B = true,
  }) {
    final spec = modelId != null
        ? modelSpecById(modelId)
        : modelSpec(is2B: is2B);
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

  static String formatBytes(int bytes) {
    if (bytes <= 0) return '—';
    if (bytes >= 1000 * 1000 * 1000) {
      final gb = (bytes / (1000 * 1000 * 1000)).toStringAsFixed(2);
      return '${gb.replaceAll('.', ',')} GB';
    }
    final mb = (bytes / (1000 * 1000)).round();
    return '$mb MB';
  }
}
