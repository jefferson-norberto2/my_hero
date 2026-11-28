import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';

abstract class GemmaLocalDatasource {
  PreferredBackend? get activeBackend;

  PreferredBackend? get activeVisionBackend;

  Future<void> initialize();

  Future<bool> isModelInstalled(String modelFileName);

  Future<List<String>> listInstalledModels();

  Future<void> uninstallModel(String modelFileName);

  Future<void> installModelFromCatalog({
    required InferenceVariant variant,
    required void Function(int percent) onProgress,
    CancelToken? cancelToken,
  });

  void cancelActiveDownload();

  Future<void> createActiveModel({
    required PreferredBackend preferredBackend,
    required int maxTokens,
    bool supportImage = true,
    int maxNumImages = 1,
    ActivationDataType? activationDataType,
  });

  Future<void> openChat({
    ModelType? modelType,
    String? systemInstruction,
    bool supportImage = true,
    bool enableThinking = false,
    double? temperature,
    int? topK,
    double? topP,
  });

  Future<void> addQueryChunk(Message message);

  Stream<ModelResponse> generateChatResponseAsync();

  Future<void> stopGeneration();

  Future<void> close();
}

class GemmaLocalDatasourceImpl implements GemmaLocalDatasource {
  InferenceModel? _activeModel;
  InferenceChat? _activeChat;
  CancelToken? _activeDownloadCancelToken;

  @override
  PreferredBackend? get activeBackend => _activeModel?.activeBackend;

  @override
  PreferredBackend? get activeVisionBackend =>
      _activeModel?.activeVisionBackend;

  @override
  Future<void> initialize() async {
    await FlutterEdgeAi.initialize(
      inferenceEngines: const [LiteRtLmEngine()],
    );
  }

  @override
  Future<bool> isModelInstalled(String modelFileName) async {
    return FlutterEdgeAi.isModelInstalled(modelFileName);
  }

  @override
  Future<List<String>> listInstalledModels() async {
    return FlutterEdgeAi.listInstalledModels();
  }

  @override
  Future<void> uninstallModel(String modelFileName) async {
    final activeSpec = FlutterEdgeAi.activeModelSpec;
    final isCurrentlyActive = activeSpec != null &&
        activeSpec.files.any((f) => f.filename == modelFileName);
    if (isCurrentlyActive) {
      await close();
    }
    await FlutterEdgeAi.uninstallModel(modelFileName);
  }

  @override
  Future<void> installModelFromCatalog({
    required InferenceVariant variant,
    required void Function(int percent) onProgress,
    CancelToken? cancelToken,
  }) async {
    final token = cancelToken ?? CancelToken();
    _activeDownloadCancelToken = token;

    final useForeground = !kIsWeb &&
        defaultTargetPlatform == TargetPlatform.android;

    try {
      await _uninstallConflictingActiveModelIfNeeded(variant);

      await FlutterEdgeAi.installModelFromCatalog(
        variant,
        foreground: useForeground,
      )
          .withProgress(onProgress)
          .withCancelToken(token)
          .install();
    } finally {
      if (identical(_activeDownloadCancelToken, token)) {
        _activeDownloadCancelToken = null;
      }
    }
  }

  /// Se o usuário possuía o mesmo arquivo instalado por outra URL (ex.:
  /// `/resolve/main/` da `flutter_gemma 1.11.4` ou de uma revisão anterior do
  /// catálogo) como modelo ativo, desinstala-o antes de instalar a variante
  /// pinada do catálogo para evitar o `StateError` de substituição em uso.
  Future<void> _uninstallConflictingActiveModelIfNeeded(
    InferenceVariant variant,
  ) async {
    final file = variant.file;
    if (file == null) return;

    final activeSpec = FlutterEdgeAi.activeModelSpec;
    if (activeSpec == null) return;

    final usesSameFilename = activeSpec.files.any(
      (f) => f.filename == file.installFilename,
    );
    if (!usesSameFilename) return;

    final activeSource = activeSpec.modelSource;
    final samePinnedUrl =
        activeSource is NetworkSource && activeSource.url == file.url;
    if (samePinnedUrl) return;

    await close();
    await FlutterEdgeAi.uninstallModel(file.installFilename);
  }

  @override
  void cancelActiveDownload() {
    _activeDownloadCancelToken?.cancel('Download cancelado pelo usuário.');
    _activeDownloadCancelToken = null;
  }

  @override
  Future<void> createActiveModel({
    required PreferredBackend preferredBackend,
    required int maxTokens,
    bool supportImage = true,
    int maxNumImages = 1,
    ActivationDataType? activationDataType,
  }) async {
    if (_activeChat != null) {
      await _activeChat?.close();
      _activeChat = null;
    }

    _activeModel = await FlutterEdgeAi.getActiveModel(
      maxTokens: maxTokens,
      preferredBackend: preferredBackend,
      supportImage: supportImage,
      maxNumImages: supportImage ? maxNumImages : null,
      activationDataType: activationDataType,
    );
  }

  @override
  Future<void> openChat({
    ModelType? modelType,
    String? systemInstruction,
    bool supportImage = true,
    bool enableThinking = false,
    double? temperature,
    int? topK,
    double? topP,
  }) async {
    final model = _activeModel;
    if (model == null) {
      throw StateError('Modelo ativo do Gemma ainda não foi inicializado.');
    }

    if (_activeChat != null) {
      await _activeChat?.close();
      _activeChat = null;
    }

    _activeChat = await model.openChat(
      modelType: modelType,
      supportImage: supportImage,
      enableThinking: enableThinking,
      temperature: temperature,
      topK: topK,
      topP: topP,
      systemInstruction: systemInstruction ?? AppConstants.systemInstruction,
    );
  }

  @override
  Future<void> addQueryChunk(Message message) async {
    final chat = _activeChat;
    if (chat == null) {
      throw StateError('Chat do Gemma não foi inicializado.');
    }
    await chat.addQueryChunk(message);
  }

  @override
  Stream<ModelResponse> generateChatResponseAsync() {
    final chat = _activeChat;
    if (chat == null) {
      throw StateError('Chat do Gemma não foi inicializado.');
    }
    return chat.generateChatResponseAsync();
  }

  @override
  Future<void> stopGeneration() async {
    await _activeChat?.stopGeneration();
  }

  @override
  Future<void> close() async {
    await _activeChat?.close();
    _activeChat = null;
    await _activeModel?.close();
    _activeModel = null;
  }
}
