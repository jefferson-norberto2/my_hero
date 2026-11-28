import 'dart:io';
import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';

abstract class GemmaLocalDatasource {
  PreferredBackend? get activeBackend;

  PreferredBackend? get activeVisionBackend;

  Future<void> initialize();

  Future<bool> isModelInstalled(String modelFileName);

  Future<List<String>> listInstalledModels();

  Future<void> uninstallModel(String modelFileName);

  Future<void> activateInstalledModel({
    required String fileName,
    required ModelType modelType,
    required ModelFileType fileType,
  });

  Future<void> installModelFromCatalog({
    required InferenceVariant variant,
    required void Function(int percent) onProgress,
    CancelToken? cancelToken,
  });

  Future<void> cleanupStorage();

  void cancelActiveDownload();

  Future<void> createActiveModel({
    required PreferredBackend preferredBackend,
    required int maxTokens,
    bool supportImage = true,
    int maxNumImages = 1,
    ActivationDataType? activationDataType,
    int? maxConcurrentSessions,
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

typedef IsModelInstalledFn = Future<bool> Function(String fileName);
typedef ListInstalledModelsFn = Future<List<String>> Function();
typedef ActiveModelSpecFn = InferenceModelSpec? Function();
typedef GetModelPathFn = Future<String> Function(String fileName);
typedef ActivateFromFileFn = Future<void> Function({
  required ModelType modelType,
  required ModelFileType fileType,
  required String filePath,
});
typedef InstallFromCatalogFn = Future<void> Function({
  required InferenceVariant variant,
  required bool foreground,
  required void Function(int percent) onProgress,
  required CancelToken cancelToken,
});
typedef UninstallModelFn = Future<void> Function(String fileName);
typedef PerformCleanupFn = Future<void> Function();
typedef SupportDirectoryFn = Future<Directory> Function();
typedef GetActiveModelFn = Future<InferenceModel> Function({
  required int maxTokens,
  required PreferredBackend preferredBackend,
  required bool supportImage,
  int? maxNumImages,
  ActivationDataType? activationDataType,
  int? maxConcurrentSessions,
});

class GemmaLocalDatasourceImpl implements GemmaLocalDatasource {
  InferenceModel? _activeModel;
  InferenceChat? _activeChat;
  CancelToken? _activeDownloadCancelToken;

  final TargetPlatform? _platformOverride;
  final IsModelInstalledFn _isModelInstalledFn;
  final ListInstalledModelsFn _listInstalledModelsFn;
  final ActiveModelSpecFn _activeModelSpecFn;
  final GetModelPathFn _getModelPathFn;
  final ActivateFromFileFn _activateFromFileFn;
  final InstallFromCatalogFn _installFromCatalogFn;
  final UninstallModelFn _uninstallModelFn;
  final PerformCleanupFn _performCleanupFn;
  final SupportDirectoryFn _supportDirectoryFn;
  final GetActiveModelFn _getActiveModelFn;

  GemmaLocalDatasourceImpl({
    TargetPlatform? platform,
    IsModelInstalledFn? isModelInstalledFn,
    ListInstalledModelsFn? listInstalledModelsFn,
    ActiveModelSpecFn? activeModelSpecFn,
    GetModelPathFn? getModelPathFn,
    ActivateFromFileFn? activateFromFileFn,
    InstallFromCatalogFn? installFromCatalogFn,
    UninstallModelFn? uninstallModelFn,
    PerformCleanupFn? performCleanupFn,
    SupportDirectoryFn? supportDirectoryFn,
    GetActiveModelFn? getActiveModelFn,
  })  : _platformOverride = platform,
        _isModelInstalledFn =
            isModelInstalledFn ?? FlutterEdgeAi.isModelInstalled,
        _listInstalledModelsFn =
            listInstalledModelsFn ?? FlutterEdgeAi.listInstalledModels,
        _activeModelSpecFn =
            activeModelSpecFn ?? (() => FlutterEdgeAi.activeModelSpec),
        _getModelPathFn = getModelPathFn ?? FlutterEdgeAi.getModelPath,
        _activateFromFileFn = activateFromFileFn ?? _defaultActivateFromFile,
        _installFromCatalogFn =
            installFromCatalogFn ?? _defaultInstallFromCatalog,
        _uninstallModelFn = uninstallModelFn ?? FlutterEdgeAi.uninstallModel,
        _performCleanupFn = performCleanupFn ?? FlutterEdgeAi.performCleanup,
        _supportDirectoryFn =
            supportDirectoryFn ?? getApplicationSupportDirectory,
        _getActiveModelFn = getActiveModelFn ?? _defaultGetActiveModel;

  static Future<void> _defaultActivateFromFile({
    required ModelType modelType,
    required ModelFileType fileType,
    required String filePath,
  }) async {
    await FlutterEdgeAi.installModel(
      modelType: modelType,
      fileType: fileType,
    ).fromFile(filePath).install();
  }

  static Future<void> _defaultInstallFromCatalog({
    required InferenceVariant variant,
    required bool foreground,
    required void Function(int percent) onProgress,
    required CancelToken cancelToken,
  }) async {
    await FlutterEdgeAi.installModelFromCatalog(
      variant,
      foreground: foreground,
    ).withProgress(onProgress).withCancelToken(cancelToken).install();
  }

  static Future<InferenceModel> _defaultGetActiveModel({
    required int maxTokens,
    required PreferredBackend preferredBackend,
    required bool supportImage,
    int? maxNumImages,
    ActivationDataType? activationDataType,
    int? maxConcurrentSessions,
  }) {
    return FlutterEdgeAi.getActiveModel(
      maxTokens: maxTokens,
      preferredBackend: preferredBackend,
      supportImage: supportImage,
      maxNumImages: supportImage ? maxNumImages : null,
      activationDataType: activationDataType,
      maxConcurrentSessions: maxConcurrentSessions,
    );
  }

  bool get _isAndroid =>
      !kIsWeb &&
      (_platformOverride ?? defaultTargetPlatform) == TargetPlatform.android;

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
    return _isModelInstalledFn(modelFileName);
  }

  @override
  Future<List<String>> listInstalledModels() async {
    return _listInstalledModelsFn();
  }

  @override
  Future<void> uninstallModel(String modelFileName) async {
    final activeSpec = _activeModelSpecFn();
    final isCurrentlyActive = activeSpec != null &&
        activeSpec.files.any((f) => f.filename == modelFileName);
    if (isCurrentlyActive) {
      await close();
    }
    await _uninstallModelFn(modelFileName);
    await _deleteModelNativeCaches(modelFileName);
  }

  /// Remove caches nativos compilados pelo LiteRT-LM (`xnnpack_cache`,
  /// `opencl_cache`, etc.) pertencentes ao modelo desinstalado sem afetar
  /// arquivos `.json` de histórico ou caches de outros modelos.
  Future<void> _deleteModelNativeCaches(String modelFileName) async {
    if (kIsWeb) return;
    try {
      final supportDir = await _supportDirectoryFn();
      if (!await supportDir.exists()) return;

      final baseName = p.basenameWithoutExtension(modelFileName);
      if (baseName.isEmpty) return;

      await for (final entity in supportDir.list(followLinks: false)) {
        if (entity is! File) continue;
        final name = p.basename(entity.path);
        if (name.endsWith('.json')) continue;
        if (_isCacheFileForModel(name, modelFileName, baseName)) {
          try {
            await entity.delete();
          } catch (_) {
            // Ignora arquivo em uso; próximo cleanup/início limpa.
          }
        }
      }
    } catch (_) {
      // Limpeza de cache é best-effort e nunca deve falhar a desinstalação.
    }
  }

  static bool _isCacheFileForModel(
    String fileName,
    String modelFileName,
    String baseName,
  ) {
    final matchesModel =
        fileName.contains(modelFileName) || fileName.contains(baseName);
    if (!matchesModel) return false;

    final lower = fileName.toLowerCase();
    return lower.contains('cache') ||
        lower.endsWith('.bin') ||
        lower.endsWith('.tmp') ||
        lower.startsWith('${modelFileName.toLowerCase()}.');
  }

  @override
  Future<void> activateInstalledModel({
    required String fileName,
    required ModelType modelType,
    required ModelFileType fileType,
  }) async {
    final activeSpec = _activeModelSpecFn();
    final alreadyActive = activeSpec != null &&
        activeSpec.files.any((f) => f.filename == fileName) &&
        activeSpec.modelType == modelType &&
        activeSpec.fileType == fileType;
    if (alreadyActive) {
      return;
    }

    final filePath = await _getModelPathFn(fileName);
    await _activateFromFileFn(
      modelType: modelType,
      fileType: fileType,
      filePath: filePath,
    );
  }

  @override
  Future<void> installModelFromCatalog({
    required InferenceVariant variant,
    required void Function(int percent) onProgress,
    CancelToken? cancelToken,
  }) async {
    final token = cancelToken ?? CancelToken();
    _activeDownloadCancelToken = token;

    try {
      await _installFromCatalogFn(
        variant: variant,
        foreground: _isAndroid,
        onProgress: onProgress,
        cancelToken: token,
      );
    } finally {
      if (identical(_activeDownloadCancelToken, token)) {
        _activeDownloadCancelToken = null;
      }
    }
  }

  @override
  Future<void> cleanupStorage() async {
    await _performCleanupFn();
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
    int? maxConcurrentSessions,
  }) async {
    if (_activeChat != null) {
      await _activeChat?.close();
      _activeChat = null;
    }

    final effectiveMaxConcurrentSessions =
        maxConcurrentSessions ?? (_isAndroid ? 1 : null);

    try {
      _activeModel = await _getActiveModelFn(
        maxTokens: maxTokens,
        preferredBackend: preferredBackend,
        supportImage: supportImage,
        maxNumImages: supportImage ? maxNumImages : null,
        activationDataType: activationDataType,
        maxConcurrentSessions: effectiveMaxConcurrentSessions,
      );
    } catch (_) {
      _activeModel = null;
      rethrow;
    }
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
