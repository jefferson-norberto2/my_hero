import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import '../../../../core/constants/app_constants.dart';

abstract class GemmaLocalDatasource {
  PreferredBackend? get activeBackend;

  Future<void> initialize();

  Future<bool> isModelInstalled(String modelFileName);

  Future<void> installModelFromNetwork({
    required String url,
    required void Function(int percent) onProgress,
    ModelType modelType = ModelType.gemma4,
    ModelFileType fileType = ModelFileType.litertlm,
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

  Future<void> createChat({
    ModelType modelType = ModelType.gemma4,
    String? systemInstruction,
    bool supportImage = true,
    bool enableThinking = false,
    double temperature = 0.8,
    int topK = 40,
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
  Future<void> initialize() async {
    await FlutterGemma.initialize(
      inferenceEngines: [LiteRtLmEngine()],
    );
  }

  @override
  Future<bool> isModelInstalled(String modelFileName) async {
    return FlutterGemma.isModelInstalled(modelFileName);
  }

  @override
  Future<void> installModelFromNetwork({
    required String url,
    required void Function(int percent) onProgress,
    ModelType modelType = ModelType.gemma4,
    ModelFileType fileType = ModelFileType.litertlm,
    CancelToken? cancelToken,
  }) async {
    final token = cancelToken ?? CancelToken();
    _activeDownloadCancelToken = token;

    final useForeground = !kIsWeb &&
        defaultTargetPlatform == TargetPlatform.android;

    try {
      await FlutterGemma.installModel(
        modelType: modelType,
        fileType: fileType,
      )
          .fromNetwork(url, foreground: useForeground)
          .withProgress(onProgress)
          .withCancelToken(token)
          .install();
    } finally {
      if (identical(_activeDownloadCancelToken, token)) {
        _activeDownloadCancelToken = null;
      }
    }
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

    _activeModel = await FlutterGemma.getActiveModel(
      maxTokens: maxTokens,
      preferredBackend: preferredBackend,
      supportImage: supportImage,
      maxNumImages: supportImage ? maxNumImages : null,
      activationDataType: activationDataType,
    );
  }

  @override
  Future<void> createChat({
    ModelType modelType = ModelType.gemma4,
    String? systemInstruction,
    bool supportImage = true,
    bool enableThinking = false,
    double temperature = 0.8,
    int topK = 40,
  }) async {
    final model = _activeModel;
    if (model == null) {
      throw StateError('Modelo ativo do Gemma ainda não foi inicializado.');
    }

    if (_activeChat != null) {
      await _activeChat?.close();
      _activeChat = null;
    }

    _activeChat = await model.createChat(
      modelType: modelType,
      supportImage: supportImage,
      isThinking: enableThinking,
      temperature: temperature,
      topK: topK,
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
