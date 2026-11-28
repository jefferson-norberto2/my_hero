import 'dart:typed_data';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';

class ChatRepositoryImpl implements ChatRepository {
  final GemmaLocalDatasource _gemmaDatasource;
  final ChatHistoryDatasource _historyDatasource;

  ChatRepositoryImpl({
    required this._gemmaDatasource,
    required this._historyDatasource,
  });

  @override
  PreferredBackend? get activeBackend => _gemmaDatasource.activeBackend;

  AiModelSpec _resolveSpec({String? modelId, bool is2B = true}) {
    return modelId != null
        ? AppConstants.modelSpecById(modelId)
        : AppConstants.modelSpec(is2B: is2B);
  }

  @override
  Future<bool> isModelInstalled({String? modelId, bool is2B = true}) {
    final spec = _resolveSpec(modelId: modelId, is2B: is2B);
    return _gemmaDatasource.isModelInstalled(spec.fileName);
  }

  @override
  Future<Set<String>> getInstalledModelFileNames() async {
    final installed = await _gemmaDatasource.listInstalledModels();
    return installed.toSet();
  }

  @override
  Future<void> uninstallModel(String modelFileName) {
    return _gemmaDatasource.uninstallModel(modelFileName);
  }

  @override
  Future<void> loadModel({
    String? modelId,
    bool is2B = true,
    required PreferredBackend preferredBackend,
    required int maxTokens,
    required void Function(double progress, String status) onProgress,
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool? supportImage,
    bool enableThinking = false,
  }) async {
    final spec = _resolveSpec(modelId: modelId, is2B: is2B);
    final safeMaxTokens = AppConstants.clampMaxTokens(
      maxTokens,
      modelId: spec.id,
    );
    final effectiveSupportImage = (supportImage ?? true) && spec.supportsImage;
    final effectiveThinking = enableThinking && spec.supportsThinking;

    onProgress(0.0, 'Verificando armazenamento local...');
    final alreadyInstalled =
        await _gemmaDatasource.isModelInstalled(spec.fileName);

    if (alreadyInstalled) {
      onProgress(1.0, 'Modelo ${spec.displayName} encontrado localmente.');
    } else {
      onProgress(
        0.0,
        'Iniciando download de ${spec.displayName} (${spec.formattedSize})...',
      );
    }

    await _gemmaDatasource.installModelFromCatalog(
      variant: spec.catalogVariant,
      onProgress: (percent) {
        final normalized = (percent / 100.0).clamp(0.0, 1.0);
        onProgress(
          normalized,
          'Baixando ${spec.displayName}: $percent%',
        );
      },
    );

    onProgress(
      1.0,
      'Carregando ${spec.displayName} (${AppConstants.backendLabel(preferredBackend)}, ${AppConstants.formatTokenCount(safeMaxTokens)} tokens)...',
    );
    await _gemmaDatasource.createActiveModel(
      preferredBackend: preferredBackend,
      maxTokens: safeMaxTokens,
      supportImage: effectiveSupportImage,
    );
    await _gemmaDatasource.openChat(
      modelType: spec.modelType,
      supportImage: effectiveSupportImage,
      enableThinking: effectiveThinking,
      systemInstruction: responseStyle.systemInstruction,
    );

    // Sincroniza o contexto recente exclusivo desta conversa com o novo modelo carregado
    await _replayContextMessages(
      contextMessages,
      includeImages: effectiveSupportImage,
    );
  }

  @override
  void cancelDownload() {
    _gemmaDatasource.cancelActiveDownload();
  }

  @override
  Future<void> startNewChatSession({
    String? modelId,
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool? supportImage,
    bool enableThinking = false,
  }) async {
    final spec = _resolveSpec(modelId: modelId);
    final effectiveSupportImage = (supportImage ?? true) && spec.supportsImage;
    final effectiveThinking = enableThinking && spec.supportsThinking;

    // Descarta qualquer sessão anterior e inicia uma nova sessão limpa com o estilo desejado
    await _gemmaDatasource.openChat(
      modelType: spec.modelType,
      supportImage: effectiveSupportImage,
      enableThinking: effectiveThinking,
      systemInstruction: responseStyle.systemInstruction,
    );

    // Sincroniza apenas as mensagens exclusivas da conversa atual
    await _replayContextMessages(
      contextMessages,
      includeImages: effectiveSupportImage,
    );
  }

  Future<void> _replayContextMessages(
    List<ChatMessage>? contextMessages, {
    bool includeImages = true,
  }) async {
    if (contextMessages == null || contextMessages.isEmpty) return;

    final relevantMessages = contextMessages
        .where((m) => !m.text.startsWith('Sistema:') && m.text.trim().isNotEmpty)
        .toList();

    final recentContext = relevantMessages.length > 6
        ? relevantMessages.sublist(relevantMessages.length - 6)
        : relevantMessages;

    for (final msg in recentContext) {
      final replayMsg = includeImages
          ? msg
          : ChatMessage(text: msg.text, isUser: msg.isUser);
      await _gemmaDatasource.addQueryChunk(
        ChatMessageAdapter.toGemmaMessage(replayMsg),
      );
    }
  }

  @override
  Stream<ChatStreamChunk> sendMessage({
    required String text,
    Uint8List? imageBytes,
  }) async* {
    final message = ChatMessageAdapter.toGemmaMessage(
      ChatMessage(text: text, isUser: true, imageBytes: imageBytes),
    );

    await _gemmaDatasource.addQueryChunk(message);
    final responseStream = _gemmaDatasource.generateChatResponseAsync();

    await for (final chunk in responseStream) {
      switch (chunk) {
        case TextResponse(:final token):
          yield ChatTextChunk(token);
        case ThinkingResponse(:final content):
          yield ChatThinkingChunk(content);
        case FunctionCallResponse() || ParallelFunctionCallResponse():
          break;
      }
    }
  }

  @override
  Future<void> stopGeneration() async {
    await _gemmaDatasource.stopGeneration();
  }

  @override
  Future<void> reset() async {
    await _gemmaDatasource.close();
  }

  @override
  Future<List<Conversation>> getConversations() {
    return _historyDatasource.loadConversations();
  }

  @override
  Future<void> saveConversations(List<Conversation> conversations) {
    return _historyDatasource.saveConversations(conversations);
  }
}
