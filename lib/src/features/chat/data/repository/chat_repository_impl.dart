import 'dart:typed_data';
import 'package:flutter_gemma/flutter_gemma.dart';
import '../../../../core/constants/app_constants.dart';
import '../adapter/chat_message_adapter.dart';
import '../adapter/conversation_model.dart';
import '../adapter/response_style.dart';
import '../datasource/chat_history_datasource.dart';
import '../datasource/gemma_local_datasource.dart';
import 'chat_repository.dart';

class ChatRepositoryImpl implements ChatRepository {
  final GemmaLocalDatasource _gemmaDatasource;
  final ChatHistoryDatasource _historyDatasource;

  ChatRepositoryImpl({
    required this._gemmaDatasource,
    required this._historyDatasource,
  });

  @override
  PreferredBackend? get activeBackend => _gemmaDatasource.activeBackend;

  @override
  Future<bool> isModelInstalled({required bool is2B}) {
    final spec = AppConstants.modelSpec(is2B: is2B);
    return _gemmaDatasource.isModelInstalled(spec.fileName);
  }

  @override
  Future<void> loadModel({
    required bool is2B,
    required PreferredBackend preferredBackend,
    required int maxTokens,
    required void Function(double progress, String status) onProgress,
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool supportImage = true,
    bool enableThinking = false,
  }) async {
    final spec = AppConstants.modelSpec(is2B: is2B);
    final safeMaxTokens = AppConstants.clampMaxTokens(maxTokens, is2B: is2B);

    onProgress(0.0, 'Verificando armazenamento local...');
    final alreadyInstalled =
        await _gemmaDatasource.isModelInstalled(spec.fileName);

    if (alreadyInstalled) {
      onProgress(1.0, 'Modelo ${spec.displayName} encontrado localmente.');
    } else {
      onProgress(0.0, 'Iniciando download de ${spec.displayName}...');
    }

    await _gemmaDatasource.installModelFromNetwork(
      url: spec.downloadUrl,
      modelType: spec.modelType,
      fileType: spec.fileType,
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
      supportImage: supportImage,
    );
    await _gemmaDatasource.createChat(
      modelType: spec.modelType,
      supportImage: supportImage,
      enableThinking: enableThinking,
      systemInstruction: responseStyle.systemInstruction,
    );

    // Sincroniza o contexto recente exclusivo desta conversa com o novo modelo carregado
    await _replayContextMessages(contextMessages);
  }

  @override
  void cancelDownload() {
    _gemmaDatasource.cancelActiveDownload();
  }

  @override
  Future<void> startNewChatSession({
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool supportImage = true,
    bool enableThinking = false,
  }) async {
    // Descarta qualquer sessão anterior e inicia uma nova sessão limpa com o estilo desejado
    await _gemmaDatasource.createChat(
      modelType: ModelType.gemma4,
      supportImage: supportImage,
      enableThinking: enableThinking,
      systemInstruction: responseStyle.systemInstruction,
    );

    // Sincroniza apenas as mensagens exclusivas da conversa atual
    await _replayContextMessages(contextMessages);
  }

  Future<void> _replayContextMessages(List<ChatMessage>? contextMessages) async {
    if (contextMessages == null || contextMessages.isEmpty) return;

    final relevantMessages = contextMessages
        .where((m) => !m.text.startsWith('Sistema:') && m.text.trim().isNotEmpty)
        .toList();

    final recentContext = relevantMessages.length > 6
        ? relevantMessages.sublist(relevantMessages.length - 6)
        : relevantMessages;

    for (final msg in recentContext) {
      await _gemmaDatasource.addQueryChunk(
        ChatMessageAdapter.toGemmaMessage(msg),
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
