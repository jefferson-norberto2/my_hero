import 'dart:typed_data';
import 'package:flutter_gemma/flutter_gemma.dart';
import '../adapter/chat_message_adapter.dart';
import '../adapter/conversation_model.dart';
import '../adapter/response_style.dart';
import '../datasource/chat_history_datasource.dart';
import '../datasource/gemma_local_datasource.dart';
import '../datasource/model_download_datasource.dart';
import 'chat_repository.dart';

class ChatRepositoryImpl implements ChatRepository {
  final GemmaLocalDatasource _gemmaDatasource;
  final ModelDownloadDatasource _downloadDatasource;
  final ChatHistoryDatasource _historyDatasource;

  ChatRepositoryImpl({
    required GemmaLocalDatasource gemmaDatasource,
    required ModelDownloadDatasource downloadDatasource,
    required ChatHistoryDatasource historyDatasource,
  })  : _gemmaDatasource = gemmaDatasource,
        _downloadDatasource = downloadDatasource,
        _historyDatasource = historyDatasource;

  @override
  Future<void> loadModel({
    required bool is2B,
    required bool useCpu,
    required void Function(double progress, String status) onProgress,
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
  }) async {
    onProgress(0.0, 'Verificando armazenamento local...');
    final filePath = await _downloadDatasource.downloadModel(
      is2B: is2B,
      onProgress: onProgress,
    );

    onProgress(1.0, 'Carregando modelo na memória...');
    await _gemmaDatasource.installModelFromFile(filePath);
    await _gemmaDatasource.createActiveModel(useCpu: useCpu);
    await _gemmaDatasource.createChat(
      systemInstruction: responseStyle.systemInstruction,
    );

    // Sincroniza o contexto recente exclusivo desta conversa com o novo modelo carregado
    await _replayContextMessages(contextMessages);
  }

  @override
  Future<void> startNewChatSession({
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
  }) async {
    // Descarta qualquer sessão anterior e inicia uma nova sessão limpa com o estilo desejado
    await _gemmaDatasource.createChat(
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
  Stream<String> sendMessage({
    required String text,
    Uint8List? imageBytes,
  }) async* {
    final message = ChatMessageAdapter.toGemmaMessage(
      ChatMessage(text: text, isUser: true, imageBytes: imageBytes),
    );

    await _gemmaDatasource.addQueryChunk(message);
    final responseStream = _gemmaDatasource.generateChatResponseAsync();

    await for (final chunk in responseStream) {
      if (chunk is TextResponse) {
        yield chunk.token;
      }
    }
  }

  @override
  void stopGeneration() {
    _gemmaDatasource.stopGeneration();
  }

  @override
  void reset() {
    _gemmaDatasource.close();
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
