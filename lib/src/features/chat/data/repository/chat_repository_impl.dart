import 'dart:typed_data';
import '../adapter/chat_message_adapter.dart';
import '../adapter/conversation_model.dart';
import '../adapter/ollama_model.dart';
import '../adapter/response_style.dart';
import '../datasource/chat_history_datasource.dart';
import '../datasource/ollama_remote_datasource.dart';
import 'chat_repository.dart';

class ChatRepositoryImpl implements ChatRepository {
  final OllamaRemoteDatasource _ollamaDatasource;
  final ChatHistoryDatasource _historyDatasource;

  ChatRepositoryImpl({
    required OllamaRemoteDatasource ollamaDatasource,
    required ChatHistoryDatasource historyDatasource,
  })  : _ollamaDatasource = ollamaDatasource,
        _historyDatasource = historyDatasource;

  @override
  Future<bool> checkOllamaHealth() {
    return _ollamaDatasource.checkHealth();
  }

  @override
  Future<List<OllamaModel>> getAvailableModels() {
    return _ollamaDatasource.getAvailableModels();
  }

  @override
  Future<void> loadModel({
    required String modelName,
    required void Function(double progress, String status) onProgress,
  }) async {
    onProgress(0.2, 'Conectando ao Ollama...');
    final isHealthy = await _ollamaDatasource.checkHealth();
    if (!isHealthy) {
      throw StateError('Ollama não está respondendo em http://localhost:11434.');
    }

    onProgress(0.5, 'Carregando "$modelName" na memória...');
    await _ollamaDatasource.preloadModel(modelName);
    onProgress(1.0, 'Modelo "$modelName" pronto!');
  }

  @override
  Future<void> startNewChatSession({
    String? modelName,
    ResponseStyle responseStyle = ResponseStyle.simple,
  }) async {
    // Para Ollama, cada conversa é isolada pelo histórico de mensagens enviado na requisição
  }

  @override
  Stream<String> sendMessage({
    required String text,
    Uint8List? imageBytes,
    required String modelName,
    required List<ChatMessage> history,
    ResponseStyle responseStyle = ResponseStyle.simple,
  }) {
    final messages = <Map<String, dynamic>>[];

    // 1. Mensagem de sistema com instrução de estilo
    messages.add({
      'role': 'system',
      'content': responseStyle.systemInstruction,
    });

    // 2. Histórico recente de mensagens para contexto (excluindo mensagens de sistema do app)
    final relevantHistory = history
        .where((m) => !m.text.startsWith('Sistema:') && m.text.trim().isNotEmpty)
        .toList();

    // Mantém as últimas 10 mensagens para contexto ágil
    final contextHistory = relevantHistory.length > 10
        ? relevantHistory.sublist(relevantHistory.length - 10)
        : relevantHistory;

    for (final msg in contextHistory) {
      messages.add(ChatMessageAdapter.toOllamaMessage(msg));
    }

    // 3. Nova mensagem do usuário
    final userMessage = ChatMessage(
      text: text,
      isUser: true,
      imageBytes: imageBytes,
    );
    messages.add(ChatMessageAdapter.toOllamaMessage(userMessage));

    return _ollamaDatasource.generateChatResponseAsync(
      modelName: modelName,
      messages: messages,
    );
  }

  @override
  void stopGeneration() {
    _ollamaDatasource.stopGeneration();
  }

  @override
  void reset() {
    _ollamaDatasource.stopGeneration();
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
