import 'dart:typed_data';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/settings/data/datasource/settings_datasource.dart';
import 'package:my_hero/src/shared/models/chat_stream_chunk.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';

/// Default [ChatRepository] implementation backed by Ollama and local JSON storage.
class ChatRepositoryImpl implements ChatRepository {
  final OllamaRemoteDatasource _ollamaDatasource;
  final ChatHistoryDatasource _historyDatasource;
  final SettingsDatasource? _settingsDatasource;

  ChatRepositoryImpl({
    required OllamaRemoteDatasource ollamaDatasource,
    required ChatHistoryDatasource historyDatasource,
    SettingsDatasource? settingsDatasource,
  })  : _ollamaDatasource = ollamaDatasource,
        _historyDatasource = historyDatasource,
        _settingsDatasource = settingsDatasource;

  @override
  String get ollamaBaseUrl => _ollamaDatasource.baseUrl;

  @override
  void updateOllamaBaseUrl(String url) {
    _ollamaDatasource.updateBaseUrl(url);
  }

  @override
  Future<String> loadOllamaBaseUrl() async {
    if (_settingsDatasource != null) {
      final url = await _settingsDatasource.getOllamaBaseUrl();
      _ollamaDatasource.updateBaseUrl(url);
      return url;
    }
    return _ollamaDatasource.baseUrl;
  }

  @override
  Future<void> saveOllamaBaseUrl(String url) async {
    _ollamaDatasource.updateBaseUrl(url);
    if (_settingsDatasource != null) {
      await _settingsDatasource.saveOllamaBaseUrl(url);
    }
  }

  @override
  Future<ContextMode> loadContextMode() async {
    if (_settingsDatasource != null) {
      return _settingsDatasource.getContextMode();
    }
    return ContextMode.automatic;
  }

  @override
  Future<int> loadManualContextLength() async {
    if (_settingsDatasource != null) {
      return _settingsDatasource.getManualContextLength();
    }
    return ContextSettings.fallbackMaxContext;
  }

  @override
  Future<double?> loadServerMemoryLimitGb() async {
    if (_settingsDatasource != null) {
      return _settingsDatasource.getServerMemoryLimitGb();
    }
    return null;
  }

  @override
  Future<bool> checkOllamaHealth([String? customUrl]) {
    return _ollamaDatasource.checkHealth(customUrl);
  }

  @override
  Future<List<OllamaModel>> getAvailableModels() {
    return _ollamaDatasource.getAvailableModels();
  }

  @override
  Future<Map<String, dynamic>?> getLoadedModelInfo(String modelName) {
    return _ollamaDatasource.getLoadedModelInfo(modelName);
  }

  @override
  Future<int> getModelMaxContextLength(String modelName) {
    return _ollamaDatasource.getModelMaxContextLength(modelName);
  }

  @override
  Future<void> loadModel({
    required String modelName,
    required void Function(double progress, String status) onProgress,
  }) async {
    onProgress(0.2, 'Conectando ao Ollama...');
    final isHealthy = await _ollamaDatasource.checkHealth();
    if (!isHealthy) {
      throw StateError(
        'Ollama não está respondendo em ${_ollamaDatasource.baseUrl}.',
      );
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
    // Each Ollama conversation is isolated by the message history sent per request.
  }

  @override
  Stream<ChatStreamChunk> sendMessage({
    required String text,
    Uint8List? imageBytes,
    ChatAttachment? attachment,
    required String modelName,
    required List<ChatMessage> history,
    ResponseStyle responseStyle = ResponseStyle.simple,
    int? numCtx,
    void Function(int promptTokens, int evalTokens)? onMetrics,
  }) {
    final messages = <Map<String, dynamic>>[];

    // 1. System prompt enforcing response style.
    messages.add({
      'role': 'system',
      'content': responseStyle.systemInstruction,
    });

    // 2. Recent message history excluding internal app system banners.
    final relevantHistory = history
        .where((m) => !m.text.startsWith('Sistema:') && m.text.trim().isNotEmpty)
        .toList();

    // Keep up to the last 10 messages for responsive context size.
    final contextHistory = relevantHistory.length > 10
        ? relevantHistory.sublist(relevantHistory.length - 10)
        : relevantHistory;

    for (final msg in contextHistory) {
      messages.add(ChatMessageAdapter.toOllamaMessage(msg));
    }

    // 3. New user message.
    final effectiveImage = imageBytes ??
        (attachment != null && attachment.type == AttachmentType.image
            ? attachment.bytes
            : null);

    final userMessage = ChatMessage(
      text: text,
      isUser: true,
      imageBytes: effectiveImage,
      attachment: attachment,
    );
    messages.add(ChatMessageAdapter.toOllamaMessage(userMessage));

    return _ollamaDatasource.generateChatResponseAsync(
      modelName: modelName,
      messages: messages,
      numCtx: numCtx,
      onMetrics: onMetrics,
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
