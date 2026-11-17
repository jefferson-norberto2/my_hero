import 'dart:typed_data';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/shared/models/chat_stream_chunk.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Repository contract for chat sessions, messaging, and Ollama model operations.
abstract class ChatRepository {
  String get ollamaBaseUrl;

  void updateOllamaBaseUrl(String url);

  Future<String> loadOllamaBaseUrl();

  Future<void> saveOllamaBaseUrl(String url);

  Future<ContextMode> loadContextMode();

  Future<int> loadManualContextLength();

  Future<bool> checkOllamaHealth([String? customUrl]);

  Future<List<OllamaModel>> getAvailableModels();

  Future<Map<String, dynamic>?> getLoadedModelInfo(String modelName);

  Future<int> getModelMaxContextLength(String modelName);

  Future<void> loadModel({
    required String modelName,
    required void Function(double progress, String status) onProgress,
  });

  Future<void> startNewChatSession({
    String? modelName,
  });

  Stream<ChatStreamChunk> sendMessage({
    required String text,
    Uint8List? imageBytes,
    ChatAttachment? attachment,
    required String modelName,
    required List<ChatMessage> history,
    int? numCtx,
    void Function(int promptTokens, int evalTokens)? onMetrics,
  });

  void stopGeneration();

  void reset();

  Future<void> initialize();

  Future<List<Conversation>> getConversations();

  Future<void> saveConversations(List<Conversation> conversations);
}
