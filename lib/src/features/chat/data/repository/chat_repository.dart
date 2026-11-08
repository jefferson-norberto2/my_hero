import 'dart:typed_data';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import '../adapter/chat_message_adapter.dart';
import '../adapter/conversation_model.dart';
import '../adapter/ollama_model.dart';
import '../adapter/response_style.dart';

abstract class ChatRepository {
  Future<bool> checkOllamaHealth();

  Future<List<OllamaModel>> getAvailableModels();

  Future<void> loadModel({
    required String modelName,
    required void Function(double progress, String status) onProgress,
  });

  Future<void> startNewChatSession({
    String? modelName,
    ResponseStyle responseStyle = ResponseStyle.simple,
  });

  Stream<String> sendMessage({
    required String text,
    Uint8List? imageBytes,
    ChatAttachment? attachment,
    required String modelName,
    required List<ChatMessage> history,
    ResponseStyle responseStyle = ResponseStyle.simple,
  });

  void stopGeneration();

  void reset();

  Future<List<Conversation>> getConversations();

  Future<void> saveConversations(List<Conversation> conversations);
}
