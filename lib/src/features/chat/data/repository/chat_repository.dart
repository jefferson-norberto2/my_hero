import 'dart:typed_data';
import 'package:flutter_gemma/flutter_gemma.dart';
import '../adapter/chat_message_adapter.dart';
import '../adapter/conversation_model.dart';
import '../adapter/response_style.dart';

abstract class ChatRepository {
  PreferredBackend? get activeBackend;

  Future<bool> isModelInstalled({required bool is2B});

  Future<void> loadModel({
    required bool is2B,
    required PreferredBackend preferredBackend,
    required int maxTokens,
    required void Function(double progress, String status) onProgress,
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool supportImage = true,
    bool enableThinking = false,
  });

  void cancelDownload();

  Future<void> startNewChatSession({
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool supportImage = true,
    bool enableThinking = false,
  });

  Stream<ChatStreamChunk> sendMessage({
    required String text,
    Uint8List? imageBytes,
  });

  Future<void> stopGeneration();

  Future<void> reset();

  Future<List<Conversation>> getConversations();

  Future<void> saveConversations(List<Conversation> conversations);
}
