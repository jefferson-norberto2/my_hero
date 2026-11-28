import 'dart:typed_data';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';

abstract class ChatRepository {
  PreferredBackend? get activeBackend;

  Future<bool> isModelInstalled({String? modelId, bool is2B = true});

  Future<Set<String>> getInstalledModelFileNames();

  Future<void> uninstallModel(String modelFileName);

  Future<void> cleanupStorage();

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
  });

  void cancelDownload();

  Future<void> startNewChatSession({
    String? modelId,
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool? supportImage,
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
