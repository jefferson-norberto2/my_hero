import 'dart:typed_data';
import '../adapter/chat_message_adapter.dart';
import '../adapter/conversation_model.dart';
import '../adapter/response_style.dart';

abstract class ChatRepository {
  Future<void> loadModel({
    required bool is2B,
    required bool useCpu,
    required void Function(double progress, String status) onProgress,
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool supportImage = true,
  });

  Future<void> startNewChatSession({
    List<ChatMessage>? contextMessages,
    ResponseStyle responseStyle = ResponseStyle.simple,
    bool supportImage = true,
  });

  Stream<String> sendMessage({
    required String text,
    Uint8List? imageBytes,
  });

  void stopGeneration();

  void reset();

  Future<List<Conversation>> getConversations();

  Future<void> saveConversations(List<Conversation> conversations);
}
