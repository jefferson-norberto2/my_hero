import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import '../adapter/conversation_model.dart';

abstract class ChatHistoryDatasource {
  Future<List<Conversation>> loadConversations();
  Future<void> saveConversations(List<Conversation> conversations);
}

class ChatHistoryDatasourceImpl implements ChatHistoryDatasource {
  static const String _historyFileName = 'chat_conversations.json';

  Future<File> _getFile() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$_historyFileName');
  }

  @override
  Future<List<Conversation>> loadConversations() async {
    try {
      final file = await _getFile();
      if (!await file.exists()) {
        return [];
      }
      final content = await file.readAsString();
      if (content.trim().isEmpty) return [];

      final list = jsonDecode(content) as List<dynamic>;
      return list
          .map((item) => Conversation.fromJson(item as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  @override
  Future<void> saveConversations(List<Conversation> conversations) async {
    try {
      final file = await _getFile();
      final data = conversations.map((c) => c.toJson()).toList();
      await file.writeAsString(jsonEncode(data));
    } catch (_) {
      // Ignora falhas de escrita silenciosas ou propaga conforme necessário
    }
  }
}
