import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';

abstract class ChatHistoryDatasource {
  Future<List<Conversation>> loadConversations();
  Future<void> saveConversations(List<Conversation> conversations);
}

class ChatHistoryDatasourceImpl implements ChatHistoryDatasource {
  static const String _historyFileName = 'chat_conversations.json';
  final TargetPlatform? _platformOverride;

  ChatHistoryDatasourceImpl({TargetPlatform? platform})
      : _platformOverride = platform;

  bool get _isDesktop {
    if (kIsWeb) return false;
    final platform = _platformOverride ?? defaultTargetPlatform;
    return platform == TargetPlatform.linux ||
        platform == TargetPlatform.windows ||
        platform == TargetPlatform.macOS;
  }

  Future<File> _getPrimaryFile() async {
    final dir = _isDesktop
        ? await getApplicationSupportDirectory()
        : await getApplicationDocumentsDirectory();
    return File(p.join(dir.path, _historyFileName));
  }

  Future<File?> _getLegacyDesktopFile() async {
    if (!_isDesktop) return null;
    try {
      final legacyDir = await getApplicationDocumentsDirectory();
      final legacyFile = File(p.join(legacyDir.path, _historyFileName));
      if (await legacyFile.exists()) {
        return legacyFile;
      }
    } catch (_) {}
    return null;
  }

  @override
  Future<List<Conversation>> loadConversations() async {
    try {
      var file = await _getPrimaryFile();
      if (!await file.exists()) {
        final legacy = await _getLegacyDesktopFile();
        if (legacy == null) return [];
        file = legacy;
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
      final file = await _getPrimaryFile();
      if (!await file.parent.exists()) {
        await file.parent.create(recursive: true);
      }
      final data = conversations.map((c) => c.toJson()).toList();
      await file.writeAsString(jsonEncode(data));
    } catch (_) {
      // Ignora falhas de escrita silenciosas ou propaga conforme necessário
    }
  }
}
