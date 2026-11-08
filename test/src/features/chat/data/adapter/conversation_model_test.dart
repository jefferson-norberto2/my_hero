import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';

void main() {
  group('Conversation Model', () {
    test('should instantiate correctly and convert to/from JSON with responseStyle and modelName', () {
      final now = DateTime.now();
      final conv = Conversation(
        id: '123',
        title: 'Conversa Teste',
        messages: [
          ChatMessage(text: 'Olá', isUser: true),
        ],
        modelName: 'deepseek-r1:8b',
        responseStyle: ResponseStyle.detailed,
        createdAt: now,
        updatedAt: now,
      );

      final json = conv.toJson();
      final fromJson = Conversation.fromJson(json);

      expect(fromJson.id, '123');
      expect(fromJson.title, 'Conversa Teste');
      expect(fromJson.modelName, 'deepseek-r1:8b');
      expect(fromJson.responseStyle, ResponseStyle.detailed);
      expect(fromJson.messages.length, 1);
      expect(fromJson.messages.first.text, 'Olá');
    });

    test('copyWith should update specified attributes including responseStyle and modelName', () {
      final now = DateTime.now();
      final conv = Conversation(
        id: '1',
        title: 'Original',
        messages: [],
        modelName: 'llama3.2:3b',
        createdAt: now,
        updatedAt: now,
      );

      final updated = conv.copyWith(
        title: 'Novo Título',
        modelName: 'mistral:latest',
        responseStyle: ResponseStyle.brief,
      );

      expect(updated.id, '1');
      expect(updated.title, 'Novo Título');
      expect(updated.modelName, 'mistral:latest');
      expect(updated.responseStyle, ResponseStyle.brief);
    });

    test('backward compatibility: parses legacy JSON without modelName', () {
      final legacyJson = {
        'id': 'legacy-1',
        'title': 'Antiga',
        'messages': [],
        'is2B': true,
        'responseStyle': 'simple',
        'createdAt': DateTime.now().toIso8601String(),
        'updatedAt': DateTime.now().toIso8601String(),
      };

      final conv = Conversation.fromJson(legacyJson);
      expect(conv.id, 'legacy-1');
      expect(conv.modelName, isNotEmpty);
      expect(conv.is2B, isTrue);
    });
  });
}
