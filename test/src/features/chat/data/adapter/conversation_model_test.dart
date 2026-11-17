import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';

void main() {
  group('Conversation Model', () {
    test('should instantiate correctly and convert to/from JSON with modelName', () {
      final now = DateTime.now();
      final conv = Conversation(
        id: '123',
        title: 'Conversa Teste',
        messages: [
          ChatMessage(text: 'Olá', isUser: true),
        ],
        modelName: 'deepseek-r1:8b',
        createdAt: now,
        updatedAt: now,
      );

      final json = conv.toJson();
      final fromJson = Conversation.fromJson(json);

      expect(fromJson.id, '123');
      expect(fromJson.title, 'Conversa Teste');
      expect(fromJson.modelName, 'deepseek-r1:8b');
      expect(json.containsKey('responseStyle'), isFalse);
      expect(fromJson.messages.length, 1);
      expect(fromJson.messages.first.text, 'Olá');
    });

    test('copyWith should update specified attributes including modelName', () {
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
      );

      expect(updated.id, '1');
      expect(updated.title, 'Novo Título');
      expect(updated.modelName, 'mistral:latest');
    });

    test('backward compatibility: parses legacy JSON with legacy responseStyle key and without modelName', () {
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
