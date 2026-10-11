import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';

void main() {
  group('Conversation Model', () {
    test('should instantiate correctly and convert to/from JSON with responseStyle', () {
      final now = DateTime.now();
      final conv = Conversation(
        id: '123',
        title: 'Conversa Teste',
        messages: [
          ChatMessage(text: 'Olá', isUser: true),
        ],
        is2B: false,
        useCpu: false,
        responseStyle: ResponseStyle.detailed,
        createdAt: now,
        updatedAt: now,
      );

      final json = conv.toJson();
      final fromJson = Conversation.fromJson(json);

      expect(fromJson.id, '123');
      expect(fromJson.title, 'Conversa Teste');
      expect(fromJson.is2B, isFalse);
      expect(fromJson.useCpu, isFalse);
      expect(fromJson.responseStyle, ResponseStyle.detailed);
      expect(fromJson.messages.length, 1);
      expect(fromJson.messages.first.text, 'Olá');
    });

    test('copyWith should update specified attributes including responseStyle', () {
      final now = DateTime.now();
      final conv = Conversation(
        id: '1',
        title: 'Original',
        messages: [],
        createdAt: now,
        updatedAt: now,
      );

      final updated = conv.copyWith(
        title: 'Novo Título',
        is2B: false,
        responseStyle: ResponseStyle.brief,
      );

      expect(updated.id, '1');
      expect(updated.title, 'Novo Título');
      expect(updated.is2B, isFalse);
      expect(updated.useCpu, isTrue);
      expect(updated.responseStyle, ResponseStyle.brief);
    });
  });
}
