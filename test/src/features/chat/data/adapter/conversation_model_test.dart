import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';

void main() {
  group('Conversation Model', () {
    test('should instantiate correctly and convert to/from JSON with backend, maxTokens, thinking and responseStyle', () {
      final now = DateTime.now();
      final conv = Conversation(
        id: '123',
        title: 'Conversa Teste',
        messages: [
          ChatMessage(text: 'Olá', isUser: true),
        ],
        is2B: false,
        preferredBackend: PreferredBackend.npu,
        maxTokens: 16384,
        enableThinking: true,
        responseStyle: ResponseStyle.detailed,
        createdAt: now,
        updatedAt: now,
      );

      final json = conv.toJson();
      final fromJson = Conversation.fromJson(json);

      expect(fromJson.id, '123');
      expect(fromJson.title, 'Conversa Teste');
      expect(fromJson.is2B, isFalse);
      expect(fromJson.preferredBackend, PreferredBackend.npu);
      expect(fromJson.useCpu, isFalse);
      expect(fromJson.maxTokens, 16384);
      expect(fromJson.enableThinking, isTrue);
      expect(fromJson.responseStyle, ResponseStyle.detailed);
      expect(fromJson.messages.length, 1);
      expect(fromJson.messages.first.text, 'Olá');
    });

    test('fromJson should maintain backward compatibility with legacy useCpu JSON', () {
      final now = DateTime.now();
      final legacyJson = {
        'id': 'legacy',
        'title': 'Antiga',
        'messages': <dynamic>[],
        'is2B': true,
        'useCpu': false,
        'responseStyle': 'simple',
        'createdAt': now.toIso8601String(),
        'updatedAt': now.toIso8601String(),
      };

      final conv = Conversation.fromJson(legacyJson);

      expect(conv.preferredBackend, PreferredBackend.gpu);
      expect(conv.enableThinking, isFalse);
      expect(conv.maxTokens, greaterThanOrEqualTo(1024));
    });

    test('copyWith should update specified attributes including backend, maxTokens, thinking and responseStyle', () {
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
        preferredBackend: PreferredBackend.gpu,
        maxTokens: 8192,
        enableThinking: true,
        responseStyle: ResponseStyle.brief,
      );

      expect(updated.id, '1');
      expect(updated.title, 'Novo Título');
      expect(updated.is2B, isFalse);
      expect(updated.preferredBackend, PreferredBackend.gpu);
      expect(updated.useCpu, isFalse);
      expect(updated.maxTokens, 8192);
      expect(updated.enableThinking, isTrue);
      expect(updated.responseStyle, ResponseStyle.brief);
    });
  });
}
