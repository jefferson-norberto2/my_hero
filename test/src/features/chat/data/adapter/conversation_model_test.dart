import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';

void main() {
  group('Conversation Model', () {
    test('should instantiate correctly and convert to/from JSON with backend, maxTokens, thinking, vision and responseStyle', () {
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
        enableVision: false,
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
      expect(fromJson.enableVision, isFalse);
      expect(fromJson.responseStyle, ResponseStyle.detailed);
      expect(fromJson.messages.length, 1);
      expect(fromJson.messages.first.text, 'Olá');
    });

    test('fromJson should maintain backward compatibility with legacy useCpu JSON and default enableVision to true', () {
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
      expect(conv.enableVision, isTrue);
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

    test('should persist arbitrary catalog modelId and clamp maxTokens to that model', () {
      final now = DateTime.now();
      final qwenSpec = AppConstants.modelSpecById('qwen2.5-1.5b-instruct');
      final conv = Conversation(
        id: 'qwen-1',
        title: 'Chat Qwen',
        messages: [],
        modelId: qwenSpec.id,
        maxTokens: 32768, // Deve ser limitado ao teto de 4096 do q8-ekv4096
        createdAt: now,
        updatedAt: now,
      );

      expect(conv.modelId, qwenSpec.id);
      expect(conv.maxTokens, 4096);

      final restored = Conversation.fromJson(conv.toJson());
      expect(restored.modelId, qwenSpec.id);
      expect(restored.modelSpec.displayName, qwenSpec.displayName);
      expect(restored.maxTokens, 4096);
    });
  });
}
