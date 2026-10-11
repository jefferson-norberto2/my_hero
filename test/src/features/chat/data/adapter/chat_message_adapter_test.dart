import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';

void main() {
  group('ChatMessage', () {
    test('should instantiate correctly with required parameters', () {
      final message = ChatMessage(text: 'Olá Gemma', isUser: true);

      expect(message.text, 'Olá Gemma');
      expect(message.isUser, isTrue);
      expect(message.imageBytes, isNull);
    });

    test('should instantiate correctly with imageBytes', () {
      final fakeBytes = Uint8List.fromList([1, 2, 3, 4]);
      final message = ChatMessage(
        text: 'Descreva a foto',
        isUser: true,
        imageBytes: fakeBytes,
      );

      expect(message.text, 'Descreva a foto');
      expect(message.isUser, isTrue);
      expect(message.imageBytes, fakeBytes);
    });

    test('copyWith should update specified fields and preserve others', () {
      final initialBytes = Uint8List.fromList([10, 20]);
      final message = ChatMessage(
        text: 'Texto inicial',
        isUser: false,
        imageBytes: initialBytes,
      );

      final updatedText = message.copyWith(text: 'Texto atualizado');
      expect(updatedText.text, 'Texto atualizado');
      expect(updatedText.isUser, isFalse);
      expect(updatedText.imageBytes, initialBytes);

      final updatedUser = message.copyWith(isUser: true);
      expect(updatedUser.text, 'Texto inicial');
      expect(updatedUser.isUser, isTrue);
      expect(updatedUser.imageBytes, initialBytes);

      final newBytes = Uint8List.fromList([99]);
      final updatedBytes = message.copyWith(imageBytes: newBytes);
      expect(updatedBytes.text, 'Texto inicial');
      expect(updatedBytes.isUser, isFalse);
      expect(updatedBytes.imageBytes, newBytes);
    });
  });

  group('ChatMessageAdapter', () {
    test('should convert ChatMessage without image to Message.text', () {
      final chatMessage = ChatMessage(text: 'Pergunta teste', isUser: true);

      final gemmaMessage = ChatMessageAdapter.toGemmaMessage(chatMessage);

      expect(gemmaMessage, isA<Message>());
      expect(gemmaMessage.text, 'Pergunta teste');
      expect(gemmaMessage.isUser, isTrue);
      expect(gemmaMessage.imageBytes, isNull);
    });

    test('should convert ChatMessage with imageBytes to Message with image', () {
      final imageBytes = Uint8List.fromList([1, 2, 3]);
      final chatMessage = ChatMessage(
        text: 'Veja a imagem',
        isUser: true,
        imageBytes: imageBytes,
      );

      final gemmaMessage = ChatMessageAdapter.toGemmaMessage(chatMessage);

      expect(gemmaMessage, isA<Message>());
      expect(gemmaMessage.text, 'Veja a imagem');
      expect(gemmaMessage.isUser, isTrue);
      expect(gemmaMessage.imageBytes, imageBytes);
    });

    test('should convert AI response ChatMessage correctly', () {
      final chatMessage = ChatMessage(
        text: 'Resposta gerada pela IA',
        isUser: false,
      );

      final gemmaMessage = ChatMessageAdapter.toGemmaMessage(chatMessage);

      expect(gemmaMessage.text, 'Resposta gerada pela IA');
      expect(gemmaMessage.isUser, isFalse);
      expect(gemmaMessage.imageBytes, isNull);
    });
  });
}
