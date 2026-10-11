import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';

void main() {
  late GemmaLocalDatasourceImpl datasource;

  setUp(() {
    datasource = GemmaLocalDatasourceImpl();
  });

  group('GemmaLocalDatasourceImpl state validation', () {
    test('createChat should throw StateError when active model is not initialized', () async {
      expect(
        () => datasource.createChat(),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Modelo ativo do Gemma ainda não foi inicializado'),
          ),
        ),
      );
    });

    test('addQueryChunk should throw StateError when active chat is not initialized', () async {
      final message = Message.text(text: 'Teste', isUser: true);

      expect(
        () => datasource.addQueryChunk(message),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Chat do Gemma não foi inicializado'),
          ),
        ),
      );
    });

    test('generateChatResponseAsync should throw StateError when active chat is not initialized', () {
      expect(
        () => datasource.generateChatResponseAsync(),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Chat do Gemma não foi inicializado'),
          ),
        ),
      );
    });

    test('stopGeneration should not throw even when chat is null', () {
      expect(() => datasource.stopGeneration(), returnsNormally);
    });

    test('close should not throw when called without active model or chat', () {
      expect(() => datasource.close(), returnsNormally);
    });
  });
}
