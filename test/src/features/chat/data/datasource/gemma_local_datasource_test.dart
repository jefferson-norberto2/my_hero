import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';

void main() {
  late GemmaLocalDatasourceImpl datasource;

  setUp(() {
    datasource = GemmaLocalDatasourceImpl();
  });

  group('GemmaLocalDatasourceImpl state validation', () {
    test('activeBackend should be null before active model is initialized', () {
      expect(datasource.activeBackend, isNull);
    });

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

    test('stopGeneration should complete normally even when chat is null', () async {
      await expectLater(datasource.stopGeneration(), completes);
    });

    test('close should complete normally when called without active model or chat', () async {
      await expectLater(datasource.close(), completes);
    });

    test('cancelActiveDownload should not throw when no download is active', () {
      expect(() => datasource.cancelActiveDownload(), returnsNormally);
    });
  });
}
