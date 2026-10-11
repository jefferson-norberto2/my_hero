import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/model_download_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository_impl.dart';

class MockGemmaLocalDatasource extends Mock implements GemmaLocalDatasource {}
class MockModelDownloadDatasource extends Mock implements ModelDownloadDatasource {}
class MockChatHistoryDatasource extends Mock implements ChatHistoryDatasource {}
class FakeMessage extends Fake implements Message {}

void main() {
  late MockGemmaLocalDatasource mockGemmaDatasource;
  late MockModelDownloadDatasource mockDownloadDatasource;
  late MockChatHistoryDatasource mockHistoryDatasource;
  late ChatRepositoryImpl repository;

  setUpAll(() {
    registerFallbackValue(FakeMessage());
  });

  setUp(() {
    mockGemmaDatasource = MockGemmaLocalDatasource();
    mockDownloadDatasource = MockModelDownloadDatasource();
    mockHistoryDatasource = MockChatHistoryDatasource();
    repository = ChatRepositoryImpl(
      gemmaDatasource: mockGemmaDatasource,
      downloadDatasource: mockDownloadDatasource,
      historyDatasource: mockHistoryDatasource,
    );
  });

  group('ChatRepositoryImpl - loadModel', () {
    test('should download model, install it, create active model and chat successfully', () async {
      const filePath = '/mock/path/gemma_2b.litertlm';
      final recordedStatuses = <String>[];
      final recordedProgresses = <double>[];

      when(
        () => mockDownloadDatasource.downloadModel(
          is2B: true,
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((invocation) async {
        final onProgress = invocation.namedArguments[#onProgress] as void Function(double, String);
        onProgress(0.5, 'Baixando...');
        return filePath;
      });

      when(() => mockGemmaDatasource.installModelFromFile(filePath)).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.createActiveModel(useCpu: true)).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.createChat()).thenAnswer((_) async {});

      await repository.loadModel(
        is2B: true,
        useCpu: true,
        onProgress: (progress, status) {
          recordedProgresses.add(progress);
          recordedStatuses.add(status);
        },
      );

      expect(recordedStatuses, contains('Verificando armazenamento local...'));
      expect(recordedStatuses, contains('Baixando...'));
      expect(recordedStatuses, contains('Carregando modelo na memória...'));
      expect(recordedProgresses, contains(0.0));
      expect(recordedProgresses, contains(0.5));
      expect(recordedProgresses, contains(1.0));

      verify(
        () => mockDownloadDatasource.downloadModel(
          is2B: true,
          onProgress: any(named: 'onProgress'),
        ),
      ).called(1);
      verify(() => mockGemmaDatasource.installModelFromFile(filePath)).called(1);
      verify(() => mockGemmaDatasource.createActiveModel(useCpu: true)).called(1);
      verify(() => mockGemmaDatasource.createChat()).called(1);
    });

    test('should replay contextMessages when provided', () async {
      const filePath = '/mock/path/gemma_4b.litertlm';

      when(
        () => mockDownloadDatasource.downloadModel(
          is2B: false,
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((_) async => filePath);

      when(() => mockGemmaDatasource.installModelFromFile(filePath)).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.createActiveModel(useCpu: false)).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.createChat()).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.addQueryChunk(any())).thenAnswer((_) async {});

      final messages = [
        ChatMessage(text: 'Sistema: Carregado', isUser: false),
        ChatMessage(text: 'Primeira pergunta', isUser: true),
        ChatMessage(text: 'Primeira resposta', isUser: false),
      ];

      await repository.loadModel(
        is2B: false,
        useCpu: false,
        onProgress: (progress, status) {},
        contextMessages: messages,
      );

      // Não deve repassar mensagem de 'Sistema:', apenas 'Primeira pergunta' e 'Primeira resposta'
      verify(() => mockGemmaDatasource.addQueryChunk(any())).called(2);
    });

    test('should rethrow error when downloadModel fails', () async {
      when(
        () => mockDownloadDatasource.downloadModel(
          is2B: false,
          onProgress: any(named: 'onProgress'),
        ),
      ).thenThrow(Exception('Falha de conexão'));

      expect(
        () => repository.loadModel(
          is2B: false,
          useCpu: false,
          onProgress: (_, _) {},
        ),
        throwsA(isA<Exception>()),
      );

      verifyNever(() => mockGemmaDatasource.installModelFromFile(any()));
      verifyNever(() => mockGemmaDatasource.createActiveModel(useCpu: any(named: 'useCpu')));
      verifyNever(() => mockGemmaDatasource.createChat());
    });
  });

  group('ChatRepositoryImpl - sendMessage', () {
    test('should add query chunk and stream string tokens from TextResponse', () async {
      when(() => mockGemmaDatasource.addQueryChunk(any())).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.generateChatResponseAsync()).thenAnswer(
        (_) => Stream.fromIterable([
          TextResponse('Olá, '),
          TextResponse('tudo bem?'),
        ]),
      );

      final tokens = await repository.sendMessage(text: 'Oi').toList();

      expect(tokens, ['Olá, ', 'tudo bem?']);
      verify(() => mockGemmaDatasource.addQueryChunk(any())).called(1);
      verify(() => mockGemmaDatasource.generateChatResponseAsync()).called(1);
    });

    test('should support sending message with imageBytes', () async {
      final fakeBytes = Uint8List.fromList([1, 2, 3]);

      when(() => mockGemmaDatasource.addQueryChunk(any())).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.generateChatResponseAsync()).thenAnswer(
        (_) => Stream.fromIterable([
          TextResponse('Foto analisada.'),
        ]),
      );

      final tokens = await repository.sendMessage(
        text: 'Descreva',
        imageBytes: fakeBytes,
      ).toList();

      expect(tokens, ['Foto analisada.']);
      verify(() => mockGemmaDatasource.addQueryChunk(any())).called(1);
    });

    test('should ignore chunks that are not TextResponse', () async {
      when(() => mockGemmaDatasource.addQueryChunk(any())).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.generateChatResponseAsync()).thenAnswer(
        (_) => Stream.fromIterable([
          'chunk desconhecido',
          TextResponse('Texto válido'),
          123,
        ]),
      );

      final tokens = await repository.sendMessage(text: 'Teste').toList();

      expect(tokens, ['Texto válido']);
    });
  });

  group('ChatRepositoryImpl - stopGeneration and reset', () {
    test('stopGeneration should call gemma datasource stopGeneration', () {
      when(() => mockGemmaDatasource.stopGeneration()).thenReturn(null);

      repository.stopGeneration();

      verify(() => mockGemmaDatasource.stopGeneration()).called(1);
    });

    test('reset should call gemma datasource close', () {
      when(() => mockGemmaDatasource.close()).thenReturn(null);

      repository.reset();

      verify(() => mockGemmaDatasource.close()).called(1);
    });
  });

  group('ChatRepositoryImpl - history', () {
    test('getConversations should delegate to historyDatasource', () async {
      final mockList = [
        Conversation(
          id: '1',
          title: 'Conv 1',
          messages: [],
          createdAt: DateTime.now(),
          updatedAt: DateTime.now(),
        ),
      ];

      when(() => mockHistoryDatasource.loadConversations()).thenAnswer((_) async => mockList);

      final result = await repository.getConversations();

      expect(result, mockList);
      verify(() => mockHistoryDatasource.loadConversations()).called(1);
    });

    test('saveConversations should delegate to historyDatasource', () async {
      final mockList = [
        Conversation(
          id: '1',
          title: 'Conv 1',
          messages: [],
          createdAt: DateTime.now(),
          updatedAt: DateTime.now(),
        ),
      ];

      when(() => mockHistoryDatasource.saveConversations(mockList)).thenAnswer((_) async {});

      await repository.saveConversations(mockList);

      verify(() => mockHistoryDatasource.saveConversations(mockList)).called(1);
    });
  });
}
