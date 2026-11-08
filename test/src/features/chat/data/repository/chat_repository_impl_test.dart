import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository_impl.dart';

class MockOllamaRemoteDatasource extends Mock implements OllamaRemoteDatasource {}
class MockChatHistoryDatasource extends Mock implements ChatHistoryDatasource {}

void main() {
  late MockOllamaRemoteDatasource mockOllamaDatasource;
  late MockChatHistoryDatasource mockHistoryDatasource;
  late ChatRepositoryImpl repository;

  setUp(() {
    mockOllamaDatasource = MockOllamaRemoteDatasource();
    mockHistoryDatasource = MockChatHistoryDatasource();
    repository = ChatRepositoryImpl(
      ollamaDatasource: mockOllamaDatasource,
      historyDatasource: mockHistoryDatasource,
    );
  });

  group('ChatRepositoryImpl - checkOllamaHealth and getAvailableModels', () {
    test('checkOllamaHealth should delegate to datasource', () async {
      when(() => mockOllamaDatasource.checkHealth()).thenAnswer((_) async => true);

      final result = await repository.checkOllamaHealth();

      expect(result, isTrue);
      verify(() => mockOllamaDatasource.checkHealth()).called(1);
    });

    test('getAvailableModels should delegate to datasource', () async {
      final mockModels = [
        const OllamaModel(
          name: 'llama3.2:3b',
          model: 'llama3.2:3b',
          size: 2000000000,
          parameterSize: '3.2B',
        ),
      ];

      when(() => mockOllamaDatasource.getAvailableModels()).thenAnswer((_) async => mockModels);

      final result = await repository.getAvailableModels();

      expect(result, mockModels);
      verify(() => mockOllamaDatasource.getAvailableModels()).called(1);
    });
  });

  group('ChatRepositoryImpl - loadModel', () {
    test('should preload model into memory if Ollama is healthy', () async {
      when(() => mockOllamaDatasource.checkHealth()).thenAnswer((_) async => true);
      when(() => mockOllamaDatasource.preloadModel('llama3.2:3b')).thenAnswer((_) async {});

      final progresses = <double>[];
      final statuses = <String>[];

      await repository.loadModel(
        modelName: 'llama3.2:3b',
        onProgress: (p, s) {
          progresses.add(p);
          statuses.add(s);
        },
      );

      expect(progresses, contains(1.0));
      verify(() => mockOllamaDatasource.preloadModel('llama3.2:3b')).called(1);
    });

    test('should throw StateError if Ollama is unhealthy', () async {
      when(() => mockOllamaDatasource.checkHealth()).thenAnswer((_) async => false);

      expect(
        () => repository.loadModel(
          modelName: 'llama3.2:3b',
          onProgress: (_, _) {},
        ),
        throwsA(isA<StateError>()),
      );

      verifyNever(() => mockOllamaDatasource.preloadModel(any()));
    });
  });

  group('ChatRepositoryImpl - sendMessage', () {
    test('should format messages correctly and stream tokens from Ollama', () async {
      when(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: any(named: 'messages'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Olá, ', 'mundo!']));

      final stream = repository.sendMessage(
        text: 'Oi',
        modelName: 'llama3.2:3b',
        history: [],
        responseStyle: ResponseStyle.simple,
      );

      final tokens = await stream.toList();

      expect(tokens, ['Olá, ', 'mundo!']);
      verify(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: any(named: 'messages'),
        ),
      ).called(1);
    });

    test('should support sending message with imageBytes', () async {
      final fakeBytes = Uint8List.fromList([1, 2, 3]);

      when(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'lfm3-vl:latest',
          messages: any(named: 'messages'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Imagem descrita.']));

      final tokens = await repository.sendMessage(
        text: 'Descreva',
        imageBytes: fakeBytes,
        modelName: 'lfm3-vl:latest',
        history: [],
      ).toList();

      expect(tokens, ['Imagem descrita.']);
    });

    test('should support sending message with document ChatAttachment', () async {
      final docAttachment = ChatAttachment(
        name: 'codigo.dart',
        type: AttachmentType.document,
        bytes: Uint8List.fromList([1, 2]),
        textContent: 'void main() {}',
      );

      when(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: any(named: 'messages'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Código analisado.']));

      final tokens = await repository.sendMessage(
        text: 'Explique',
        attachment: docAttachment,
        modelName: 'llama3.2:3b',
        history: [],
      ).toList();

      expect(tokens, ['Código analisado.']);
    });
  });

  group('ChatRepositoryImpl - stopGeneration and reset', () {
    test('stopGeneration should call datasource stopGeneration', () {
      when(() => mockOllamaDatasource.stopGeneration()).thenReturn(null);

      repository.stopGeneration();

      verify(() => mockOllamaDatasource.stopGeneration()).called(1);
    });

    test('reset should stop generation', () {
      when(() => mockOllamaDatasource.stopGeneration()).thenReturn(null);

      repository.reset();

      verify(() => mockOllamaDatasource.stopGeneration()).called(1);
    });
  });

  group('ChatRepositoryImpl - history', () {
    test('getConversations should delegate to historyDatasource', () async {
      final mockList = [
        Conversation(
          id: '1',
          title: 'Conv 1',
          messages: [],
          modelName: 'llama3.2:3b',
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
          modelName: 'llama3.2:3b',
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
