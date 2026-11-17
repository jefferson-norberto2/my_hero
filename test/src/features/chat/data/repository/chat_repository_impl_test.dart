import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository_impl.dart';
import 'package:my_hero/src/features/chat/data/services/foreground_task_manager.dart';
import 'package:my_hero/src/shared/models/chat_stream_chunk.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

class MockOllamaRemoteDatasource extends Mock implements OllamaRemoteDatasource {}
class MockChatHistoryDatasource extends Mock implements ChatHistoryDatasource {}
class MockForegroundTaskManager extends Mock implements ForegroundTaskManager {}

void main() {
  late MockOllamaRemoteDatasource mockOllamaDatasource;
  late MockChatHistoryDatasource mockHistoryDatasource;
  late ChatRepositoryImpl repository;

  setUp(() {
    mockOllamaDatasource = MockOllamaRemoteDatasource();
    mockHistoryDatasource = MockChatHistoryDatasource();
    when(() => mockOllamaDatasource.baseUrl).thenReturn('http://localhost:11434');
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
      ).thenAnswer(
        (_) => Stream.fromIterable(const [
          ChatStreamChunk(contentDelta: 'Olá, '),
          ChatStreamChunk(contentDelta: 'mundo!', isDone: true),
        ]),
      );

      final stream = repository.sendMessage(
        text: 'Oi',
        modelName: 'llama3.2:3b',
        history: [],
      );

      final chunks = await stream.toList();

      expect(
        chunks,
        const [
          ChatStreamChunk(contentDelta: 'Olá, '),
          ChatStreamChunk(contentDelta: 'mundo!', isDone: true),
        ],
      );
      verify(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: const [
            {'role': 'user', 'content': 'Oi'},
          ],
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
      ).thenAnswer(
        (_) => Stream.fromIterable(const [
          ChatStreamChunk(contentDelta: 'Imagem descrita.', isDone: true),
        ]),
      );

      final chunks = await repository.sendMessage(
        text: 'Descreva',
        imageBytes: fakeBytes,
        modelName: 'lfm3-vl:latest',
        history: [],
      ).toList();

      expect(
        chunks,
        const [ChatStreamChunk(contentDelta: 'Imagem descrita.', isDone: true)],
      );
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
      ).thenAnswer(
        (_) => Stream.fromIterable(const [
          ChatStreamChunk(contentDelta: 'Código analisado.', isDone: true),
        ]),
      );

      final chunks = await repository.sendMessage(
        text: 'Explique',
        attachment: docAttachment,
        modelName: 'llama3.2:3b',
        history: [],
      ).toList();

      expect(
        chunks,
        const [ChatStreamChunk(contentDelta: 'Código analisado.', isDone: true)],
      );
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
    test('initialize should delegate to historyDatasource', () async {
      when(() => mockHistoryDatasource.initialize()).thenAnswer((_) async {});

      await repository.initialize();

      verify(() => mockHistoryDatasource.initialize()).called(1);
    });

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

  group('ChatRepositoryImpl - Context Configuration', () {
    test('getModelMaxContextLength delegates to ollamaDatasource', () async {
      when(() => mockOllamaDatasource.getModelMaxContextLength('qwen2.5:7b'))
          .thenAnswer((_) async => 32768);

      final result = await repository.getModelMaxContextLength('qwen2.5:7b');

      expect(result, 32768);
      verify(() => mockOllamaDatasource.getModelMaxContextLength('qwen2.5:7b'))
          .called(1);
    });

    test('sendMessage forwards numCtx to generateChatResponseAsync', () async {
      when(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: any(named: 'modelName'),
          messages: any(named: 'messages'),
          numCtx: any(named: 'numCtx'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable(const [
          ChatStreamChunk(contentDelta: 'Ok', isDone: true),
        ]),
      );

      await repository.sendMessage(
        text: 'Olá',
        modelName: 'llama3.2:3b',
        history: const [],
        numCtx: 16384,
      ).toList();

      verify(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: any(named: 'messages'),
          numCtx: 16384,
        ),
      ).called(1);
    });
  });

  group('ChatRepositoryImpl - Foreground Service & Decoupled Stream', () {
    late MockForegroundTaskManager mockForegroundManager;
    late ChatRepositoryImpl managedRepository;

    setUp(() {
      mockForegroundManager = MockForegroundTaskManager();
      when(() => mockForegroundManager.beginChatGeneration())
          .thenAnswer((_) async {});
      when(() => mockForegroundManager.endChatGeneration())
          .thenAnswer((_) async {});
      when(() => mockOllamaDatasource.stopGeneration()).thenReturn(null);

      managedRepository = ChatRepositoryImpl(
        ollamaDatasource: mockOllamaDatasource,
        historyDatasource: mockHistoryDatasource,
        foregroundTaskManager: mockForegroundManager,
      );
    });

    test('starts foreground service on sendMessage and stops when stream completes', () async {
      when(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: any(named: 'messages'),
          numCtx: any(named: 'numCtx'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable(const [
          ChatStreamChunk(contentDelta: 'Hello', isDone: true),
        ]),
      );

      final chunks = await managedRepository.sendMessage(
        text: 'Hi',
        modelName: 'llama3.2:3b',
        history: const [],
      ).toList();

      expect(chunks.length, 1);
      verify(() => mockForegroundManager.beginChatGeneration()).called(1);
      verify(() => mockForegroundManager.endChatGeneration()).called(1);
    });

    test('stops foreground service when stream fails with an error', () async {
      when(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: any(named: 'messages'),
          numCtx: any(named: 'numCtx'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).thenAnswer((_) => Stream<ChatStreamChunk>.error(StateError('Socket lost')));

      await expectLater(
        managedRepository.sendMessage(
          text: 'Hi',
          modelName: 'llama3.2:3b',
          history: const [],
        ).toList(),
        throwsA(isA<StateError>()),
      );

      verify(() => mockForegroundManager.beginChatGeneration()).called(1);
      verify(() => mockForegroundManager.endChatGeneration()).called(1);
    });

    test('keeps upstream stream running if downstream listener cancels and ends service on stopGeneration', () async {
      final upstreamController = StreamController<ChatStreamChunk>();
      when(
        () => mockOllamaDatasource.generateChatResponseAsync(
          modelName: 'llama3.2:3b',
          messages: any(named: 'messages'),
          numCtx: any(named: 'numCtx'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).thenAnswer((_) => upstreamController.stream);

      final sub = managedRepository
          .sendMessage(
            text: 'Hi',
            modelName: 'llama3.2:3b',
            history: const [],
          )
          .listen((_) {});

      await Future<void>.delayed(Duration.zero);
      verify(() => mockForegroundManager.beginChatGeneration()).called(1);

      // Simulate UI listener detaching without calling stopGeneration().
      await sub.cancel();
      expect(upstreamController.hasListener, isTrue);
      verifyNever(() => mockForegroundManager.endChatGeneration());

      managedRepository.stopGeneration();
      await Future<void>.delayed(Duration.zero);

      verify(() => mockOllamaDatasource.stopGeneration()).called(1);
      verify(() => mockForegroundManager.endChatGeneration()).called(1);
      await upstreamController.close();
    });
  });
}
