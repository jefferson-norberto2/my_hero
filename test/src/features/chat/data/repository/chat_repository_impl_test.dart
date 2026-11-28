import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository_impl.dart';

class MockGemmaLocalDatasource extends Mock implements GemmaLocalDatasource {}
class MockChatHistoryDatasource extends Mock implements ChatHistoryDatasource {}
class FakeMessage extends Fake implements Message {}

void main() {
  late MockGemmaLocalDatasource mockGemmaDatasource;
  late MockChatHistoryDatasource mockHistoryDatasource;
  late ChatRepositoryImpl repository;

  setUpAll(() {
    registerFallbackValue(FakeMessage());
    registerFallbackValue(PreferredBackend.cpu);
    registerFallbackValue(ModelType.gemma4);
    registerFallbackValue(ModelFileType.litertlm);
    registerFallbackValue(AppConstants.model2bCatalogVariant);
  });

  setUp(() {
    mockGemmaDatasource = MockGemmaLocalDatasource();
    mockHistoryDatasource = MockChatHistoryDatasource();
    when(() => mockGemmaDatasource.close()).thenAnswer((_) async {});
    when(
      () => mockGemmaDatasource.activateInstalledModel(
        fileName: any(named: 'fileName'),
        modelType: any(named: 'modelType'),
        fileType: any(named: 'fileType'),
      ),
    ).thenAnswer((_) async {});
    when(() => mockGemmaDatasource.cleanupStorage()).thenAnswer((_) async {});

    repository = ChatRepositoryImpl(
      gemmaDatasource: mockGemmaDatasource,
      historyDatasource: mockHistoryDatasource,
    );
  });

  group('ChatRepositoryImpl - loadModel', () {
    test('should close active model, install via catalog when missing, create active model and openChat', () async {
      final recordedStatuses = <String>[];
      final recordedProgresses = <double>[];

      when(() => mockGemmaDatasource.isModelInstalled(AppConstants.model2bFileName))
          .thenAnswer((_) async => false);
      when(
        () => mockGemmaDatasource.installModelFromCatalog(
          variant: AppConstants.model2bCatalogVariant,
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((invocation) async {
        final onProgress =
            invocation.namedArguments[#onProgress] as void Function(int);
        onProgress(50);
      });
      when(
        () => mockGemmaDatasource.createActiveModel(
          preferredBackend: PreferredBackend.gpu,
          maxTokens: 8192,
          supportImage: any(named: 'supportImage'),
        ),
      ).thenAnswer((_) async {});
      when(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.gemma4,
          systemInstruction: any(named: 'systemInstruction'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenAnswer((_) async {});

      await repository.loadModel(
        is2B: true,
        preferredBackend: PreferredBackend.gpu,
        maxTokens: 8192,
        responseStyle: ResponseStyle.detailed,
        enableThinking: true,
        onProgress: (progress, status) {
          recordedProgresses.add(progress);
          recordedStatuses.add(status);
        },
      );

      expect(recordedStatuses, contains('Verificando armazenamento local...'));
      expect(recordedStatuses.any((s) => s.contains('50%')), isTrue);
      expect(recordedProgresses, contains(0.0));
      expect(recordedProgresses, contains(0.5));
      expect(recordedProgresses, contains(1.0));

      verify(() => mockGemmaDatasource.close()).called(1);
      verify(
        () => mockGemmaDatasource.installModelFromCatalog(
          variant: AppConstants.model2bCatalogVariant,
          onProgress: any(named: 'onProgress'),
        ),
      ).called(1);
      verifyNever(
        () => mockGemmaDatasource.activateInstalledModel(
          fileName: any(named: 'fileName'),
          modelType: any(named: 'modelType'),
          fileType: any(named: 'fileType'),
        ),
      );
      verify(
        () => mockGemmaDatasource.createActiveModel(
          preferredBackend: PreferredBackend.gpu,
          maxTokens: 8192,
          supportImage: true,
        ),
      ).called(1);
      verify(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.gemma4,
          systemInstruction: ResponseStyle.detailed.systemInstruction,
          supportImage: true,
          enableThinking: true,
        ),
      ).called(1);
    });

    test('should activate already-installed model directly without invoking installModelFromCatalog and respect supportImage=false', () async {
      when(() => mockGemmaDatasource.isModelInstalled(AppConstants.model4bFileName))
          .thenAnswer((_) async => true);
      when(
        () => mockGemmaDatasource.createActiveModel(
          preferredBackend: PreferredBackend.cpu,
          maxTokens: AppConstants.maxContextTokens,
          supportImage: false,
        ),
      ).thenAnswer((_) async {});
      when(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.gemma4,
          systemInstruction: any(named: 'systemInstruction'),
          supportImage: false,
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.addQueryChunk(any()))
          .thenAnswer((_) async {});

      final messages = [
        ChatMessage(text: 'Sistema: Carregado', isUser: false),
        ChatMessage(text: 'Primeira pergunta', isUser: true),
        ChatMessage(text: 'Primeira resposta', isUser: false),
      ];

      await repository.loadModel(
        is2B: false,
        preferredBackend: PreferredBackend.cpu,
        maxTokens: 999999,
        supportImage: false,
        onProgress: (progress, status) {},
        contextMessages: messages,
      );

      verify(() => mockGemmaDatasource.close()).called(1);
      verify(
        () => mockGemmaDatasource.activateInstalledModel(
          fileName: AppConstants.model4bFileName,
          modelType: AppConstants.model4b.modelType,
          fileType: AppConstants.model4b.fileType,
        ),
      ).called(1);
      verifyNever(
        () => mockGemmaDatasource.installModelFromCatalog(
          variant: any(named: 'variant'),
          onProgress: any(named: 'onProgress'),
        ),
      );
      verify(
        () => mockGemmaDatasource.createActiveModel(
          preferredBackend: PreferredBackend.cpu,
          maxTokens: AppConstants.maxContextTokens,
          supportImage: false,
        ),
      ).called(1);
      verify(() => mockGemmaDatasource.addQueryChunk(any())).called(2);
    });

    test('should rethrow error when installModelFromCatalog fails', () async {
      when(() => mockGemmaDatasource.isModelInstalled(AppConstants.model4bFileName))
          .thenAnswer((_) async => false);
      when(
        () => mockGemmaDatasource.installModelFromCatalog(
          variant: AppConstants.model4bCatalogVariant,
          onProgress: any(named: 'onProgress'),
        ),
      ).thenThrow(Exception('Falha de conexão'));

      expect(
        () => repository.loadModel(
          is2B: false,
          preferredBackend: PreferredBackend.cpu,
          maxTokens: 4096,
          onProgress: (_, _) {},
        ),
        throwsA(isA<Exception>()),
      );

      verifyNever(
        () => mockGemmaDatasource.createActiveModel(
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
        ),
      );
    });
  });

  group('ChatRepositoryImpl - sendMessage', () {
    test('should stream ChatThinkingChunk and ChatTextChunk from ModelResponse', () async {
      when(() => mockGemmaDatasource.addQueryChunk(any()))
          .thenAnswer((_) async {});
      when(() => mockGemmaDatasource.generateChatResponseAsync()).thenAnswer(
        (_) => Stream<ModelResponse>.fromIterable([
          const ThinkingResponse('Analisando pergunta...'),
          const TextResponse('Olá, '),
          const TextResponse('tudo bem?'),
          const FunctionCallResponse(name: 'noop', args: {}),
        ]),
      );

      final chunks = await repository.sendMessage(text: 'Oi').toList();

      expect(chunks.length, 3);
      expect(chunks[0], isA<ChatThinkingChunk>());
      expect((chunks[0] as ChatThinkingChunk).content, 'Analisando pergunta...');
      expect(chunks[1], isA<ChatTextChunk>());
      expect((chunks[1] as ChatTextChunk).token, 'Olá, ');
      expect(chunks[2], isA<ChatTextChunk>());
      expect((chunks[2] as ChatTextChunk).token, 'tudo bem?');
      verify(() => mockGemmaDatasource.addQueryChunk(any())).called(1);
      verify(() => mockGemmaDatasource.generateChatResponseAsync()).called(1);
    });

    test('should support sending message with imageBytes', () async {
      final fakeBytes = Uint8List.fromList([1, 2, 3]);

      when(() => mockGemmaDatasource.addQueryChunk(any()))
          .thenAnswer((_) async {});
      when(() => mockGemmaDatasource.generateChatResponseAsync()).thenAnswer(
        (_) => Stream<ModelResponse>.fromIterable([
          const TextResponse('Foto analisada.'),
        ]),
      );

      final chunks = await repository
          .sendMessage(
            text: 'Descreva',
            imageBytes: fakeBytes,
          )
          .toList();

      expect(chunks.length, 1);
      expect((chunks.first as ChatTextChunk).token, 'Foto analisada.');
      verify(() => mockGemmaDatasource.addQueryChunk(any())).called(1);
    });
  });

  group('ChatRepositoryImpl - stopGeneration, reset and cancelDownload', () {
    test('stopGeneration should await gemma datasource stopGeneration', () async {
      when(() => mockGemmaDatasource.stopGeneration()).thenAnswer((_) async {});

      await repository.stopGeneration();

      verify(() => mockGemmaDatasource.stopGeneration()).called(1);
    });

    test('reset should await gemma datasource close', () async {
      await repository.reset();

      verify(() => mockGemmaDatasource.close()).called(1);
    });

    test('cancelDownload should delegate to gemma datasource cancelActiveDownload', () {
      when(() => mockGemmaDatasource.cancelActiveDownload()).thenReturn(null);

      repository.cancelDownload();

      verify(() => mockGemmaDatasource.cancelActiveDownload()).called(1);
    });

    test('activeBackend should expose datasource activeBackend', () {
      when(() => mockGemmaDatasource.activeBackend)
          .thenReturn(PreferredBackend.gpu);

      expect(repository.activeBackend, PreferredBackend.gpu);
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

      when(() => mockHistoryDatasource.loadConversations())
          .thenAnswer((_) async => mockList);

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

      when(() => mockHistoryDatasource.saveConversations(mockList))
          .thenAnswer((_) async {});

      await repository.saveConversations(mockList);

      verify(() => mockHistoryDatasource.saveConversations(mockList)).called(1);
    });
  });

  group('ChatRepositoryImpl - startNewChatSession', () {
    test('should open fresh chat with brief style instruction and thinking flag', () async {
      when(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.gemma4,
          systemInstruction: any(named: 'systemInstruction'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenAnswer((_) async {});

      await repository.startNewChatSession(
        contextMessages: null,
        responseStyle: ResponseStyle.brief,
        enableThinking: true,
      );

      verify(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.gemma4,
          systemInstruction: ResponseStyle.brief.systemInstruction,
          supportImage: true,
          enableThinking: true,
        ),
      ).called(1);
      verifyNever(() => mockGemmaDatasource.addQueryChunk(any()));
    });

    test('should open fresh chat and replay only non-system messages when contextMessages provided', () async {
      when(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.gemma4,
          systemInstruction: any(named: 'systemInstruction'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.addQueryChunk(any()))
          .thenAnswer((_) async {});

      final messages = [
        ChatMessage(text: 'Sistema: Aviso', isUser: false),
        ChatMessage(text: 'Meu nome é Alice', isUser: true),
        ChatMessage(text: 'Prazer, Alice!', isUser: false),
      ];

      await repository.startNewChatSession(
        contextMessages: messages,
        responseStyle: ResponseStyle.detailed,
      );

      verify(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.gemma4,
          systemInstruction: ResponseStyle.detailed.systemInstruction,
          supportImage: true,
          enableThinking: false,
        ),
      ).called(1);
      verify(() => mockGemmaDatasource.addQueryChunk(any())).called(2);
    });
  });

  group('ChatRepositoryImpl - open catalog & storage management', () {
    test('should load text-only catalog model with supportImage=false and strip image bytes during replay', () async {
      final qwenSpec = AppConstants.modelSpecById('qwen3-0.6b');
      final fakeBytes = Uint8List.fromList([1, 2, 3]);

      when(() => mockGemmaDatasource.isModelInstalled(qwenSpec.fileName))
          .thenAnswer((_) async => false);
      when(
        () => mockGemmaDatasource.installModelFromCatalog(
          variant: qwenSpec.catalogVariant,
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((_) async {});
      when(
        () => mockGemmaDatasource.createActiveModel(
          preferredBackend: PreferredBackend.cpu,
          maxTokens: 4096,
          supportImage: false,
        ),
      ).thenAnswer((_) async {});
      when(
        () => mockGemmaDatasource.openChat(
          modelType: ModelType.qwen3,
          systemInstruction: any(named: 'systemInstruction'),
          supportImage: false,
          enableThinking: true,
        ),
      ).thenAnswer((_) async {});
      when(() => mockGemmaDatasource.addQueryChunk(any()))
          .thenAnswer((_) async {});

      await repository.loadModel(
        modelId: qwenSpec.id,
        preferredBackend: PreferredBackend.cpu,
        maxTokens: 4096,
        enableThinking: true,
        contextMessages: [
          ChatMessage(text: 'Olhe esta foto', isUser: true, imageBytes: fakeBytes),
        ],
        onProgress: (_, _) {},
      );

      final captured = verify(
        () => mockGemmaDatasource.addQueryChunk(captureAny()),
      ).captured;
      expect((captured.single as Message).imageBytes, isNull);
    });

    test('getInstalledModelFileNames, uninstallModel and cleanupStorage should delegate to datasource', () async {
      when(() => mockGemmaDatasource.listInstalledModels()).thenAnswer(
        (_) async => [AppConstants.model2bFileName, 'Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm'],
      );
      when(() => mockGemmaDatasource.uninstallModel(AppConstants.model2bFileName))
          .thenAnswer((_) async {});

      final installed = await repository.getInstalledModelFileNames();
      expect(installed, contains(AppConstants.model2bFileName));
      expect(installed.length, 2);

      await repository.uninstallModel(AppConstants.model2bFileName);
      verify(() => mockGemmaDatasource.uninstallModel(AppConstants.model2bFileName))
          .called(1);

      await repository.cleanupStorage();
      verify(() => mockGemmaDatasource.cleanupStorage()).called(1);
    });
  });
}
