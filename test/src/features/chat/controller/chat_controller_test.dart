import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/shared/models/chat_stream_chunk.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;
  late ChatController controller;

  setUpAll(() {
    registerFallbackValue(ResponseStyle.simple);
  });

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();

    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenReturn(null);
    when(() => mockRepository.checkOllamaHealth(any())).thenAnswer((_) async => true);
    when(() => mockRepository.checkOllamaHealth()).thenAnswer((_) async => true);
    when(() => mockRepository.loadOllamaBaseUrl()).thenAnswer((_) async => 'http://localhost:11434');
    when(() => mockRepository.updateOllamaBaseUrl(any())).thenReturn(null);
    when(() => mockRepository.saveOllamaBaseUrl(any())).thenAnswer((_) async {});
    when(() => mockRepository.getAvailableModels()).thenAnswer((_) async => [
      const OllamaModel(
        name: 'llama3.2:3b',
        model: 'llama3.2:3b',
        size: 2000000000,
        parameterSize: '3.2B',
      ),
    ]);

    controller = ChatController(
      repository: mockRepository,
      mediaPickerService: mockMediaPicker,
    );
    controller.selectedModelName.value = 'llama3.2:3b';
  });

  group('ChatController - Attachments (Image, Audio, Document)', () {
    test('initial attachedAttachment should be null', () {
      expect(controller.attachedAttachment.value, isNull);
    });

    test('setAttachment and clearAttachment should update state correctly', () {
      final fakeBytes = Uint8List.fromList([1, 2, 3, 4]);
      final attachment = ChatAttachment(
        name: 'teste.png',
        type: AttachmentType.image,
        bytes: fakeBytes,
      );

      controller.setAttachment(attachment);
      expect(controller.attachedAttachment.value, attachment);
      expect(controller.attachedImage.value, fakeBytes);

      controller.clearAttachment();
      expect(controller.attachedAttachment.value, isNull);
      expect(controller.attachedImage.value, isNull);
    });

    test('pickImage from gallery should store image in attachedAttachment', () async {
      final fakeBytes = Uint8List.fromList([10, 20, 30]);
      final attachment = ChatAttachment(
        name: 'foto.png',
        type: AttachmentType.image,
        bytes: fakeBytes,
      );
      when(() => mockMediaPicker.pickImageFromGallery()).thenAnswer((_) async => attachment);

      await controller.pickImage(fromCamera: false);

      expect(controller.attachedAttachment.value, attachment);
      verify(() => mockMediaPicker.pickImageFromGallery()).called(1);
    });

    test('pickAudio should store audio in attachedAttachment', () async {
      final fakeBytes = Uint8List.fromList([50, 60]);
      final attachment = ChatAttachment(
        name: 'gravacao.mp3',
        type: AttachmentType.audio,
        bytes: fakeBytes,
      );
      when(() => mockMediaPicker.pickAudioFile()).thenAnswer((_) async => attachment);

      await controller.pickAudio();

      expect(controller.attachedAttachment.value, attachment);
      expect(controller.attachedAttachment.value?.type, AttachmentType.audio);
      verify(() => mockMediaPicker.pickAudioFile()).called(1);
    });

    test('pickDocument should store document with textContent in attachedAttachment', () async {
      final fakeBytes = Uint8List.fromList([70, 80]);
      final attachment = ChatAttachment(
        name: 'script.dart',
        type: AttachmentType.document,
        bytes: fakeBytes,
        textContent: 'void main() => print("hi");',
      );
      when(() => mockMediaPicker.pickDocumentFile()).thenAnswer((_) async => attachment);

      await controller.pickDocument();

      expect(controller.attachedAttachment.value, attachment);
      expect(controller.attachedAttachment.value?.type, AttachmentType.document);
      expect(controller.attachedAttachment.value?.textContent, 'void main() => print("hi");');
      verify(() => mockMediaPicker.pickDocumentFile()).called(1);
    });

    test('sendMessage should consume attachedAttachment and clear it', () async {
      final fakeBytes = Uint8List.fromList([7, 8, 9]);
      final attachment = ChatAttachment(
        name: 'foto.png',
        type: AttachmentType.image,
        bytes: fakeBytes,
      );
      controller.isModelLoaded.value = true;
      controller.setAttachment(attachment);

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
          attachment: any(named: 'attachment'),
          modelName: any(named: 'modelName'),
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable(const [
          ChatStreamChunk(contentDelta: 'A foto mostra um gato.', isDone: true),
        ]),
      );

      await controller.sendMessage('O que tem aqui?');

      expect(controller.attachedAttachment.value, isNull);
      expect(controller.chatHistory.value.length, 2);
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
      expect(controller.chatHistory.value.first.text, 'O que tem aqui?');
      expect(controller.chatHistory.value.last.text, 'A foto mostra um gato.');

      verify(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: fakeBytes,
          attachment: attachment,
          modelName: 'llama3.2:3b',
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).called(1);
    });

    test('sendMessage with empty text but document attachment defaults prompt', () async {
      final fakeBytes = Uint8List.fromList([11, 22, 33]);
      final docAttachment = ChatAttachment(
        name: 'exemplo.py',
        type: AttachmentType.document,
        bytes: fakeBytes,
        textContent: 'print(42)',
      );
      controller.isModelLoaded.value = true;
      controller.setAttachment(docAttachment);

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
          attachment: any(named: 'attachment'),
          modelName: any(named: 'modelName'),
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable(const [
          ChatStreamChunk(contentDelta: 'Código analisado', isDone: true),
        ]),
      );

      await controller.sendMessage('');

      expect(controller.attachedAttachment.value, isNull);
      expect(controller.chatHistory.value.first.text, 'Analise este documento/código');
      expect(controller.chatHistory.value.first.attachment, docAttachment);
    });
  });

  group('ChatController - Ollama Models', () {
    test('refreshModels populates availableModels when healthy', () async {
      await controller.refreshModels();

      expect(controller.isOllamaConnected.value, isTrue);
      expect(controller.availableModels.value.length, 1);
      expect(controller.availableModels.value.first.name, 'llama3.2:3b');
    });

    test('loadModel requests preload on repository', () async {
      when(
        () => mockRepository.loadModel(
          modelName: any(named: 'modelName'),
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((_) async {});

      await controller.loadModel('llama3.2:3b');

      expect(controller.isModelLoaded.value, isTrue);
      expect(controller.selectedModelName.value, 'llama3.2:3b');
      verify(
        () => mockRepository.loadModel(
          modelName: 'llama3.2:3b',
          onProgress: any(named: 'onProgress'),
        ),
      ).called(1);
    });
  });

  group('ChatController - Ollama Network Host Settings', () {
    test('testOllamaConnection should call checkOllamaHealth with normalized URL', () async {
      when(() => mockRepository.checkOllamaHealth('http://192.168.1.55:11434'))
          .thenAnswer((_) async => true);

      final result = await controller.testOllamaConnection('192.168.1.55');

      expect(result, isTrue);
      verify(() => mockRepository.checkOllamaHealth('http://192.168.1.55:11434')).called(1);
    });

    test('updateOllamaHost should update repository, save URL, and refresh models', () async {
      when(() => mockRepository.saveOllamaBaseUrl(any())).thenAnswer((_) async {});
      when(() => mockRepository.checkOllamaHealth()).thenAnswer((_) async => true);

      final success = await controller.updateOllamaHost('192.168.1.80:11434');

      expect(success, isTrue);
      expect(controller.ollamaBaseUrl.value, 'http://192.168.1.80:11434');
      verify(() => mockRepository.updateOllamaBaseUrl('http://192.168.1.80:11434')).called(1);
      verify(() => mockRepository.saveOllamaBaseUrl('http://192.168.1.80:11434')).called(1);
    });

    test('loadOllamaServerUrl should load URL from repository', () async {
      when(() => mockRepository.loadOllamaBaseUrl())
          .thenAnswer((_) async => 'http://10.0.0.5:11434');

      await controller.loadOllamaServerUrl();

      expect(controller.ollamaBaseUrl.value, 'http://10.0.0.5:11434');
    });
  });

  group('ChatController - Settings Integration', () {
    test('buildSettingsArgs captures current controller state', () {
      controller.ollamaBaseUrl.value = 'http://192.168.1.244:11434';
      controller.selectedModelName.value = 'mimo-v2.6:9b';
      controller.responseStyle.value = ResponseStyle.detailed;
      controller.isOllamaConnected.value = true;
      controller.isModelLoaded.value = true;

      final args = controller.buildSettingsArgs();

      expect(args.ollamaBaseUrl, 'http://192.168.1.244:11434');
      expect(args.selectedModelName, 'mimo-v2.6:9b');
      expect(args.responseStyle, ResponseStyle.detailed);
      expect(args.isOllamaConnected, isTrue);
      expect(args.isModelLoaded, isTrue);
    });

    test('applySettings updates loaded model, style, metrics, and host without duplicate preload', () async {
      await controller.createNewConversation();
      controller.selectedModelName.value = 'llama3.2:3b';
      controller.isModelLoaded.value = false;

      const loadedModel = OllamaModel(
        name: 'mimo-v2.6:9b',
        model: 'mimo-v2.6:9b',
        size: 5000000000,
        isLoaded: true,
        sizeVram: 4080218931, // ~3.8 GB
        contextLength: 8192,
      );

      await controller.applySettings(
        const SettingsResult(
          selectedModelName: 'mimo-v2.6:9b',
          responseStyle: ResponseStyle.brief,
          ollamaBaseUrl: 'http://192.168.1.244:11434',
          availableModels: [loadedModel],
          isOllamaConnected: true,
          isModelLoaded: true,
        ),
      );

      expect(controller.ollamaBaseUrl.value, 'http://192.168.1.244:11434');
      expect(controller.selectedModelName.value, 'mimo-v2.6:9b');
      expect(controller.responseStyle.value, ResponseStyle.brief);
      expect(controller.isModelLoaded.value, isTrue);
      expect(controller.currentMetrics.value.vramFormatted, '3.8 GB');
      expect(controller.currentMetrics.value.contextLimit, 8192);
      expect(controller.currentConversation?.modelName, 'mimo-v2.6:9b');
      expect(controller.currentConversation?.responseStyle, ResponseStyle.brief);

      verifyNever(
        () => mockRepository.loadModel(
          modelName: any(named: 'modelName'),
          onProgress: any(named: 'onProgress'),
        ),
      );
    });

    test('applySettings preloads newly selected unloaded model', () async {
      await controller.createNewConversation();
      when(
        () => mockRepository.loadModel(
          modelName: 'deepseek-r1:8b',
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((_) async {});

      const unloadedModel = OllamaModel(
        name: 'deepseek-r1:8b',
        model: 'deepseek-r1:8b',
        size: 5000000000,
        isLoaded: false,
      );

      await controller.applySettings(
        const SettingsResult(
          selectedModelName: 'deepseek-r1:8b',
          responseStyle: ResponseStyle.simple,
          ollamaBaseUrl: 'http://localhost:11434',
          availableModels: [unloadedModel],
          isOllamaConnected: true,
          isModelLoaded: false,
        ),
      );

      verify(
        () => mockRepository.loadModel(
          modelName: 'deepseek-r1:8b',
          onProgress: any(named: 'onProgress'),
        ),
      ).called(1);
    });
  });

  group('ChatController - Thinking Stream', () {
    test('sendMessage updates thinkingContent, isThinking, and final content live', () async {
      final streamController = StreamController<ChatStreamChunk>();
      controller.isModelLoaded.value = true;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
          attachment: any(named: 'attachment'),
          modelName: any(named: 'modelName'),
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).thenAnswer((_) => streamController.stream);

      final sendFuture = controller.sendMessage('Qual a capital?');
      await Future<void>.delayed(Duration.zero);

      streamController.add(
        const ChatStreamChunk(thinkingDelta: 'Pensando na geografia', isThinking: true),
      );
      await Future<void>.delayed(Duration.zero);

      expect(controller.chatHistory.value.last.thinkingContent, 'Pensando na geografia');
      expect(controller.chatHistory.value.last.isThinking, isTrue);
      expect(controller.chatHistory.value.last.content, isEmpty);

      streamController.add(
        const ChatStreamChunk(contentDelta: 'Brasília.', isThinking: false, isDone: true),
      );
      await streamController.close();
      await sendFuture;

      expect(controller.chatHistory.value.last.thinkingContent, 'Pensando na geografia');
      expect(controller.chatHistory.value.last.content, 'Brasília.');
      expect(controller.chatHistory.value.last.isThinking, isFalse);
      expect(controller.isGenerating.value, isFalse);
    });

    test('stopGeneration resets active isThinking flag on assistant message', () async {
      final streamController = StreamController<ChatStreamChunk>();
      controller.isModelLoaded.value = true;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
          attachment: any(named: 'attachment'),
          modelName: any(named: 'modelName'),
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
          onMetrics: any(named: 'onMetrics'),
        ),
      ).thenAnswer((_) => streamController.stream);

      final sendFuture = controller.sendMessage('Explique');
      await Future<void>.delayed(Duration.zero);

      streamController.add(
        const ChatStreamChunk(thinkingDelta: 'Etapa 1', isThinking: true),
      );
      await Future<void>.delayed(Duration.zero);
      expect(controller.chatHistory.value.last.isThinking, isTrue);

      controller.stopGeneration();
      expect(controller.chatHistory.value.last.isThinking, isFalse);
      expect(controller.isGenerating.value, isFalse);

      await streamController.close();
      await sendFuture;
    });
  });
}
