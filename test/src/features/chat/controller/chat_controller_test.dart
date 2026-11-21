import 'dart:typed_data';
import 'package:flutter/foundation.dart' show TargetPlatform;
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;
  late ChatController controller;

  setUpAll(() {
    registerFallbackValue(PreferredBackend.cpu);
    registerFallbackValue(ResponseStyle.simple);
  });

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();

    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenAnswer((_) async {});
    when(() => mockRepository.stopGeneration()).thenAnswer((_) async {});
    when(() => mockRepository.activeBackend).thenReturn(null);

    controller = ChatController(
      repository: mockRepository,
      mediaPickerService: mockMediaPicker,
      platform: TargetPlatform.android,
    );
  });

  group('ChatController - Platform Capabilities & Configuration', () {
    test('Android and Windows should expose NPU; Linux should expose CPU and GPU only', () {
      final androidCtrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.android,
      );
      final windowsCtrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.windows,
      );
      final linuxCtrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.linux,
      );

      expect(
        androidCtrl.availableBackends,
        [PreferredBackend.cpu, PreferredBackend.gpu, PreferredBackend.npu],
      );
      expect(
        windowsCtrl.availableBackends,
        [PreferredBackend.cpu, PreferredBackend.gpu, PreferredBackend.npu],
      );
      expect(
        linuxCtrl.availableBackends,
        [PreferredBackend.cpu, PreferredBackend.gpu],
      );
      expect(androidCtrl.maxTokens.value, 4096);
      expect(linuxCtrl.maxTokens.value, 8192);
      expect(windowsCtrl.maxTokens.value, 8192);
      expect(androidCtrl.supportsCamera, isTrue);
      expect(linuxCtrl.supportsCamera, isFalse);
      expect(windowsCtrl.supportsCamera, isFalse);
    });

    test('setMaxTokens should clamp between model min and max limits', () {
      controller.setMaxTokens(100);
      expect(controller.maxTokens.value, AppConstants.minContextTokens);

      controller.setMaxTokens(100000);
      expect(controller.maxTokens.value, AppConstants.maxContextTokens);

      controller.setMaxTokens(16384);
      expect(controller.maxTokens.value, 16384);
    });

    test('downloadAndLoadModel should report fallback activeBackend when runtime differs from requested', () async {
      when(
        () => mockRepository.loadModel(
          is2B: true,
          preferredBackend: PreferredBackend.npu,
          maxTokens: 4096,
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          enableThinking: true,
        ),
      ).thenAnswer((_) async {});
      when(() => mockRepository.activeBackend).thenReturn(PreferredBackend.gpu);

      controller.selectBackend(PreferredBackend.npu);
      controller.setEnableThinking(true);

      await controller.downloadAndLoadModel(is2B: true);

      expect(controller.isModelLoaded.value, isTrue);
      expect(controller.selectedBackend.value, PreferredBackend.npu);
      expect(controller.activeBackend.value, PreferredBackend.gpu);
      expect(controller.backendDisplaySummary, 'NPU → GPU');
    });

    test('updateConversationSettings changing only enableThinking should restart session without reloading model', () async {
      controller.isModelLoaded.value = true;
      when(
        () => mockRepository.startNewChatSession(
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          enableThinking: true,
        ),
      ).thenAnswer((_) async {});

      await controller.updateConversationSettings(
        is2B: controller.is2BModel.value,
        preferredBackend: controller.selectedBackend.value,
        maxTokens: controller.maxTokens.value,
        enableThinking: true,
        newResponseStyle: controller.responseStyle.value,
      );

      expect(controller.enableThinking.value, isTrue);
      verify(
        () => mockRepository.startNewChatSession(
          contextMessages: any(named: 'contextMessages'),
          responseStyle: ResponseStyle.simple,
          enableThinking: true,
        ),
      ).called(1);
      verifyNever(
        () => mockRepository.loadModel(
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
        ),
      );
    });
  });

  group('ChatController - Multimodal Image & Thinking Handling', () {
    test('initial attachedImage should be null', () {
      expect(controller.attachedImage.value, isNull);
    });

    test('setAttachedImage and clearAttachedImage should update state correctly', () {
      final fakeBytes = Uint8List.fromList([1, 2, 3, 4]);

      controller.setAttachedImage(fakeBytes);
      expect(controller.attachedImage.value, fakeBytes);

      controller.clearAttachedImage();
      expect(controller.attachedImage.value, isNull);
    });

    test('pickImage from gallery should store image in attachedImage', () async {
      final fakeBytes = Uint8List.fromList([10, 20, 30]);
      when(() => mockMediaPicker.pickImageFromGallery())
          .thenAnswer((_) async => fakeBytes);

      await controller.pickImage(fromCamera: false);

      expect(controller.attachedImage.value, fakeBytes);
      verify(() => mockMediaPicker.pickImageFromGallery()).called(1);
    });

    test('pickImage from camera should store image in attachedImage on mobile', () async {
      final fakeBytes = Uint8List.fromList([40, 50, 60]);
      when(() => mockMediaPicker.pickImageFromCamera())
          .thenAnswer((_) async => fakeBytes);

      await controller.pickImage(fromCamera: true);

      expect(controller.attachedImage.value, fakeBytes);
      verify(() => mockMediaPicker.pickImageFromCamera()).called(1);
    });

    test('sendMessage should separate thinkingText and answer text from stream chunks', () async {
      controller.isModelLoaded.value = true;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable([
          const ChatThinkingChunk('Vou calcular 2+2. '),
          const ChatThinkingChunk('O resultado é 4.'),
          const ChatTextChunk('A resposta é '),
          const ChatTextChunk('4.'),
        ]),
      );

      await controller.sendMessage('Quanto é 2+2?');

      expect(controller.chatHistory.value.length, 2);
      expect(
        controller.chatHistory.value.last.thinkingText,
        'Vou calcular 2+2. O resultado é 4.',
      );
      expect(controller.chatHistory.value.last.text, 'A resposta é 4.');
    });

    test('sendMessage should consume attachedImage and clear it', () async {
      final fakeBytes = Uint8List.fromList([7, 8, 9]);
      controller.isModelLoaded.value = true;
      controller.attachedImage.value = fakeBytes;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable([
          const ChatTextChunk('A foto mostra um gato.'),
        ]),
      );

      await controller.sendMessage('O que tem aqui?');

      expect(controller.attachedImage.value, isNull);
      expect(controller.chatHistory.value.length, 2);
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
      expect(controller.chatHistory.value.first.text, 'O que tem aqui?');
      expect(controller.chatHistory.value.last.text, 'A foto mostra um gato.');

      verify(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: fakeBytes,
        ),
      ).called(1);
    });

    test('sendMessage with empty text but attachedImage should default to prompt', () async {
      final fakeBytes = Uint8List.fromList([11, 22, 33]);
      controller.isModelLoaded.value = true;
      controller.attachedImage.value = fakeBytes;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable([const ChatTextChunk('Foto analisada')]),
      );

      await controller.sendMessage('');

      expect(controller.attachedImage.value, isNull);
      expect(controller.chatHistory.value.first.text, 'Analise esta imagem');
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
    });

    test('pickAndSendImage should pick image from gallery and send immediately', () async {
      final fakeBytes = Uint8List.fromList([100, 200]);
      controller.isModelLoaded.value = true;

      when(() => mockMediaPicker.pickImageFromGallery())
          .thenAnswer((_) async => fakeBytes);
      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable([const ChatTextChunk('Imagem identificada')]),
      );

      await controller.pickAndSendImage('Descreva esta foto');

      expect(controller.chatHistory.value.first.text, 'Descreva esta foto');
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
    });
  });
}
