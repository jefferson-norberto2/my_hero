import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show TargetPlatform;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/shared/services/device_resources_service.dart';
import 'package:my_hero/src/shared/services/inference_notification_service.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}
class MockInferenceNotificationService extends Mock
    implements InferenceNotificationService {}
class MockDeviceResourcesService extends Mock
    implements DeviceResourcesService {}

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
    when(() => mockRepository.getInstalledModelFileNames())
        .thenAnswer((_) async => <String>{});
    when(() => mockRepository.uninstallModel(any())).thenAnswer((_) async {});
    when(() => mockRepository.cleanupStorage()).thenAnswer((_) async {});
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
    test('platform capability helpers and model-filtered backends behave correctly', () {
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
        AppConstants.availableBackendsForPlatform(TargetPlatform.android),
        [PreferredBackend.cpu, PreferredBackend.gpu, PreferredBackend.npu],
      );
      expect(
        AppConstants.availableBackendsForPlatform(TargetPlatform.windows),
        [PreferredBackend.cpu, PreferredBackend.gpu, PreferredBackend.npu],
      );
      expect(
        AppConstants.availableBackendsForPlatform(TargetPlatform.linux),
        [PreferredBackend.cpu, PreferredBackend.gpu],
      );

      // Bundles genéricos abertos do catálogo não suportam NPU
      expect(
        androidCtrl.availableBackends,
        [PreferredBackend.cpu, PreferredBackend.gpu],
      );
      expect(
        windowsCtrl.availableBackends,
        [PreferredBackend.cpu, PreferredBackend.gpu],
      );
      expect(
        linuxCtrl.availableBackends,
        [PreferredBackend.cpu, PreferredBackend.gpu],
      );

      // TranslateGemma 4B restringe GPU e roda apenas em CPU
      final translateSpec = AppConstants.modelSpecById('translategemma-4b-it');
      expect(
        androidCtrl.availableBackendsForModel(translateSpec),
        [PreferredBackend.cpu],
      );

      expect(androidCtrl.maxTokens.value, 2048);
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
          modelId: AppConstants.model2b.id,
          is2B: true,
          preferredBackend: PreferredBackend.gpu,
          maxTokens: 2048,
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: true,
          enableThinking: true,
        ),
      ).thenAnswer((_) async {});
      when(() => mockRepository.activeBackend).thenReturn(PreferredBackend.cpu);

      controller.selectBackend(PreferredBackend.gpu);
      controller.setEnableThinking(true);

      await controller.downloadAndLoadModel(modelId: AppConstants.model2b.id);

      expect(controller.isModelLoaded.value, isTrue);
      expect(controller.selectedBackend.value, PreferredBackend.gpu);
      expect(controller.activeBackend.value, PreferredBackend.cpu);
      expect(controller.backendDisplaySummary, 'GPU → CPU');
    });

    test('updateConversationSettings changing only enableThinking should restart session without reloading model', () async {
      controller.isModelLoaded.value = true;
      when(
        () => mockRepository.startNewChatSession(
          modelId: AppConstants.model2b.id,
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: true,
          enableThinking: true,
        ),
      ).thenAnswer((_) async {});

      await controller.updateConversationSettings(
        modelId: controller.selectedModelId.value,
        preferredBackend: controller.selectedBackend.value,
        maxTokens: controller.maxTokens.value,
        enableThinking: true,
        enableVision: controller.enableVision.value,
        newResponseStyle: controller.responseStyle.value,
      );

      expect(controller.enableThinking.value, isTrue);
      verify(
        () => mockRepository.startNewChatSession(
          modelId: AppConstants.model2b.id,
          contextMessages: any(named: 'contextMessages'),
          responseStyle: ResponseStyle.simple,
          supportImage: true,
          enableThinking: true,
        ),
      ).called(1);
      verifyNever(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          supportImage: any(named: 'supportImage'),
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

  group('ChatController - Typed flutter_edge_ai Error Handling', () {
    test('sendMessage should show friendly message on ContextLimitException', () async {
      controller.isModelLoaded.value = true;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer(
        (_) => Stream<ChatStreamChunk>.error(
          const ContextLimitException('Input exceeds context window'),
        ),
      );

      await controller.sendMessage('Prompt gigante');

      expect(controller.isGenerating.value, isFalse);
      expect(
        controller.chatHistory.value.last.text,
        contains('janela de contexto'),
      );
      expect(controller.chatHistory.value.last.text, contains('2K'));
    });

    test('downloadAndLoadModel should report incomplete bundle on LiteRtLmBundleException', () async {
      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenThrow(
        const LiteRtLmBundleException(
          LiteRtLmBundleProblem.incomplete,
          'File truncated',
        ),
      );

      await controller.downloadAndLoadModel(is2B: true);

      expect(controller.isLoading.value, isFalse);
      expect(
        controller.errorMessage.value,
        contains('incompleto ou corrompido'),
      );
    });

    test('downloadAndLoadModel should report validation mismatch on ModelValidationException', () async {
      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenThrow(const ModelValidationException('size mismatch'));

      await controller.downloadAndLoadModel(is2B: true);

      expect(controller.isLoading.value, isFalse);
      expect(
        controller.errorMessage.value,
        contains('verificação de integridade'),
      );
    });

    test('downloadAndLoadModel should report missing active model on NoActiveModelError', () async {
      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenThrow(NoActiveModelError('No active inference model'));

      await controller.downloadAndLoadModel(is2B: true);

      expect(controller.isLoading.value, isFalse);
      expect(
        controller.errorMessage.value,
        contains('Nenhum modelo ativo foi encontrado'),
      );
    });

    test('downloadAndLoadModel should report ALL backend attempts, keep installed files and clear stale loaded state on BackendInitException', () async {
      final gemma2b = AppConstants.model2b;
      controller.isModelLoaded.value = true;
      controller.loadedModelId.value = gemma2b.id;
      controller.activeBackend.value = PreferredBackend.cpu;

      when(() => mockRepository.getInstalledModelFileNames())
          .thenAnswer((_) async => {gemma2b.fileName});
      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenThrow(
        BackendInitException(
          attempts: [
            BackendInitAttemptFailure(
              backend: PreferredBackend.gpu,
              error: 'Vulkan device unavailable',
              stackTrace: StackTrace.empty,
            ),
            BackendInitAttemptFailure(
              backend: PreferredBackend.cpu,
              error: 'Cannot allocate memory',
              stackTrace: StackTrace.empty,
            ),
          ],
        ),
      );

      await controller.downloadAndLoadModel(is2B: true);
      await Future<void>.delayed(Duration.zero);

      expect(controller.isLoading.value, isFalse);
      expect(controller.isModelLoaded.value, isFalse);
      expect(controller.loadedModelId.value, isNull);
      expect(controller.activeBackend.value, isNull);
      expect(controller.isModelInstalled(gemma2b), isTrue);
      expect(controller.errorMessage.value, contains('GPU'));
      expect(controller.errorMessage.value, contains('Vulkan device unavailable'));
      expect(controller.errorMessage.value, contains('CPU'));
      expect(controller.errorMessage.value, contains('Cannot allocate memory'));
    });

    test('updateConversationSettings should NOT overwrite saved Conversation when loadModel fails', () async {
      final now = DateTime.now();
      final originalConv = Conversation(
        id: 'conv-1',
        title: 'Conversa Estável',
        messages: [],
        modelId: AppConstants.model2b.id,
        preferredBackend: PreferredBackend.cpu,
        maxTokens: 2048,
        enableThinking: false,
        enableVision: true,
        responseStyle: ResponseStyle.simple,
        createdAt: now,
        updatedAt: now,
      );
      controller.conversations.value = [originalConv];
      controller.currentConversationId.value = originalConv.id;
      controller.isModelLoaded.value = true;
      controller.loadedModelId.value = AppConstants.model2b.id;

      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenThrow(
        BackendInitException(
          attempts: [
            BackendInitAttemptFailure(
              backend: PreferredBackend.gpu,
              error: 'Out of memory',
              stackTrace: StackTrace.empty,
            ),
          ],
        ),
      );

      await controller.updateConversationSettings(
        modelId: AppConstants.model4b.id,
        preferredBackend: PreferredBackend.gpu,
        maxTokens: 16384,
        enableThinking: true,
        enableVision: false,
        newResponseStyle: ResponseStyle.detailed,
      );

      expect(originalConv.modelId, AppConstants.model2b.id);
      expect(originalConv.maxTokens, 2048);
      expect(originalConv.preferredBackend, PreferredBackend.cpu);
      expect(originalConv.enableVision, isTrue);
      verifyNever(() => mockRepository.saveConversations(any()));
    });

    test('downloadAndLoadModel should treat DownloadException(CanceledError) from Android notification cancel as user cancel', () async {
      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenThrow(const DownloadException(DownloadError.canceled()));

      await controller.downloadAndLoadModel(is2B: true);

      expect(controller.isLoading.value, isFalse);
      expect(controller.errorMessage.value, 'Download cancelado pelo usuário.');
    });

    test('downloadAndLoadModel should surface localized network error on DownloadException(NetworkError)', () async {
      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenThrow(const DownloadException(DownloadError.network('timeout')));

      await controller.downloadAndLoadModel(is2B: true);

      expect(controller.isLoading.value, isFalse);
      expect(controller.errorMessage.value, contains('Erro de conexão'));
      expect(controller.errorMessage.value, contains('timeout'));
    });
  });

  group('ChatController - Device RAM & Storage Guardrails', () {
    late MockDeviceResourcesService mockDeviceResources;

    setUp(() {
      mockDeviceResources = MockDeviceResourcesService();
    });

    test('preflight blocks download when free storage is insufficient', () async {
      when(() => mockDeviceResources.getDeviceResources()).thenAnswer(
        (_) async => const DeviceResourcesInfo(
          totalRamBytes: 8 * 1024 * 1024 * 1024,
          availableRamBytes: 4 * 1024 * 1024 * 1024,
          lowMemory: false,
          memoryThresholdBytes: 256 * 1024 * 1024,
          freeStorageBytes: 500 * 1024 * 1024,
        ),
      );

      final ctrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        deviceResourcesService: mockDeviceResources,
        platform: TargetPlatform.android,
      );

      await ctrl.downloadAndLoadModel(modelId: AppConstants.model2b.id);

      expect(ctrl.isLoading.value, isFalse);
      expect(ctrl.errorMessage.value, contains('Espaço insuficiente'));
      verifyNever(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          supportImage: any(named: 'supportImage'),
        ),
      );
    });

    test('low-RAM Android device applies safe initial defaults (1024 tokens, vision off) and assesses high risk for 4B model', () async {
      when(() => mockDeviceResources.getDeviceResources()).thenAnswer(
        (_) async => const DeviceResourcesInfo(
          totalRamBytes: 4 * 1024 * 1024 * 1024,
          availableRamBytes: 1500 * 1024 * 1024,
          lowMemory: false,
          memoryThresholdBytes: 256 * 1024 * 1024,
          freeStorageBytes: 20 * 1024 * 1024 * 1024,
        ),
      );

      final ctrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        deviceResourcesService: mockDeviceResources,
        platform: TargetPlatform.android,
      );

      await ctrl.init();

      expect(ctrl.maxTokens.value, 1024);
      expect(ctrl.enableVision.value, isFalse);
      expect(
        ctrl.memoryAssessmentFor(
          spec: AppConstants.model4b,
          maxTokens: 4096,
          enableVision: true,
        ).risk,
        ModelMemoryRisk.high,
      );
    });
  });

  group('ChatController - Background Inference & Loading Notification', () {
    late MockInferenceNotificationService mockNotificationService;
    late ChatController notifiedController;
    Future<void> Function()? capturedStopCallback;

    setUp(() {
      mockNotificationService = MockInferenceNotificationService();
      capturedStopCallback = null;

      when(() => mockNotificationService.setOnStopRequested(any()))
          .thenAnswer((invocation) {
        capturedStopCallback =
            invocation.positionalArguments.first as Future<void> Function()?;
      });
      when(
        () => mockNotificationService.start(
          title: any(named: 'title'),
          body: any(named: 'body'),
        ),
      ).thenAnswer((_) async {});
      when(
        () => mockNotificationService.update(
          title: any(named: 'title'),
          body: any(named: 'body'),
        ),
      ).thenAnswer((_) async {});
      when(
        () => mockNotificationService.stop(
          notifyIfBackgrounded: any(named: 'notifyIfBackgrounded'),
          completedTitle: any(named: 'completedTitle'),
          completedBody: any(named: 'completedBody'),
        ),
      ).thenAnswer((_) async {});

      notifiedController = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        inferenceNotificationService: mockNotificationService,
        platform: TargetPlatform.android,
      );
    });

    test('starts foreground service during native model load and always stops it in finally even on error', () async {
      when(
        () => mockRepository.loadModel(
          modelId: any(named: 'modelId'),
          is2B: any(named: 'is2B'),
          preferredBackend: any(named: 'preferredBackend'),
          maxTokens: any(named: 'maxTokens'),
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          supportImage: any(named: 'supportImage'),
          enableThinking: any(named: 'enableThinking'),
        ),
      ).thenAnswer((invocation) async {
        final onProgress = invocation.namedArguments[#onProgress]
            as void Function(double, String);
        onProgress(1.0, 'Carregando Gemma 4 E2B (CPU, 2K tokens)...');
        throw BackendInitException(
          attempts: [
            BackendInitAttemptFailure(
              backend: PreferredBackend.cpu,
              error: 'OOM',
              stackTrace: StackTrace.empty,
            ),
          ],
        );
      });

      await notifiedController.downloadAndLoadModel(modelId: AppConstants.model2b.id);

      verify(
        () => mockNotificationService.start(
          title: 'My Hero • Gemma 4 E2B',
          body: 'Carregando modelo na memória...',
        ),
      ).called(1);
      verify(
        () => mockNotificationService.stop(
          notifyIfBackgrounded: false,
          completedTitle: null,
          completedBody: null,
        ),
      ).called(1);
    });

    test('should start thinking notification, transition to answering once, and notify completion', () async {
      notifiedController.isModelLoaded.value = true;
      notifiedController.setEnableThinking(true);

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer(
        (_) => Stream.fromIterable([
          const ChatThinkingChunk('Analisando...'),
          const ChatThinkingChunk(' quase lá.'),
          const ChatTextChunk('Resposta '),
          const ChatTextChunk('final completa.'),
        ]),
      );

      await notifiedController.sendMessage('Explique gravidade');

      verify(
        () => mockNotificationService.start(
          title: 'My Hero • Gemma 4 E2B',
          body: 'Pensando na resposta...',
        ),
      ).called(1);
      verify(
        () => mockNotificationService.update(
          title: 'My Hero • Gemma 4 E2B',
          body: 'Gerando resposta...',
        ),
      ).called(1);
      verify(
        () => mockNotificationService.stop(
          notifyIfBackgrounded: true,
          completedTitle: 'Resposta pronta (Gemma 4 E2B)',
          completedBody: 'Resposta final completa.',
        ),
      ).called(1);
    });

    test('should stop generation without completion notification when stopped from notification action', () async {
      notifiedController.isModelLoaded.value = true;
      final streamController = StreamController<ChatStreamChunk>();

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer((_) => streamController.stream);
      when(() => mockRepository.stopGeneration()).thenAnswer((_) async {
        await streamController.close();
      });

      final sendFuture = notifiedController.sendMessage('Conte uma história longa');
      await Future<void>.delayed(Duration.zero);

      expect(capturedStopCallback, isNotNull);
      await capturedStopCallback!.call();
      await sendFuture;

      expect(notifiedController.isGenerating.value, isFalse);
      verify(() => mockRepository.stopGeneration()).called(1);
      verifyNever(
        () => mockNotificationService.stop(
          notifyIfBackgrounded: true,
          completedTitle: any(named: 'completedTitle'),
          completedBody: any(named: 'completedBody'),
        ),
      );
    });

    test('should stop notification without completion alert on inference error', () async {
      notifiedController.isModelLoaded.value = true;
      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer(
        (_) => Stream<ChatStreamChunk>.error(
          const ContextLimitException('Context full'),
        ),
      );

      await notifiedController.sendMessage('Prompt muito grande');

      verify(
        () => mockNotificationService.stop(
          notifyIfBackgrounded: false,
          completedTitle: null,
          completedBody: null,
        ),
      ).called(1);
    });
  });

  group('ChatController - Open Model Catalog & Storage Management', () {
    test('availableModels should list only open (non-gated) native models with Gemma 4 featured first', () {
      final models = controller.availableModels;

      expect(models.length, greaterThanOrEqualTo(10));
      expect(models.first.id, AppConstants.model2b.id);
      expect(models[1].id, AppConstants.model4b.id);
      for (final spec in models) {
        expect(spec.catalogVariant.file?.gated, isFalse);
        expect(spec.catalogVariant.npuTarget, isNull);
      }
    });

    test('selectModel should clamp maxTokens, disable thinking, normalize backend and clear image when unsupported', () {
      controller.selectBackend(PreferredBackend.gpu);
      controller.setMaxTokens(32768);
      controller.setEnableThinking(true);
      controller.setAttachedImage(Uint8List.fromList([1, 2, 3]));

      final qwen25 = AppConstants.modelSpecById('qwen2.5-1.5b-instruct');
      controller.selectModel(qwen25.id);

      expect(controller.selectedModelId.value, qwen25.id);
      expect(controller.maxTokens.value, 4096);
      expect(controller.enableThinking.value, isFalse);
      expect(controller.attachedImage.value, isNull);
      expect(controller.supportsImage, isFalse);
      expect(controller.supportsThinking, isFalse);

      // TranslateGemma não suporta GPU: backend cai para CPU
      final translateSpec = AppConstants.modelSpecById('translategemma-4b-it');
      controller.selectModel(translateSpec.id);
      expect(controller.selectedBackend.value, PreferredBackend.cpu);
    });

    test('refreshInstalledModels and uninstallModel should track downloaded models and unload active one', () async {
      final gemma2b = AppConstants.model2b;
      final qwen3 = AppConstants.modelSpecById('qwen3-0.6b');

      var installed = {gemma2b.fileName, qwen3.fileName};
      when(() => mockRepository.getInstalledModelFileNames())
          .thenAnswer((_) async => Set<String>.from(installed));
      when(() => mockRepository.uninstallModel(gemma2b.fileName))
          .thenAnswer((_) async {
        installed = {qwen3.fileName};
      });

      await controller.refreshInstalledModels();
      expect(controller.isModelInstalled(gemma2b), isTrue);
      expect(controller.isModelInstalled(qwen3), isTrue);
      expect(controller.installedModelsCount, 2);

      controller.isModelLoaded.value = true;
      controller.loadedModelId.value = gemma2b.id;

      await controller.uninstallModel(gemma2b);

      verify(() => mockRepository.uninstallModel(gemma2b.fileName)).called(1);
      expect(controller.isModelInstalled(gemma2b), isFalse);
      expect(controller.isModelInstalled(qwen3), isTrue);
      expect(controller.isModelLoaded.value, isFalse);
      expect(controller.loadedModelId.value, isNull);
    });
  });
}
