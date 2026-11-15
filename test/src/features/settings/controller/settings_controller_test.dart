import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';

class MockSettingsRepository extends Mock implements SettingsRepository {}

void main() {
  late MockSettingsRepository mockRepository;
  late SettingsController controller;

  const modelA = OllamaModel(
    name: 'mimo-v2.6:9b',
    model: 'mimo-v2.6:9b',
    size: 5000000000,
    parameterSize: '9B',
    isLoaded: false,
  );
  const modelB = OllamaModel(
    name: 'llama3.2:3b',
    model: 'llama3.2:3b',
    size: 2000000000,
    parameterSize: '3.2B',
    isLoaded: true,
    sizeVram: 2500000000,
  );

  setUp(() {
    mockRepository = MockSettingsRepository();
    when(() => mockRepository.ollamaBaseUrl)
        .thenReturn('http://localhost:11434');
    when(() => mockRepository.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://localhost:11434');
    when(() => mockRepository.saveOllamaBaseUrl(any()))
        .thenAnswer((_) async {});
    when(() => mockRepository.updateOllamaBaseUrl(any())).thenReturn(null);
    when(() => mockRepository.checkOllamaHealth(any()))
        .thenAnswer((_) async => true);
    when(() => mockRepository.checkOllamaHealth())
        .thenAnswer((_) async => true);
    when(() => mockRepository.getAvailableModels())
        .thenAnswer((_) async => [modelA, modelB]);

    controller = SettingsController(repository: mockRepository);
  });

  group('SettingsController', () {
    test('init with SettingsArgs populates draft signals without extra fetch', () async {
      await controller.init(
        const SettingsArgs(
          selectedModelName: 'mimo-v2.6:9b',
          responseStyle: ResponseStyle.detailed,
          ollamaBaseUrl: 'http://192.168.1.244:11434',
          availableModels: [modelA, modelB],
          isOllamaConnected: true,
          isModelLoaded: false,
        ),
      );

      expect(controller.ollamaBaseUrl.value, 'http://192.168.1.244:11434');
      expect(controller.selectedModelName.value, 'mimo-v2.6:9b');
      expect(controller.responseStyle.value, ResponseStyle.detailed);
      expect(controller.availableModels.value.length, 2);
      expect(controller.isModelLoaded.value, isFalse);
      verifyNever(() => mockRepository.getAvailableModels());
    });

    test('init without args loads persisted url and available models', () async {
      when(() => mockRepository.loadOllamaBaseUrl())
          .thenAnswer((_) async => 'http://192.168.1.244:11434');

      await controller.init();

      expect(controller.ollamaBaseUrl.value, 'http://192.168.1.244:11434');
      expect(controller.availableModels.value.length, 2);
      expect(controller.selectedModelName.value, 'llama3.2:3b');
      expect(controller.isModelLoaded.value, isTrue);
    });

    test('loadModelIntoMemory preloads model and marks it loaded', () async {
      await controller.init(
        const SettingsArgs(
          selectedModelName: 'mimo-v2.6:9b',
          ollamaBaseUrl: 'http://localhost:11434',
          availableModels: [modelA],
        ),
      );

      when(
        () => mockRepository.loadModel(
          modelName: 'mimo-v2.6:9b',
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((_) async {});
      when(() => mockRepository.getAvailableModels()).thenAnswer(
        (_) async => [modelA.copyWith(isLoaded: true, sizeVram: 4080218931)],
      );

      await controller.loadModelIntoMemory();

      expect(controller.isLoadingModel.value, isFalse);
      expect(controller.isModelLoaded.value, isTrue);
      expect(controller.availableModels.value.first.isLoaded, isTrue);
      expect(controller.loadErrorMessage.value, isNull);
    });

    test('applyServerHost updates draft url without persisting and discardChanges reverts it', () async {
      await controller.init(
        const SettingsArgs(
          selectedModelName: 'mimo-v2.6:9b',
          ollamaBaseUrl: 'http://localhost:11434',
          availableModels: [modelA],
        ),
      );

      final connected = await controller.applyServerHost('192.168.1.244');

      expect(connected, isTrue);
      expect(controller.ollamaBaseUrl.value, 'http://192.168.1.244:11434');
      verify(() => mockRepository.updateOllamaBaseUrl('http://192.168.1.244:11434'))
          .called(1);
      verifyNever(() => mockRepository.saveOllamaBaseUrl(any()));

      controller.discardChanges();
      verify(() => mockRepository.updateOllamaBaseUrl('http://localhost:11434'))
          .called(1);
    });

    test('saveSettings applies pending host input, persists url, and returns SettingsResult', () async {
      await controller.init(
        const SettingsArgs(
          selectedModelName: 'mimo-v2.6:9b',
          responseStyle: ResponseStyle.simple,
          ollamaBaseUrl: 'http://localhost:11434',
          availableModels: [modelA, modelB],
        ),
      );

      controller.toggleServerEditor();
      controller.selectModel('llama3.2:3b');
      controller.selectResponseStyle(ResponseStyle.brief);

      final result = await controller.saveSettings(
        pendingHostInput: '192.168.1.244',
      );

      expect(result.ollamaBaseUrl, 'http://192.168.1.244:11434');
      expect(result.selectedModelName, 'llama3.2:3b');
      expect(result.responseStyle, ResponseStyle.brief);
      expect(result.isModelLoaded, isTrue);
      verify(() => mockRepository.saveOllamaBaseUrl('http://192.168.1.244:11434'))
          .called(1);
    });
  });
}
