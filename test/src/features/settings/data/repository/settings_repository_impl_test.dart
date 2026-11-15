import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/settings/data/datasource/settings_datasource.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository_impl.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

class MockOllamaRemoteDatasource extends Mock
    implements OllamaRemoteDatasource {}

class MockSettingsDatasource extends Mock implements SettingsDatasource {}

void main() {
  late MockOllamaRemoteDatasource mockOllamaDatasource;
  late MockSettingsDatasource mockSettingsDatasource;
  late SettingsRepositoryImpl repository;

  setUp(() {
    mockOllamaDatasource = MockOllamaRemoteDatasource();
    mockSettingsDatasource = MockSettingsDatasource();
    when(() => mockOllamaDatasource.baseUrl)
        .thenReturn('http://localhost:11434');
    repository = SettingsRepositoryImpl(
      ollamaDatasource: mockOllamaDatasource,
      settingsDatasource: mockSettingsDatasource,
    );
  });

  group('SettingsRepositoryImpl', () {
    test('loadOllamaBaseUrl reads from settings datasource and updates remote datasource', () async {
      when(() => mockSettingsDatasource.getOllamaBaseUrl())
          .thenAnswer((_) async => 'http://192.168.1.244:11434');
      when(() => mockOllamaDatasource.updateBaseUrl(any())).thenReturn(null);

      final url = await repository.loadOllamaBaseUrl();

      expect(url, 'http://192.168.1.244:11434');
      verify(() => mockOllamaDatasource.updateBaseUrl('http://192.168.1.244:11434'))
          .called(1);
    });

    test('saveOllamaBaseUrl updates remote datasource and persists url', () async {
      when(() => mockOllamaDatasource.updateBaseUrl(any())).thenReturn(null);
      when(() => mockSettingsDatasource.saveOllamaBaseUrl(any()))
          .thenAnswer((_) async {});

      await repository.saveOllamaBaseUrl('http://192.168.1.50:11434');

      verify(() => mockOllamaDatasource.updateBaseUrl('http://192.168.1.50:11434'))
          .called(1);
      verify(() => mockSettingsDatasource.saveOllamaBaseUrl('http://192.168.1.50:11434'))
          .called(1);
    });

    test('checkOllamaHealth and getAvailableModels delegate to remote datasource', () async {
      const model = OllamaModel(
        name: 'mimo-v2.6:9b',
        model: 'mimo-v2.6:9b',
        size: 5000000000,
      );
      when(() => mockOllamaDatasource.checkHealth('http://10.0.0.2:11434'))
          .thenAnswer((_) async => true);
      when(() => mockOllamaDatasource.getAvailableModels())
          .thenAnswer((_) async => [model]);

      expect(
        await repository.checkOllamaHealth('http://10.0.0.2:11434'),
        isTrue,
      );
      expect(await repository.getAvailableModels(), [model]);
    });

    test('loadModel preloads model when server is healthy', () async {
      when(() => mockOllamaDatasource.checkHealth())
          .thenAnswer((_) async => true);
      when(() => mockOllamaDatasource.preloadModel('mimo-v2.6:9b'))
          .thenAnswer((_) async {});

      final progresses = <double>[];
      await repository.loadModel(
        modelName: 'mimo-v2.6:9b',
        onProgress: (p, _) => progresses.add(p),
      );

      expect(progresses, contains(1.0));
      verify(() => mockOllamaDatasource.preloadModel('mimo-v2.6:9b')).called(1);
    });

    test('loadModel throws StateError when server is unhealthy', () async {
      when(() => mockOllamaDatasource.checkHealth())
          .thenAnswer((_) async => false);

      expect(
        () => repository.loadModel(
          modelName: 'mimo-v2.6:9b',
          onProgress: (_, _) {},
        ),
        throwsA(isA<StateError>()),
      );
      verifyNever(() => mockOllamaDatasource.preloadModel(any()));
    });

    test('context and memory settings delegate cleanly to datasources', () async {
      when(() => mockSettingsDatasource.getContextMode())
          .thenAnswer((_) async => ContextMode.manual);
      when(() => mockSettingsDatasource.saveContextMode(ContextMode.automatic))
          .thenAnswer((_) async {});
      when(() => mockSettingsDatasource.getManualContextLength())
          .thenAnswer((_) async => 16384);
      when(() => mockSettingsDatasource.saveManualContextLength(32768))
          .thenAnswer((_) async {});
      when(() => mockSettingsDatasource.getServerMemoryLimitGb())
          .thenAnswer((_) async => 32.0);
      when(() => mockSettingsDatasource.saveServerMemoryLimitGb(64.0))
          .thenAnswer((_) async {});
      when(() => mockOllamaDatasource.getModelMaxContextLength('qwen2.5:7b'))
          .thenAnswer((_) async => 32768);

      expect(await repository.loadContextMode(), ContextMode.manual);
      await repository.saveContextMode(ContextMode.automatic);
      expect(await repository.loadManualContextLength(), 16384);
      await repository.saveManualContextLength(32768);
      expect(await repository.loadServerMemoryLimitGb(), 32.0);
      await repository.saveServerMemoryLimitGb(64.0);
      expect(await repository.getModelMaxContextLength('qwen2.5:7b'), 32768);

      verify(() => mockSettingsDatasource.saveContextMode(ContextMode.automatic))
          .called(1);
      verify(() => mockSettingsDatasource.saveManualContextLength(32768))
          .called(1);
      verify(() => mockSettingsDatasource.saveServerMemoryLimitGb(64.0))
          .called(1);
    });
  });
}
