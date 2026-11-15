import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/data/datasource/settings_datasource.dart';

class MockPathProviderPlatform extends Mock
    with MockPlatformInterfaceMixin
    implements PathProviderPlatform {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late MockPathProviderPlatform mockPathProviderPlatform;
  late SettingsDatasourceImpl datasource;
  late Directory tempDir;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('settings_test_');

    mockPathProviderPlatform = MockPathProviderPlatform();
    PathProviderPlatform.instance = mockPathProviderPlatform;

    when(() => mockPathProviderPlatform.getApplicationDocumentsPath())
        .thenAnswer((_) async => tempDir.path);

    datasource = SettingsDatasourceImpl();
  });

  tearDown(() async {
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  group('SettingsDatasourceImpl', () {
    test('getOllamaBaseUrl should return default localhost when settings file does not exist', () async {
      final result = await datasource.getOllamaBaseUrl();
      expect(result, equals(AppConstants.ollamaBaseUrl));
    });

    test('saveOllamaBaseUrl and getOllamaBaseUrl should persist custom IP', () async {
      const customUrl = 'http://192.168.1.100:11434';
      await datasource.saveOllamaBaseUrl(customUrl);

      final result = await datasource.getOllamaBaseUrl();
      expect(result, equals(customUrl));
    });

    test('getOllamaBaseUrl should return default if settings file is empty or corrupted', () async {
      final file = File('${tempDir.path}/app_settings.json');
      await file.writeAsString('   ');

      final result = await datasource.getOllamaBaseUrl();
      expect(result, equals(AppConstants.ollamaBaseUrl));
    });
  });
}
