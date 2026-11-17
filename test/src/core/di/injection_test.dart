import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/di/injection.dart';
import 'package:my_hero/src/core/theme/theme_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockPathProviderPlatform extends Mock
    with MockPlatformInterfaceMixin
    implements PathProviderPlatform {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late Directory supportDir;
  late Directory documentsDir;
  late Map<String, String> secureVault;
  late MockPathProviderPlatform mockPathProvider;

  setUp(() async {
    await getIt.reset();
    supportDir = await Directory.systemTemp.createTemp('di_support_');
    documentsDir = await Directory.systemTemp.createTemp('di_docs_');

    secureVault = <String, String>{};
    FlutterSecureStorage.setMockInitialValues(secureVault);

    mockPathProvider = MockPathProviderPlatform();
    PathProviderPlatform.instance = mockPathProvider;

    when(() => mockPathProvider.getApplicationSupportPath())
        .thenAnswer((_) async => supportDir.path);
    when(() => mockPathProvider.getApplicationDocumentsPath())
        .thenAnswer((_) async => documentsDir.path);
  });

  tearDown(() async {
    await getIt.reset();
    if (await supportDir.exists()) {
      await supportDir.delete(recursive: true);
    }
    if (await documentsDir.exists()) {
      await documentsDir.delete(recursive: true);
    }
  });

  test('setupDependencies initializes encrypted storage and migrates legacy JSON', () async {
    final now = DateTime(2026, 2, 1, 12, 0);
    final legacyConv = Conversation(
      id: 'di-legacy-1',
      title: 'Migrated on Boot',
      messages: [ChatMessage(text: 'Sensitive history', isUser: true)],
      modelName: 'llama3.2:3b',
      createdAt: now,
      updatedAt: now,
    );

    final legacyFile = File(
      '${documentsDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
    );
    await legacyFile.writeAsString(jsonEncode([legacyConv.toJson()]));

    await setupDependencies();

    expect(await legacyFile.exists(), isFalse);
    final encryptedFile = File(
      '${supportDir.path}/${ChatHistoryDatasourceImpl.encryptedHistoryFileName}',
    );
    expect(await encryptedFile.exists(), isTrue);
    expect(secureVault.keys, isNotEmpty);

    final loaded = await getIt<ChatRepository>().getConversations();
    expect(loaded.length, 1);
    expect(loaded.first.id, 'di-legacy-1');
    expect(loaded.first.title, 'Migrated on Boot');
    expect(loaded.first.messages.first.text, 'Sensitive history');
  });

  test('setupDependencies registers ThemeController and preloads persisted dark theme before render', () async {
    final settingsFile = File('${documentsDir.path}/app_settings.json');
    await settingsFile.writeAsString(jsonEncode({'theme_mode': 'dark'}));

    await setupDependencies();

    expect(getIt.isRegistered<ThemeController>(), isTrue);
    expect(getIt<ThemeController>().themeMode.value, ThemeMode.dark);
    expect(getIt<ThemeController>().isDarkMode, isTrue);
  });
}
