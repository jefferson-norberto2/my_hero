import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart' show TargetPlatform;
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';

class MockPathProviderPlatform extends Mock
    with MockPlatformInterfaceMixin
    implements PathProviderPlatform {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late MockPathProviderPlatform mockPathProviderPlatform;
  late ChatHistoryDatasourceImpl datasource;
  late Directory docsDir;
  late Directory supportDir;

  setUp(() async {
    docsDir = await Directory.systemTemp.createTemp('history_docs_test_');
    supportDir = await Directory.systemTemp.createTemp('history_support_test_');

    mockPathProviderPlatform = MockPathProviderPlatform();
    PathProviderPlatform.instance = mockPathProviderPlatform;

    when(() => mockPathProviderPlatform.getApplicationDocumentsPath())
        .thenAnswer((_) async => docsDir.path);
    when(() => mockPathProviderPlatform.getApplicationSupportPath())
        .thenAnswer((_) async => supportDir.path);

    datasource = ChatHistoryDatasourceImpl(platform: TargetPlatform.android);
  });

  tearDown(() async {
    if (await docsDir.exists()) {
      await docsDir.delete(recursive: true);
    }
    if (await supportDir.exists()) {
      await supportDir.delete(recursive: true);
    }
  });

  group('ChatHistoryDatasourceImpl', () {
    test('loadConversations should return empty list when file does not exist', () async {
      final result = await datasource.loadConversations();
      expect(result, isEmpty);
    });

    test('saveConversations and loadConversations should persist data to disk on mobile', () async {
      final now = DateTime.now();
      final convs = [
        Conversation(
          id: '1',
          title: 'Primeira Conversa',
          messages: [
            ChatMessage(text: 'Oi', isUser: true),
            ChatMessage(
              text: 'Olá humano',
              isUser: false,
              thinkingText: 'Cumprimentar de volta',
            ),
          ],
          is2B: true,
          preferredBackend: PreferredBackend.gpu,
          maxTokens: 4096,
          enableThinking: true,
          createdAt: now,
          updatedAt: now,
        ),
      ];

      await datasource.saveConversations(convs);

      final loaded = await datasource.loadConversations();

      expect(loaded.length, 1);
      expect(loaded.first.id, '1');
      expect(loaded.first.title, 'Primeira Conversa');
      expect(loaded.first.preferredBackend, PreferredBackend.gpu);
      expect(loaded.first.maxTokens, 4096);
      expect(loaded.first.enableThinking, isTrue);
      expect(loaded.first.messages.length, 2);
      expect(loaded.first.messages.last.thinkingText, 'Cumprimentar de volta');
    });

    test('desktop should save to Application Support and fallback to legacy Documents file', () async {
      final desktopDs =
          ChatHistoryDatasourceImpl(platform: TargetPlatform.windows);
      final now = DateTime.now();

      // Simula histórico legado salvo em Documents
      final legacyFile = File(p.join(docsDir.path, 'chat_conversations.json'));
      final legacyConv = Conversation(
        id: 'legacy-1',
        title: 'Conversa Antiga',
        messages: [ChatMessage(text: 'Ping', isUser: true)],
        createdAt: now,
        updatedAt: now,
      );
      await legacyFile.writeAsString(jsonEncode([legacyConv.toJson()]));

      final loadedLegacy = await desktopDs.loadConversations();
      expect(loadedLegacy.length, 1);
      expect(loadedLegacy.first.id, 'legacy-1');

      // Ao salvar novamente no desktop, persiste em Application Support
      await desktopDs.saveConversations(loadedLegacy);
      final supportFile =
          File(p.join(supportDir.path, 'chat_conversations.json'));
      expect(await supportFile.exists(), isTrue);
    });
  });
}
