import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
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
  late Directory tempDir;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('history_test_');

    mockPathProviderPlatform = MockPathProviderPlatform();
    PathProviderPlatform.instance = mockPathProviderPlatform;

    when(() => mockPathProviderPlatform.getApplicationDocumentsPath())
        .thenAnswer((_) async => tempDir.path);

    datasource = ChatHistoryDatasourceImpl();
  });

  tearDown(() async {
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  group('ChatHistoryDatasourceImpl', () {
    test('loadConversations should return empty list when file does not exist', () async {
      final result = await datasource.loadConversations();
      expect(result, isEmpty);
    });

    test('saveConversations and loadConversations should persist data to disk', () async {
      final now = DateTime.now();
      final convs = [
        Conversation(
          id: '1',
          title: 'Primeira Conversa',
          messages: [
            ChatMessage(text: 'Oi', isUser: true),
            ChatMessage(text: 'Olá humano', isUser: false),
          ],
          is2B: true,
          useCpu: true,
          createdAt: now,
          updatedAt: now,
        ),
      ];

      await datasource.saveConversations(convs);

      final loaded = await datasource.loadConversations();

      expect(loaded.length, 1);
      expect(loaded.first.id, '1');
      expect(loaded.first.title, 'Primeira Conversa');
      expect(loaded.first.messages.length, 2);
      expect(loaded.first.messages.first.text, 'Oi');
      expect(loaded.first.messages.last.text, 'Olá humano');
    });
  });
}
