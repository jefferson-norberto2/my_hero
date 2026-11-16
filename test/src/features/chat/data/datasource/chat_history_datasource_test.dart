import 'dart:convert';
import 'dart:io';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/services/aes_gcm_encryption_service.dart';
import 'package:my_hero/src/features/chat/data/services/encryption_key_manager.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockPathProviderPlatform extends Mock
    with MockPlatformInterfaceMixin
    implements PathProviderPlatform {}

class MockFlutterSecureStorage extends Mock implements FlutterSecureStorage {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late MockPathProviderPlatform mockPathProviderPlatform;
  late ChatHistoryDatasourceImpl datasource;
  late Directory supportDir;
  late Directory documentsDir;
  late Map<String, String> secureVault;

  ChatHistoryDatasourceImpl createDatasource() {
    final keyManager = EncryptionKeyManager();
    final encryptionService = AesGcmEncryptionService(keyManager: keyManager);
    return ChatHistoryDatasourceImpl(encryptionService: encryptionService);
  }

  Conversation buildSampleConversation({
    String id = '1',
    String title = 'First Conversation',
  }) {
    final now = DateTime(2026, 1, 15, 10, 30);
    return Conversation(
      id: id,
      title: title,
      messages: [
        ChatMessage(text: 'Hello', isUser: true),
        ChatMessage(text: 'Hi there!', isUser: false),
      ],
      modelName: 'llama3.2:3b',
      createdAt: now,
      updatedAt: now,
    );
  }

  setUp(() async {
    supportDir = await Directory.systemTemp.createTemp('history_support_');
    documentsDir = await Directory.systemTemp.createTemp('history_docs_');

    secureVault = <String, String>{};
    FlutterSecureStorage.setMockInitialValues(secureVault);

    mockPathProviderPlatform = MockPathProviderPlatform();
    PathProviderPlatform.instance = mockPathProviderPlatform;

    when(() => mockPathProviderPlatform.getApplicationSupportPath())
        .thenAnswer((_) async => supportDir.path);
    when(() => mockPathProviderPlatform.getApplicationDocumentsPath())
        .thenAnswer((_) async => documentsDir.path);

    datasource = createDatasource();
  });

  tearDown(() async {
    if (await supportDir.exists()) {
      await supportDir.delete(recursive: true);
    }
    if (await documentsDir.exists()) {
      await documentsDir.delete(recursive: true);
    }
  });

  group('ChatHistoryDatasourceImpl', () {
    test('loadConversations returns empty list when no files exist', () async {
      final result = await datasource.loadConversations();
      expect(result, isEmpty);
    });

    test('saveConversations persists encrypted data in support directory only', () async {
      final convs = [buildSampleConversation(title: 'Secret Plan')];

      await datasource.saveConversations(convs);

      final encryptedFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.encryptedHistoryFileName}',
      );
      final legacyDocsFile = File(
        '${documentsDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );
      final legacySupportFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );

      expect(await encryptedFile.exists(), isTrue);
      expect(await legacyDocsFile.exists(), isFalse);
      expect(await legacySupportFile.exists(), isFalse);

      final rawBytes = await encryptedFile.readAsBytes();
      expect(
        rawBytes.length,
        greaterThan(AesGcmEncryptionService.minimumPayloadLengthBytes),
      );
      final rawText = String.fromCharCodes(rawBytes);
      expect(rawText, isNot(contains('Secret Plan')));
      expect(rawText, isNot(contains('Hi there!')));

      final reloadedDatasource = createDatasource();
      final loaded = await reloadedDatasource.loadConversations();

      expect(loaded.length, 1);
      expect(loaded.first.id, '1');
      expect(loaded.first.title, 'Secret Plan');
      expect(loaded.first.messages.length, 2);
      expect(loaded.first.messages.first.text, 'Hello');
      expect(loaded.first.messages.last.text, 'Hi there!');
    });

    test('saveConversations uses a new random nonce on every write', () async {
      final convs = [buildSampleConversation()];

      await datasource.saveConversations(convs);
      final encryptedFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.encryptedHistoryFileName}',
      );
      final firstWrite = await encryptedFile.readAsBytes();

      await datasource.saveConversations(convs);
      final secondWrite = await encryptedFile.readAsBytes();

      expect(firstWrite, isNot(equals(secondWrite)));
      expect(
        firstWrite.sublist(0, AesGcmEncryptionService.nonceLengthBytes),
        isNot(
          equals(
            secondWrite.sublist(0, AesGcmEncryptionService.nonceLengthBytes),
          ),
        ),
      );
    });

    test('initialize migrates legacy JSON from documents directory and deletes plaintext file', () async {
      final legacyConv = buildSampleConversation(
        id: 'legacy-docs',
        title: 'Legacy Docs Chat',
      );
      final legacyFile = File(
        '${documentsDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );
      await legacyFile.writeAsString(jsonEncode([legacyConv.toJson()]));

      await datasource.initialize();

      final encryptedFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.encryptedHistoryFileName}',
      );
      expect(await encryptedFile.exists(), isTrue);
      expect(await legacyFile.exists(), isFalse);

      final loaded = await datasource.loadConversations();
      expect(loaded.length, 1);
      expect(loaded.first.id, 'legacy-docs');
      expect(loaded.first.title, 'Legacy Docs Chat');
      expect(loaded.first.messages.last.text, 'Hi there!');
    });

    test('loadConversations migrates legacy JSON from support directory and deletes plaintext file', () async {
      final legacyConv = buildSampleConversation(
        id: 'legacy-support',
        title: 'Legacy Support Chat',
      );
      final legacyFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );
      await legacyFile.writeAsString(jsonEncode([legacyConv.toJson()]));

      final loaded = await datasource.loadConversations();

      final encryptedFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.encryptedHistoryFileName}',
      );
      expect(await encryptedFile.exists(), isTrue);
      expect(await legacyFile.exists(), isFalse);
      expect(loaded.length, 1);
      expect(loaded.first.id, 'legacy-support');
      expect(loaded.first.title, 'Legacy Support Chat');
    });

    test('migrates and deduplicates legacy JSON files present in both directories', () async {
      final supportConv = buildSampleConversation(
        id: 'shared-id',
        title: 'From Support',
      );
      final docsUniqueConv = buildSampleConversation(
        id: 'docs-only',
        title: 'From Docs',
      );
      final docsDuplicateConv = buildSampleConversation(
        id: 'shared-id',
        title: 'From Docs Duplicate',
      );

      final legacySupportFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );
      final legacyDocsFile = File(
        '${documentsDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );

      await legacySupportFile.writeAsString(jsonEncode([supportConv.toJson()]));
      await legacyDocsFile.writeAsString(
        jsonEncode([docsUniqueConv.toJson(), docsDuplicateConv.toJson()]),
      );

      await datasource.initialize();

      expect(await legacySupportFile.exists(), isFalse);
      expect(await legacyDocsFile.exists(), isFalse);

      final loaded = await datasource.loadConversations();
      expect(loaded.map((c) => c.id), ['shared-id', 'docs-only']);
      expect(loaded.first.title, 'From Support');
    });

    test('preserves existing valid encrypted history and deletes stale legacy JSON', () async {
      final currentConv = buildSampleConversation(
        id: 'encrypted-1',
        title: 'Current Encrypted Chat',
      );
      await datasource.saveConversations([currentConv]);

      final staleLegacyFile = File(
        '${documentsDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );
      final staleConv = buildSampleConversation(
        id: 'stale-1',
        title: 'Stale Legacy Chat',
      );
      await staleLegacyFile.writeAsString(jsonEncode([staleConv.toJson()]));

      final freshDatasource = createDatasource();
      final loaded = await freshDatasource.loadConversations();

      expect(await staleLegacyFile.exists(), isFalse);
      expect(loaded.length, 1);
      expect(loaded.first.id, 'encrypted-1');
      expect(loaded.first.title, 'Current Encrypted Chat');
    });

    test('does not delete legacy JSON if key storage fails during migration', () async {
      final legacyConv = buildSampleConversation(
        id: 'protected-legacy',
        title: 'Must Not Be Lost',
      );
      final legacyFile = File(
        '${documentsDir.path}/${ChatHistoryDatasourceImpl.legacyHistoryFileName}',
      );
      await legacyFile.writeAsString(jsonEncode([legacyConv.toJson()]));

      final failingStorage = MockFlutterSecureStorage();
      when(() => failingStorage.read(key: any(named: 'key')))
          .thenThrow(Exception('Keyring locked'));

      final failingDatasource = ChatHistoryDatasourceImpl(
        encryptionService: AesGcmEncryptionService(
          keyManager: EncryptionKeyManager(secureStorage: failingStorage),
        ),
      );

      await expectLater(
        failingDatasource.initialize(),
        throwsException,
      );

      expect(await legacyFile.exists(), isTrue);
      final encryptedFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.encryptedHistoryFileName}',
      );
      expect(await encryptedFile.exists(), isFalse);
    });

    test('loadConversations returns empty list when encrypted file is tampered with', () async {
      await datasource.saveConversations([buildSampleConversation()]);

      final encryptedFile = File(
        '${supportDir.path}/${ChatHistoryDatasourceImpl.encryptedHistoryFileName}',
      );
      final bytes = await encryptedFile.readAsBytes();
      bytes[bytes.length - 1] ^= 0xFF;
      await encryptedFile.writeAsBytes(bytes);

      final result = await datasource.loadConversations();
      expect(result, isEmpty);
    });
  });
}
