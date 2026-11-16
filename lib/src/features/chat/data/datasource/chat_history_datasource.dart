import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/services/aes_gcm_encryption_service.dart';
import 'package:my_hero/src/features/chat/data/services/encryption_key_manager.dart';
import 'package:path_provider/path_provider.dart';

/// Local persistence contract for chat conversations.
abstract class ChatHistoryDatasource {
  /// Initializes encryption keys and migrates any legacy plaintext JSON files.
  Future<void> initialize();

  /// Loads all persisted conversations from encrypted local storage.
  Future<List<Conversation>> loadConversations();

  /// Encrypts and persists [conversations] to local storage.
  Future<void> saveConversations(List<Conversation> conversations);
}

/// Encrypted local storage implementation for [ChatHistoryDatasource].
///
/// Conversations are serialized as JSON, encrypted using AES-256-GCM with a
/// fresh 12-byte random nonce per write, and stored as `nonce + ciphertext + mac`
/// inside `getApplicationSupportDirectory()`. Legacy plaintext `.json` files in
/// both the previous documents directory and the current support directory are
/// automatically migrated and deleted.
class ChatHistoryDatasourceImpl implements ChatHistoryDatasource {
  /// File name used for the AES-256-GCM encrypted conversation history.
  static const String encryptedHistoryFileName = 'chat_conversations.enc';

  /// Legacy plaintext file name migrated automatically on initialization.
  static const String legacyHistoryFileName = 'chat_conversations.json';

  final AesGcmEncryptionService _encryptionService;
  final Future<Directory> Function() _supportDirectoryProvider;
  final Future<Directory?> Function() _documentsDirectoryProvider;

  bool _isInitialized = false;
  Future<void> _pendingOperation = Future<void>.value();

  ChatHistoryDatasourceImpl({
    AesGcmEncryptionService? encryptionService,
    Future<Directory> Function()? supportDirectoryProvider,
    Future<Directory?> Function()? documentsDirectoryProvider,
  })  : _encryptionService = encryptionService ??
            AesGcmEncryptionService(
              keyManager: EncryptionKeyManager(),
            ),
        _supportDirectoryProvider =
            supportDirectoryProvider ?? getApplicationSupportDirectory,
        _documentsDirectoryProvider =
            documentsDirectoryProvider ?? _defaultDocumentsDirectoryProvider;

  static Future<Directory?> _defaultDocumentsDirectoryProvider() async {
    try {
      return await getApplicationDocumentsDirectory();
    } catch (_) {
      return null;
    }
  }

  @override
  Future<void> initialize() {
    return _synchronized(_ensureInitializedUnlocked);
  }

  @override
  Future<List<Conversation>> loadConversations() {
    return _synchronized(() async {
      try {
        await _ensureInitializedUnlocked();
        final file = await _getEncryptedFile();
        if (!await file.exists()) {
          return <Conversation>[];
        }
        return await _readEncryptedConversationsUnlocked(file);
      } catch (_) {
        return <Conversation>[];
      }
    });
  }

  @override
  Future<void> saveConversations(List<Conversation> conversations) {
    return _synchronized(() async {
      try {
        await _ensureInitializedUnlocked();
        await _writeEncryptedConversationsUnlocked(conversations);
      } catch (_) {
        // Ignore unexpected write failures so active UI chat flows stay resilient.
      }
    });
  }

  Future<T> _synchronized<T>(Future<T> Function() action) {
    final completer = Completer<T>();
    _pendingOperation = _pendingOperation.then((_) async {
      try {
        completer.complete(await action());
      } catch (error, stackTrace) {
        completer.completeError(error, stackTrace);
      }
    });
    return completer.future;
  }

  Future<void> _ensureInitializedUnlocked() async {
    if (_isInitialized) return;
    await _encryptionService.initialize();
    await _migrateLegacyFilesUnlocked();
    _isInitialized = true;
  }

  Future<File> _getEncryptedFile() async {
    final dir = await _supportDirectoryProvider();
    return File(_buildFilePath(dir.path, encryptedHistoryFileName));
  }

  Future<List<File>> _getCandidateLegacyFiles() async {
    final supportDir = await _supportDirectoryProvider();
    final supportPath = _normalizeDirectoryPath(supportDir.path);
    final candidates = <File>[
      File(_buildFilePath(supportDir.path, legacyHistoryFileName)),
    ];

    final docsDir = await _documentsDirectoryProvider();
    if (docsDir != null) {
      final docsPath = _normalizeDirectoryPath(docsDir.path);
      if (docsPath != supportPath) {
        candidates.add(
          File(_buildFilePath(docsDir.path, legacyHistoryFileName)),
        );
      }
    }
    return candidates;
  }

  Future<void> _migrateLegacyFilesUnlocked() async {
    final existingLegacyFiles = <File>[];
    for (final candidate in await _getCandidateLegacyFiles()) {
      if (await candidate.exists()) {
        existingLegacyFiles.add(candidate);
      }
    }
    if (existingLegacyFiles.isEmpty) {
      return;
    }

    final encryptedFile = await _getEncryptedFile();
    final hasValidEncryptedFile = await _hasValidEncryptedFileUnlocked(
      encryptedFile,
    );

    if (hasValidEncryptedFile) {
      await _deleteFiles(existingLegacyFiles);
      return;
    }

    final migratedConversations = <Conversation>[];
    final seenIds = <String>{};
    final filesToDelete = <File>[];
    var hasParsedAtLeastOneLegacyFile = false;

    for (final legacyFile in existingLegacyFiles) {
      final content = await legacyFile.readAsString();
      final parsed = _parseLegacyConversations(content);
      if (parsed == null) {
        continue;
      }

      hasParsedAtLeastOneLegacyFile = true;
      filesToDelete.add(legacyFile);
      for (final conversation in parsed) {
        if (seenIds.add(conversation.id)) {
          migratedConversations.add(conversation);
        }
      }
    }

    if (!hasParsedAtLeastOneLegacyFile) {
      return;
    }

    // Persist the encrypted file before deleting any plaintext JSON file so a
    // failed encryption or disk write never causes data loss.
    await _writeEncryptedConversationsUnlocked(migratedConversations);
    await _deleteFiles(filesToDelete);
  }

  Future<bool> _hasValidEncryptedFileUnlocked(File encryptedFile) async {
    if (!await encryptedFile.exists() || await encryptedFile.length() == 0) {
      return false;
    }
    try {
      await _readEncryptedConversationsUnlocked(encryptedFile);
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<List<Conversation>> _readEncryptedConversationsUnlocked(
    File file,
  ) async {
    final encryptedBytes = await file.readAsBytes();
    if (encryptedBytes.isEmpty) {
      return <Conversation>[];
    }

    final decryptedJson = await _encryptionService.decryptString(encryptedBytes);
    if (decryptedJson.trim().isEmpty) {
      return <Conversation>[];
    }

    final decoded = jsonDecode(decryptedJson) as List<dynamic>;
    return decoded
        .map((item) => Conversation.fromJson(item as Map<String, dynamic>))
        .toList();
  }

  Future<void> _writeEncryptedConversationsUnlocked(
    List<Conversation> conversations,
  ) async {
    final file = await _getEncryptedFile();
    await file.parent.create(recursive: true);

    final serialized = jsonEncode(
      conversations.map((c) => c.toJson()).toList(),
    );
    final encryptedBytes = await _encryptionService.encryptString(serialized);

    final tempFile = File('${file.path}.tmp');
    try {
      await tempFile.writeAsBytes(encryptedBytes, flush: true);
      await tempFile.rename(file.path);
    } catch (_) {
      if (await tempFile.exists()) {
        try {
          await tempFile.delete();
        } catch (_) {}
      }
      rethrow;
    }
  }

  List<Conversation>? _parseLegacyConversations(String content) {
    if (content.trim().isEmpty) {
      return <Conversation>[];
    }
    try {
      final decoded = jsonDecode(content);
      if (decoded is! List<dynamic>) {
        return null;
      }
      return decoded
          .map((item) => Conversation.fromJson(item as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return null;
    }
  }

  Future<void> _deleteFiles(List<File> files) async {
    for (final file in files) {
      if (await file.exists()) {
        await file.delete();
      }
    }
  }

  String _buildFilePath(String directoryPath, String fileName) {
    final normalized = _normalizeDirectoryPath(directoryPath);
    if (normalized == Platform.pathSeparator) {
      return '$normalized$fileName';
    }
    return '$normalized${Platform.pathSeparator}$fileName';
  }

  String _normalizeDirectoryPath(String path) {
    if (path.length > 1 &&
        (path.endsWith('/') || path.endsWith(Platform.pathSeparator))) {
      return path.substring(0, path.length - 1);
    }
    return path;
  }
}
