import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:cryptography/cryptography.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Manages the 256-bit symmetric master key stored in the platform secure vault
/// (Android Keystore on Android and libsecret / GNOME Keyring on Linux).
class EncryptionKeyManager {
  /// Default key identifier used to persist the master key in secure storage.
  static const String defaultStorageKey = 'chat_history_master_key';

  /// Required master key length in bytes (256 bits).
  static const int keyLengthBytes = 32;

  final FlutterSecureStorage _secureStorage;
  final String _storageKey;
  final Random _random;

  Uint8List? _cachedKeyBytes;
  Future<SecretKey>? _inFlightKeyFuture;

  EncryptionKeyManager({
    FlutterSecureStorage? secureStorage,
    String storageKey = defaultStorageKey,
    Random? random,
  })  : _secureStorage = secureStorage ??
            const FlutterSecureStorage(
              aOptions: AndroidOptions(),
              lOptions: LinuxOptions(),
            ),
        _storageKey = storageKey,
        _random = random ?? Random.secure();

  /// Returns the existing 256-bit master key or generates, stores, and returns
  /// a new cryptographically random 32-byte key if none exists yet.
  Future<SecretKey> getOrCreateKey() async {
    final cached = _cachedKeyBytes;
    if (cached != null) {
      return SecretKey(Uint8List.fromList(cached));
    }

    final existingFuture = _inFlightKeyFuture;
    if (existingFuture != null) {
      return existingFuture;
    }

    final future = _loadOrGenerateKey();
    _inFlightKeyFuture = future;
    try {
      return await future;
    } finally {
      _inFlightKeyFuture = null;
    }
  }

  Future<SecretKey> _loadOrGenerateKey() async {
    final encodedKey = await _secureStorage.read(key: _storageKey);
    if (encodedKey != null && encodedKey.trim().isNotEmpty) {
      final decoded = base64Decode(encodedKey.trim());
      if (decoded.length != keyLengthBytes) {
        throw FormatException(
          'Invalid master key length: expected $keyLengthBytes bytes, got ${decoded.length}.',
        );
      }
      _cachedKeyBytes = Uint8List.fromList(decoded);
      return SecretKey(Uint8List.fromList(_cachedKeyBytes!));
    }

    final generatedBytes = _generateRandomBytes(keyLengthBytes);
    await _secureStorage.write(
      key: _storageKey,
      value: base64Encode(generatedBytes),
    );
    _cachedKeyBytes = Uint8List.fromList(generatedBytes);
    return SecretKey(Uint8List.fromList(_cachedKeyBytes!));
  }

  Uint8List _generateRandomBytes(int length) {
    final bytes = Uint8List(length);
    for (var i = 0; i < length; i++) {
      bytes[i] = _random.nextInt(256);
    }
    return bytes;
  }
}
