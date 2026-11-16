import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:cryptography/cryptography.dart';
import 'package:my_hero/src/features/chat/data/services/encryption_key_manager.dart';

/// Provides authenticated symmetric encryption and decryption using AES-256-GCM.
///
/// Encrypted payloads follow the binary layout:
/// `nonce (12 bytes) + ciphertext + mac (16 bytes)`.
class AesGcmEncryptionService {
  /// Nonce (IV) length in bytes for AES-GCM.
  static const int nonceLengthBytes = 12;

  /// Authentication tag (MAC) length in bytes for AES-GCM.
  static const int macLengthBytes = 16;

  /// Minimum valid payload length (`nonce + mac`).
  static const int minimumPayloadLengthBytes =
      nonceLengthBytes + macLengthBytes;

  final EncryptionKeyManager _keyManager;
  final AesGcm _cipher;
  final Random _random;

  AesGcmEncryptionService({
    required EncryptionKeyManager keyManager,
    AesGcm? cipher,
    Random? random,
  })  : _keyManager = keyManager,
        _cipher = cipher ?? AesGcm.with256bits(),
        _random = random ?? Random.secure();

  /// Ensures the underlying master encryption key is loaded or created.
  Future<void> initialize() async {
    await _keyManager.getOrCreateKey();
  }

  /// Encrypts [plaintextBytes] using AES-256-GCM with a fresh random 12-byte nonce
  /// and returns `nonce + ciphertext + mac`.
  Future<Uint8List> encrypt(List<int> plaintextBytes) async {
    final secretKey = await _keyManager.getOrCreateKey();
    final nonce = _generateNonce();
    final secretBox = await _cipher.encrypt(
      plaintextBytes,
      secretKey: secretKey,
      nonce: nonce,
    );
    return secretBox.concatenation();
  }

  /// Encrypts a UTF-8 [plaintext] string and returns `nonce + ciphertext + mac`.
  Future<Uint8List> encryptString(String plaintext) {
    return encrypt(utf8.encode(plaintext));
  }

  /// Authenticates and decrypts an [encryptedBytes] payload formatted as
  /// `nonce (12 bytes) + ciphertext + mac (16 bytes)`.
  Future<Uint8List> decrypt(List<int> encryptedBytes) async {
    if (encryptedBytes.length < minimumPayloadLengthBytes) {
      throw FormatException(
        'Encrypted payload is too short: expected at least $minimumPayloadLengthBytes bytes, got ${encryptedBytes.length}.',
      );
    }

    final secretKey = await _keyManager.getOrCreateKey();
    final secretBox = SecretBox.fromConcatenation(
      encryptedBytes,
      nonceLength: nonceLengthBytes,
      macLength: macLengthBytes,
      copy: false,
    );
    final clearBytes = await _cipher.decrypt(
      secretBox,
      secretKey: secretKey,
    );
    return Uint8List.fromList(clearBytes);
  }

  /// Authenticates and decrypts an [encryptedBytes] payload into a UTF-8 string.
  Future<String> decryptString(List<int> encryptedBytes) async {
    final clearBytes = await decrypt(encryptedBytes);
    return utf8.decode(clearBytes);
  }

  Uint8List _generateNonce() {
    final nonce = Uint8List(nonceLengthBytes);
    for (var i = 0; i < nonceLengthBytes; i++) {
      nonce[i] = _random.nextInt(256);
    }
    return nonce;
  }
}
