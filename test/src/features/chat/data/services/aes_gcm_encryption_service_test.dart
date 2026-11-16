import 'dart:convert';
import 'dart:typed_data';
import 'package:cryptography/cryptography.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/services/aes_gcm_encryption_service.dart';
import 'package:my_hero/src/features/chat/data/services/encryption_key_manager.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('AesGcmEncryptionService', () {
    late Map<String, String> vault;
    late EncryptionKeyManager keyManager;
    late AesGcmEncryptionService service;

    setUp(() {
      vault = <String, String>{};
      FlutterSecureStorage.setMockInitialValues(vault);
      keyManager = EncryptionKeyManager();
      service = AesGcmEncryptionService(keyManager: keyManager);
    });

    test('encryptString and decryptString round-trip UTF-8 content accurately', () async {
      const plaintext =
          '[{"id":"1","title":"Conversa secreta 🚀","text":"Olá, mundo!"}]';

      final encrypted = await service.encryptString(plaintext);
      final decrypted = await service.decryptString(encrypted);

      expect(decrypted, equals(plaintext));
    });

    test('stores payload as 12-byte nonce + ciphertext + 16-byte MAC', () async {
      final plaintextBytes = utf8.encode('sensitive conversation history');

      final encrypted = await service.encrypt(plaintextBytes);

      expect(
        encrypted.length,
        equals(
          AesGcmEncryptionService.nonceLengthBytes +
              plaintextBytes.length +
              AesGcmEncryptionService.macLengthBytes,
        ),
      );
    });

    test('generates a fresh random 12-byte nonce on every encryption', () async {
      const plaintext = 'identical plaintext payload';

      final firstEncrypted = await service.encryptString(plaintext);
      final secondEncrypted = await service.encryptString(plaintext);

      final firstNonce =
          firstEncrypted.sublist(0, AesGcmEncryptionService.nonceLengthBytes);
      final secondNonce =
          secondEncrypted.sublist(0, AesGcmEncryptionService.nonceLengthBytes);

      expect(firstNonce, isNot(equals(secondNonce)));
      expect(firstEncrypted, isNot(equals(secondEncrypted)));
      expect(await service.decryptString(firstEncrypted), equals(plaintext));
      expect(await service.decryptString(secondEncrypted), equals(plaintext));
    });

    test('throws SecretBoxAuthenticationError when ciphertext or MAC is tampered with', () async {
      final encrypted = await service.encryptString('authenticated payload');
      final tampered = Uint8List.fromList(encrypted);
      tampered[AesGcmEncryptionService.nonceLengthBytes + 2] ^= 0xFF;

      await expectLater(
        service.decryptString(tampered),
        throwsA(isA<SecretBoxAuthenticationError>()),
      );
    });

    test('throws SecretBoxAuthenticationError when decrypting with a different key', () async {
      final encrypted = await service.encryptString('bound to original key');

      final differentVault = <String, String>{};
      FlutterSecureStorage.setMockInitialValues(differentVault);
      final otherService = AesGcmEncryptionService(
        keyManager: EncryptionKeyManager(),
      );

      await expectLater(
        otherService.decryptString(encrypted),
        throwsA(isA<SecretBoxAuthenticationError>()),
      );
    });

    test('throws FormatException when encrypted payload is shorter than minimum length', () async {
      final tooShort = Uint8List(
        AesGcmEncryptionService.minimumPayloadLengthBytes - 1,
      );

      await expectLater(
        service.decrypt(tooShort),
        throwsA(isA<FormatException>()),
      );
    });
  });
}
