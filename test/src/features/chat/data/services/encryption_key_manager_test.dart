import 'dart:convert';
import 'package:cryptography/cryptography.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/services/encryption_key_manager.dart';

class MockFlutterSecureStorage extends Mock implements FlutterSecureStorage {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('EncryptionKeyManager', () {
    late Map<String, String> backingVault;

    setUp(() {
      backingVault = <String, String>{};
      FlutterSecureStorage.setMockInitialValues(backingVault);
    });

    test('generates and persists a 256-bit Base64 key when none exists', () async {
      final manager = EncryptionKeyManager();

      final secretKey = await manager.getOrCreateKey();
      final keyBytes = await secretKey.extractBytes();

      expect(keyBytes.length, EncryptionKeyManager.keyLengthBytes);

      final storedValue = backingVault[EncryptionKeyManager.defaultStorageKey];
      expect(storedValue, isNotNull);
      expect(base64Decode(storedValue!), equals(keyBytes));
    });

    test('reuses the existing master key across subsequent instances', () async {
      final firstManager = EncryptionKeyManager();
      final firstKeyBytes = await (await firstManager.getOrCreateKey()).extractBytes();

      final secondManager = EncryptionKeyManager();
      final secondKeyBytes =
          await (await secondManager.getOrCreateKey()).extractBytes();

      expect(secondKeyBytes, equals(firstKeyBytes));
      expect(
        backingVault.keys,
        contains(EncryptionKeyManager.defaultStorageKey),
      );
    });

    test('deduplicates concurrent initialization calls without overwriting key', () async {
      final mockStorage = MockFlutterSecureStorage();
      String? storedValue;

      when(() => mockStorage.read(key: EncryptionKeyManager.defaultStorageKey))
          .thenAnswer((_) async {
        await Future<void>.delayed(const Duration(milliseconds: 10));
        return storedValue;
      });
      when(
        () => mockStorage.write(
          key: EncryptionKeyManager.defaultStorageKey,
          value: any(named: 'value'),
        ),
      ).thenAnswer((invocation) async {
        storedValue = invocation.namedArguments[#value] as String?;
      });

      final manager = EncryptionKeyManager(secureStorage: mockStorage);

      final results = await Future.wait<SecretKey>([
        manager.getOrCreateKey(),
        manager.getOrCreateKey(),
        manager.getOrCreateKey(),
      ]);

      final bytes0 = await results[0].extractBytes();
      final bytes1 = await results[1].extractBytes();
      final bytes2 = await results[2].extractBytes();

      expect(bytes1, equals(bytes0));
      expect(bytes2, equals(bytes0));
      verify(() => mockStorage.read(key: EncryptionKeyManager.defaultStorageKey))
          .called(1);
      verify(
        () => mockStorage.write(
          key: EncryptionKeyManager.defaultStorageKey,
          value: any(named: 'value'),
        ),
      ).called(1);
    });

    test('throws FormatException when stored key does not decode to 32 bytes', () async {
      backingVault[EncryptionKeyManager.defaultStorageKey] =
          base64Encode(List<int>.filled(16, 7));

      final manager = EncryptionKeyManager();

      await expectLater(
        manager.getOrCreateKey(),
        throwsA(isA<FormatException>()),
      );
    });
  });
}
