import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/shared/services/device_resources_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const channel = MethodChannel(PlatformDeviceResourcesService.channelName);

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  group('PlatformDeviceResourcesService', () {
    test('returns null without invoking channel on non-Android platforms', () async {
      var invoked = false;
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (call) async {
        invoked = true;
        return null;
      });

      final service = PlatformDeviceResourcesService(
        platform: TargetPlatform.linux,
      );

      final result = await service.getDeviceResources();

      expect(result, isNull);
      expect(invoked, isFalse);
    });

    test('parses RAM and storage values on Android', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (call) async {
        expect(call.method, 'getDeviceResources');
        return <String, Object?>{
          'totalRamBytes': 4 * 1024 * 1024 * 1024,
          'availableRamBytes': 1800 * 1024 * 1024,
          'lowMemory': false,
          'memoryThresholdBytes': 256 * 1024 * 1024,
          'freeStorageBytes': 12 * 1024 * 1024 * 1024,
        };
      });

      final service = PlatformDeviceResourcesService(
        platform: TargetPlatform.android,
      );

      final info = await service.getDeviceResources();

      expect(info, isNotNull);
      expect(info!.totalRamBytes, 4 * 1024 * 1024 * 1024);
      expect(info.availableRamBytes, 1800 * 1024 * 1024);
      expect(info.lowMemory, isFalse);
      expect(info.freeStorageBytes, 12 * 1024 * 1024 * 1024);
      expect(info.hasRamInfo, isTrue);
      expect(info.hasStorageInfo, isTrue);
      expect(info.isLowRamDevice, isTrue);
    });

    test('returns null cleanly when channel throws PlatformException', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (call) async {
        throw PlatformException(code: 'UNAVAILABLE', message: 'error');
      });

      final service = PlatformDeviceResourcesService(
        platform: TargetPlatform.android,
      );

      final info = await service.getDeviceResources();
      expect(info, isNull);
    });
  });
}
