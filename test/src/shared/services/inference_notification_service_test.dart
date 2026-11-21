import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/shared/services/inference_notification_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const channel = MethodChannel(PlatformInferenceNotificationService.channelName);
  late List<MethodCall> recordedCalls;

  setUp(() {
    recordedCalls = [];
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      recordedCalls.add(call);
      return null;
    });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  group('PlatformInferenceNotificationService', () {
    test('should invoke native MethodChannel methods on Android', () async {
      final service = PlatformInferenceNotificationService(
        channel: channel,
        platform: TargetPlatform.android,
      );

      await service.start(title: 'My Hero', body: 'Pensando na resposta...');
      await service.update(title: 'My Hero', body: 'Gerando resposta...');
      await service.stop(
        notifyIfBackgrounded: true,
        completedTitle: 'Resposta pronta',
        completedBody: 'Resultado final',
      );
      await service.clearCompletedNotification();

      expect(recordedCalls.map((c) => c.method), [
        'start',
        'update',
        'stop',
        'clearCompletedNotification',
      ]);
      expect(recordedCalls[0].arguments, {
        'title': 'My Hero',
        'body': 'Pensando na resposta...',
      });
      expect(recordedCalls[2].arguments, {
        'notifyIfBackgrounded': true,
        'completedTitle': 'Resposta pronta',
        'completedBody': 'Resultado final',
      });
    });

    test('should forward native onStopRequested callback to Dart listener', () async {
      final service = PlatformInferenceNotificationService(
        channel: channel,
        platform: TargetPlatform.android,
      );

      var stopCalled = false;
      service.setOnStopRequested(() async {
        stopCalled = true;
      });

      final codec = const StandardMethodCodec();
      final encoded = codec.encodeMethodCall(const MethodCall('onStopRequested'));
      await TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .handlePlatformMessage(
        PlatformInferenceNotificationService.channelName,
        encoded,
        (_) {},
      );

      expect(stopCalled, isTrue);
    });

    test('should be a safe no-op on Linux and Windows', () async {
      for (final desktop in [TargetPlatform.linux, TargetPlatform.windows]) {
        final service = PlatformInferenceNotificationService(
          channel: channel,
          platform: desktop,
        );

        await service.start(title: 'My Hero', body: 'Gerando...');
        await service.update(title: 'My Hero', body: 'Atualizando...');
        await service.stop(notifyIfBackgrounded: true);
        await service.clearCompletedNotification();
      }

      expect(recordedCalls, isEmpty);
    });
  });
}
